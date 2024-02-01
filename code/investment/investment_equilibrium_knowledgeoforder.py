# %%
# Import packages
import numpy as onp # the "OG" numpy that we will need for some indexing functions
import autograd.numpy as np # autograd's version of numpy
from autograd.scipy.special import logsumexp
from scipy import sparse

from autograd.extend import primitive, defvjp

from itertools import permutations

import gc
import time as time

# %%
# Define functions used in the value functions

# Define take_along_axis same as numpy.take_along_axis in order to make work with autograd, which does not have the VJP defined for take_along_axis
def take_along_axis(arr, indices, axis):
    """
        Copied over from the original numpy code for numpy.take_along_axis
    """
    # normalize inputs
    if axis is None:
        arr = arr.flat
        arr_shape = (len(arr),)  # flatiter has no .shape
        axis = 0
    else:
        axis = onp.core.multiarray.normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # use the fancy index
    return arr[onp.lib.shape_base._make_along_axis_idx(arr_shape, indices, axis)]

# 
@primitive
def nanmax(arr, axis=None, keepdims=onp._NoValue):
    """
        Define nanmax using non-autograd numpy
    """
    return onp.nanmax(arr, axis=axis, keepdims=keepdims)

# Define the nanmax function using autograd's grad_chooser (same as for max)
defvjp(nanmax, np.numpy_vjps.grad_chooser)

# %%
# Define value functions

# Determine the value in a period given the value in the next period
def value_fct(firm_payoff, adjustment_cost, indices_unraveled, dims, dims_correspondence, row_policy_fcts, col_policy_fcts, firm_orders, pr_firm_orders, firm_selection=None, take_expectation=None):
    """
        Return the value and policy functions for each firm within a set
    
    Parameters
    ----------
        firm_payoff : ndarray
            (N,F) array of wholesale profits + capacity payments + value functions in next period for each firm
        adjustment_cost : ndarray
            (N,D,3) array of what the adjustment cost would be if we made an adjustment from that state to the next in that dimension
        indices_unraveled : ndarray
            (D,N) unraveled indices for the state
        dims : ndarray
            (D,) array of size of each dimension
        dims_correspondence : ndarray
            (F,D) array of bools where each row corresponds to firm and columns to dimension, entry tells us which firm in question controls that dimension
        firm_orders : ndarray
            (O_F, F) array where each row corresponds to a permutation of firm adjustment, columns correspond to order of adjustment
        pr_firm_orders : ndarray
            (O_F,) array of probability of each possible permutation
        firm_selection : ndarray
            (N,F,3) array of identity of the decision-making firm in each state and choice
        take_expectation : ndarray
            (N,F') array that we wish to take the expectation of based on policy functions in this stage (eg, of firms not adjusting in this stage)

    Returns
    -------
        v_t : ndarray
            (N,F) array of value functions in next period for each firm
        probability_adjustment : ndarray
            (N,N_possible_set) array of choice probabilities
    """

    # Initialize adjustment probabilities and value functions
    probability_adjustment = {}
    for j in range(dims_correspondence.shape[0]):
        num_firm_options = 3**np.sum(dims_correspondence[j,:])
        probability_adjustment[f"{j}"] = np.zeros((firm_payoff.shape[0], num_firm_options)) # this is the probability that a firm adjusts to a state based on state before *anyone* has adjusted, will fill in as we go
    v_t = np.zeros(firm_payoff.shape)
    if take_expectation is not None:
        take_expectation_result = np.zeros(take_expectation.shape)
    
    # Pre-compute various indexing for each firm
    indices_use = {}
    indices_raveled = {}
    vals_should_be_nan = {}
    nonnanvals = {}
    vals_should_be_zero_conversion = {}
    indices_after_adjustment = {}
    firm_options_adjustment_costs = {}
    adjusting_firm_onesandzeros = {}
    state0alongdim_onesandzeros = {}
    statem1alongdim_onesandzeros = {}
    stateinterioralongdim_onesandzeros = {}
    for j in range(dims_correspondence.shape[0]):
    
        # Determine indices of the states the firm could choose to move to in this period given current state
        dims_firm_changes = dims_correspondence[j,:] # which dimensions can this firm adjust
        num_dims_firm_changes = np.sum(dims_firm_changes)
        if (num_dims_firm_changes > 1) and (firm_selection is not None):
            raise ValueError(f"If firm_selection is provided, each source can only correspond to one dimension.")
        num_firm_options = 3**num_dims_firm_changes # number of possible options given number of dimensions the firm can change
        add_vals_j = np.zeros((indices_unraveled.shape[0],num_firm_options), dtype=int) # initialize array of what firm's options are in each period
        index_adjustments_to_firms_dims = np.array(list(np.unravel_index(np.arange(num_firm_options), tuple([3 for i in range(num_dims_firm_changes)])))) - 1 # flattened version of all the possible ways in which firm can make adjustments in that dimension, -1 gives us -1, 0 and 1 (instead of 0,1,2)
        add_vals_j[dims_firm_changes,:] = index_adjustments_to_firms_dims
#         init_probability_j = np.zeros((firm_payoff.shape[0], num_firm_options))
#         no_adjustment_idx_j = np.where(np.all(index_adjustments_to_firms_dims == 0, axis=0))[0][0]
#         init_probability_j[:,no_adjustment_idx_j] = 1.0 # if no one adjusts after you, you just stay the same
#         probability_adjustment_dict[f'{j},{combo_no_firms}'] = init_probability_j
        indices_use[f'{j}'] = indices_unraveled[:,:,np.newaxis] + add_vals_j[:,np.newaxis,:] # take the unraveled indices and add the no adjustment / adjustment in that dimension
        indices_raveled[f'{j}'] = np.ravel_multi_index(indices_use[f'{j}'], dims, mode="wrap") # ravel the indices, some will be beyond limit of that dimension, "wrap" them (just set to highest in that dimension), we'll deal with in following lines
        ones_and_nans_j = np.ones(indices_raveled[f'{j}'].shape)
        ones_and_nans_j[np.any(indices_use[f'{j}'] >= dims[:,np.newaxis,np.newaxis], axis=0)] = np.nan # replace entries w/ NaN if we exceeded that dimension from above
        ones_and_nans_j[np.any(indices_use[f'{j}'] < 0, axis=0)] = np.nan # replace entries w/ NaN if we exceeded that dimension from below
        vals_should_be_nan[f'{j}'] = ones_and_nans_j
        nonnanvals[f'{j}'] = np.nan_to_num(vals_should_be_nan[f'{j}'])
        
        # Determine firm adjustment costs
        firm_options_adjustment_costs[f'{j}'] = np.sum(np.take_along_axis(adjustment_cost, add_vals_j[np.newaxis,:,:] + 1, axis=2), axis=1) # returns N x num_firm_options, takes the adjustment_cost values based on indexing from last row according to add_vals_j + 1 (so 0 corresponds to negative change, 1 no change, 2 positive change), then sums across all of the dimensions, so there are potentially multiple groupings of generators when we have >1 dimensions a firm can adjust
        
        # Ones and zeros for value function calculation (whether or not we are inputting the firm in question)
        if firm_selection is None:
            onesandzeros_adjusting = np.zeros((firm_payoff.shape[0], firm_payoff.shape[1])) # N x number of firms x number of options
            onesandzeros_adjusting[:,j] = 1.0 # if it is the firm in question, then we're going to make it 1
            adjusting_firm_onesandzeros[f'{j}'] = onesandzeros_adjusting
        else:
            onesandzeros_adjusting = np.zeros((firm_payoff.shape[0], firm_payoff.shape[1], 2)) # N x number of firms x number of options
            onesandzeros_adjusting[np.arange(firm_payoff.shape[1])[np.newaxis,:,np.newaxis] == firm_selection[:,np.arange(firm_selection.shape[1]) == j,:]] = 1.0 # if it is the firm in question, then we're going to make it 1
            adjusting_firm_onesandzeros[f'{j}'] = onesandzeros_adjusting
            
        # Ones and zeros for whether we are in the first or last state along dimension j
        if firm_selection is not None: # we only use this if we are doing firm selection
            j_idx = np.where(dims_correspondence[j,:])[0][0] # this is the location of the index in question
            state0alongdim_onesandzeros_j = indices_unraveled[j_idx,:] == 0 # indices in which we are in the first location along this dimension
            statem1alongdim_onesandzeros_j = indices_unraveled[j_idx,:] == indices_unraveled[j_idx,-1] # indices in which we are in the last location along this dimension; based on ordering of indices_unraveled, we are in the last index along all dimension in the final column
            state0alongdim_onesandzeros[f'{j}'] = 1.0 * state0alongdim_onesandzeros_j
            statem1alongdim_onesandzeros[f'{j}'] = 1.0 * statem1alongdim_onesandzeros_j
            stateinterioralongdim_onesandzeros[f'{j}'] = 1.0 - (state0alongdim_onesandzeros_j | statem1alongdim_onesandzeros_j) # when in the interior of the state
    
    # Determine which firms would never adjust given the initial state
    if firm_selection is not None:
        nonadjusting_firms_onesandzeros = np.ones((firm_payoff.shape[0], firm_payoff.shape[1]))
        nonadjusting_firms_onesandzeros[np.any(np.arange(firm_payoff.shape[1])[np.newaxis,np.newaxis,np.newaxis,:] == firm_selection[:,:,:,np.newaxis], axis=(1,2))] = 0.0 # if it is a firm that adjusted at some point, we're going to make it 0
    
    # Go through each possible order in which firms move
    for i in range(firm_orders.shape[0]):
        # Save what each firm's payoff is, we will update this later as we go through the ordering of firms
        firm_payoff_i = np.copy(firm_payoff)
        
        # Initialize which firms have adjusted
        firms_already_adjusted = []
        
        # Initialize adjustment probabilities
        probability_adjustment_i = {} # don't need to add, we will do that later
        
        # Initialize firms' value functions
        v_t_i = np.zeros(v_t.shape)
        
        # Initialize whatever taking expectation over
        if take_expectation is not None:
            take_expectation_i = np.copy(take_expectation)
        
        # Go through each firm (backward) in the ordering and determine what adjustment probabilities are
        for j in range(firm_orders.shape[1] - 1, -1, -1):
            #print(f"\tcompleting order {i+1}/{firm_orders.shape[0]} and firm {j + 1}...", flush=True)
            # Select the firm being adjusted here
            firm_idx = firm_orders[i,j]
            
            # Create array with all of firm firm_idx's options and associated payoffs
            if firm_selection is None:
                # Set up payoff options
                firm_payoff_options = firm_payoff_i[:,firm_idx][indices_raveled[f'{firm_idx}']] # state_space_size x num_dims_firm_changes
                firm_payoff_options = firm_payoff_options * vals_should_be_nan[f'{firm_idx}'] # makes the actual change to NaNs (done in this way b/c compatible with autograd, direct replacement isn't)
                firm_payoff_options = firm_payoff_options - firm_options_adjustment_costs[f'{firm_idx}'] # subtract cost of adjusting

                # Determine probability that firm firm_idx adjusts to each state
                max_firm_payoff_options = nanmax(firm_payoff_options, axis=1, keepdims=True)
                exp_firm_payoff_options = np.exp(firm_payoff_options - max_firm_payoff_options) # subtracting maximum has better numerical properties when numbers are very different
                policy_fcts = np.nan_to_num(exp_firm_payoff_options / np.sum(np.nan_to_num(exp_firm_payoff_options), axis=1, keepdims=True)) # np.sum(np.nan_to_num(.)) is equivalent to np.nansum, doing it this way b/c autograd doesn't work well with nansum

                # Calculate expected maximum
                v_t_i_j = logsumexp(np.nan_to_num(firm_payoff_options), axis=1, b=nonnanvals[f'{firm_idx}']) # using nonnanvals as weights is the same as doing nanlogsumexp (which doesn't exist)
                
            # Consider the firm that would go second along the dimension if first firm doesn't go backward
            else:
                # Set up payoff options
                indices_raveled_select_firm = indices_raveled[f'{firm_idx}'][:,1:] # select which indices are relevant to the second firm moving
                rows_firm_payoff_options = np.reshape(indices_raveled_select_firm, (-1,)) # flatten array because if doing fancy indexing all indices must be 1-d
                cols_firm_payoff_options = np.repeat(firm_selection[:,firm_idx,-1], indices_raveled_select_firm.shape[1]) # take the correct firm depending on the state and firm_idx according to firm selection, repeated b/c we can't do broadcasting along the second dimension in the fancy indexing below, need to make explicit since flattening
                firm_payoff_options = np.reshape(firm_payoff_i[rows_firm_payoff_options,cols_firm_payoff_options], indices_raveled_select_firm.shape) # select the correct entries from firm_payoff_dict, then reshape so that it's num_states x num_options
                del indices_raveled_select_firm, rows_firm_payoff_options, cols_firm_payoff_options # don't need these anymore, remove to save memory
                firm_payoff_options = firm_payoff_options * vals_should_be_nan[f'{firm_idx}'][:,1:] # makes the actual change to NaNs (done in this way b/c compatible with autograd, direct replacement isn't)
                firm_payoff_options = firm_payoff_options - firm_options_adjustment_costs[f'{firm_idx}'][:,1:] # subtract cost of adjusting

                # Determine probability that firm firm_idx adjusts to each state
                max_firm_payoff_options = nanmax(firm_payoff_options, axis=1, keepdims=True)
                exp_firm_payoff_options = np.exp(firm_payoff_options - max_firm_payoff_options) # subtracting maximum has better numerical properties when numbers are very different
                second_policy_fcts = np.nan_to_num(exp_firm_payoff_options / np.sum(np.nan_to_num(exp_firm_payoff_options), axis=1, keepdims=True)) # np.sum(np.nan_to_num(.)) is equivalent to np.nansum, doing it this way b/c autograd doesn't work well with nansum

                # Calculate expected maximum
                v_t_i_j_second = logsumexp(np.nan_to_num(firm_payoff_options), axis=1, b=nonnanvals[f'{firm_idx}'][:,1:]) # using nonnanvals as weights is the same as doing nanlogsumexp (which doesn't exist)

                # Consider the firm that would go first along the dimension (backward or stay same)

                # Take expectation over what second firm would do and set up payoff options
                indices_raveled_select_firm = indices_raveled[f'{firm_idx}'] # select which indices are relevant (don't condition on actions of first firm b/c if second firm moves, we need that info)
                rows_firm_payoff_options = np.reshape(indices_raveled_select_firm, (-1,)) # flatten array because if doing fancy indexing all indices must be 1-d
                cols_firm_payoff_options = np.repeat(firm_selection[:,firm_idx,0], indices_raveled_select_firm.shape[1]) # take the correct firm depending on the state and firm_idx according to firm selection, repeated b/c we can't do broadcasting along the second dimension in the fancy indexing below, need to make explicit since flattening
                first_firm_payoffs = np.reshape(firm_payoff_i[rows_firm_payoff_options,cols_firm_payoff_options], indices_raveled_select_firm.shape) # select the correct entries from firm_payoff_dict, then reshape so that it's num_states x num_options (including actions that can occur due to second firm
                del indices_raveled_select_firm, rows_firm_payoff_options, cols_firm_payoff_options # don't need these anymore, remove to save memory
                second_firm_payoff_backward = first_firm_payoffs[:,0] # if go backward, second firm won't adjust
                second_firm_payoff_same = statem1alongdim_onesandzeros[f'{firm_idx}'] * first_firm_payoffs[:,1] + (1.0 - statem1alongdim_onesandzeros[f'{firm_idx}']) * np.einsum("ij,ij->i", first_firm_payoffs[:,1:], second_policy_fcts) # if we're in the last state in this dimension, there is no second firm that will adjust after; if we are not, then if firm chooses to stay same second firm will adjust, so we take expectation over that then select that choice
                firm_payoff_options = np.concatenate((second_firm_payoff_backward[:,np.newaxis], second_firm_payoff_same[:,np.newaxis]), axis=1)
                del second_firm_payoff_backward, second_firm_payoff_same # don't need them, let's save memory
                firm_payoff_options = firm_payoff_options * vals_should_be_nan[f'{firm_idx}'][:,:2] # makes the actual change to NaNs (done in this way b/c compatible with autograd, direct replacement isn't)
                firm_payoff_options = firm_payoff_options - firm_options_adjustment_costs[f'{firm_idx}'][:,:2] # subtract cost of adjusting

                # Determine probability that firm firm_idx adjusts to each state
                max_firm_payoff_options = nanmax(firm_payoff_options, axis=1, keepdims=True)
                exp_firm_payoff_options = np.exp(firm_payoff_options - max_firm_payoff_options) # subtracting maximum has better numerical properties when numbers are very different
                first_policy_fcts = np.nan_to_num(exp_firm_payoff_options / np.sum(np.nan_to_num(exp_firm_payoff_options), axis=1, keepdims=True)) # np.sum(np.nan_to_num(.)) is equivalent to np.nansum, doing it this way b/c autograd doesn't work well with nansum

                prob_first_not_adjust = first_policy_fcts[:,1]
                policy_fcts = np.zeros(indices_raveled[f'{firm_idx}'].shape)
                policy_fcts[:,0] = first_policy_fcts[:,0]
                policy_fcts[:,1:] = prob_first_not_adjust[:,np.newaxis] * second_policy_fcts

                # Calculate expected maximum
                v_t_i_j_first = logsumexp(np.nan_to_num(firm_payoff_options), axis=1, b=nonnanvals[f'{firm_idx}'][:,:2]) # using nonnanvals as weights is the same as doing nanlogsumexp (which doesn't exist)
                
            # Determine what the expected values are now for the firms that will go before this one
            policy_fcts_sparse = sparse.csr_matrix((policy_fcts.flatten(), (row_policy_fcts[f'{firm_idx}'], col_policy_fcts[f'{firm_idx}'])), shape=(policy_fcts.shape[0], policy_fcts.shape[0]))
            firm_payoff_i = sparse.csr_matrix.dot(policy_fcts_sparse, sparse.csr_matrix(firm_payoff_i)).toarray() # results in state_space_size x num_firms NOTE: we're doing this for firms that already moved too, fine b/c they aren't used again (but also not necessary)
            
            # Update the probabilities of adjustment
            for f in firms_already_adjusted:
                probability_adjustment_i[f"{f}"] = sparse.csr_matrix.dot(policy_fcts_sparse, probability_adjustment_i[f"{f}"]) # keep as sparse matrix so don't have to go back and forth as go through loop
            probability_adjustment_i[f"{firm_idx}"] = sparse.csr_matrix(policy_fcts) # convert to sparse matrix b/c we're going to be using sparse matrices here for future ones
            
            # Update whatever taking expectation over
            if take_expectation is not None:
                take_expectation_i = sparse.csr_matrix.dot(policy_fcts_sparse, sparse.csr_matrix(take_expectation_i)).toarray()
            
            # Take an expectation over value functions for the firms that have already adjusted
            v_t_i = sparse.csr_matrix.dot(policy_fcts_sparse, sparse.csr_matrix(v_t_i)).toarray()
            
            # Add on the value function for the adjusting firm
            if firm_selection is None:
                v_t_i = v_t_i + adjusting_firm_onesandzeros[f'{firm_idx}'] * v_t_i_j[:,np.newaxis]
            else: # if the firm in question depends on the state
                second_firm_value = prob_first_not_adjust * v_t_i_j_second + (1.0 - prob_first_not_adjust) * take_along_axis(firm_payoff_i, firm_selection[:,np.arange(firm_selection.shape[1]) == firm_idx,-1], axis=1)[:,0][indices_raveled[f'{firm_idx}'][:,0]]
                v_t_i_j_interior = adjusting_firm_onesandzeros[f'{firm_idx}'][:,:,0] * v_t_i_j_first[:,np.newaxis] + adjusting_firm_onesandzeros[f'{firm_idx}'][:,:,-1] * second_firm_value[:,np.newaxis] # if we are in the interior in the dimension, there is both a first and a second firm
                v_t_i_j_0 = adjusting_firm_onesandzeros[f'{firm_idx}'][:,:,-1] * v_t_i_j_second[:,np.newaxis] # if we are the first state in the dimension, there is no first firm
                v_t_i_j_m1 = adjusting_firm_onesandzeros[f'{firm_idx}'][:,:,0] * v_t_i_j_first[:,np.newaxis] # if we are at the last state in the dimension, there is no second firm
                v_t_i = v_t_i + state0alongdim_onesandzeros[f'{firm_idx}'][:,np.newaxis] * v_t_i_j_0 + statem1alongdim_onesandzeros[f'{firm_idx}'][:,np.newaxis] * v_t_i_j_m1 + stateinterioralongdim_onesandzeros[f'{firm_idx}'][:,np.newaxis] * v_t_i_j_interior
                if j == 0: # there are potentially firms (in the firm_selection != None case) that do not get a chance to adjust for some states; so, if this is the last firm we consider (so first entry -- j==0 -- in this ordering i), replace the firms that never got a chance to adjust with their expected payoff at the position of the first firm to adjust
                    v_t_i = v_t_i + nonadjusting_firms_onesandzeros * firm_payoff_i
                    
            # Add firm_idx to list of firms that have already adjusted
            firms_already_adjusted = firms_already_adjusted + [firm_idx]
                    
        # Add on the values from this ordering to the expectation over orderings
        for j in range(firm_orders.shape[1]):
            probability_adjustment[f"{j}"] = probability_adjustment[f"{j}"] + pr_firm_orders[i] * probability_adjustment_i[f"{j}"].toarray() # add on the probability of an adjustment conditional on the order in which firms move weighted by the probability of firms making that move
        v_t = v_t + pr_firm_orders[i] * v_t_i
        if take_expectation is not None:
            take_expectation_result = take_expectation_result + pr_firm_orders[i] * take_expectation_i
        
    # Return values
    res = [v_t, probability_adjustment]
    if take_expectation is not None:
        res += [take_expectation_result]
    return tuple(res)

# Determine choice probabilities in each year
def choice_probabilities(v_t_strategic, v_t_competitive, profits_strategic, profits_competitive, dims, dims_correspondence, adjustment_costs, competitive_firm_selection, row_indices, col_indices, beta, max_T, compute_specific_prob=None, save_probs=False, save_llh_t_i=False, return_sparse_matrix_indexing=False, print_msg=False):
    """
        Return the value and policy functions for all firms
    
    Parameters
    ----------
        v_tplus1 : ndarray
            (N,F+S) array of value functions in next period for each firm
        profit_t : ndarray
            (N,F+S) array of wholesale profits + capacity payments for each firm
        adjustment_cost_t : ndarray
            (N,D) array of what the adjustment cost would be if we made an adjustment from that state to the next in that dimension
        dims : tuple
            dimensions of state that has been flattened into N
        dims_correspondence : ndarray
            (F,S*F) array of bools where each row corresponds to firm and columns to dimension, entry tells us which firm in question controls that dimension
        adjustment_cost_t : ndarray
            (N,2,F) array of cost of adjusting for each firm (2 b/c firms are going to be allowed to have only two options)
        beta : float
            discount factor

    Returns
    -------
        probability_adjustment : ndarray
            (N,N_possible) array of choice probabilities
    """
    
    if print_msg:
        print(f"Beginning computation of equilibrium...", flush=True)
    
    # Determine unraveled indices, will use later b/c can only move one step of the state space at a time, and this allows us to determine that in computationally efficient way
    dims_arr = np.array(dims)
    state_space_size = np.prod(dims_arr)
    indices_unraveled = np.concatenate([x[np.newaxis,:] for x in np.unravel_index(np.arange(profits_competitive.shape[0]), dims)], axis=0) # unraveled index
    
    # Firm orders and probabilities
    firm_orders = {}
    pr_firm_orders = {}
    for set_type in ["competitive", "strategic"]:
        num_firms_set_type = dims_correspondence[set_type].shape[0]
        firm_orders[set_type] = np.array(list(permutations(np.arange(num_firms_set_type))), dtype=int)
        pr_firm_orders[set_type] = np.ones(firm_orders[set_type].shape[0]) / float(firm_orders[set_type].shape[0]) # equal probabilities
        
    # Initialize adjustment probability arrays
    llh_obs = 0.0
    if save_probs:
        probability_adjustment_dict = {}
        for set_type in ["competitive", "strategic"]:
            for j in range(dims_correspondence[set_type].shape[0]):
                dims_firm_changes = dims_correspondence[set_type][j,:] # which dimensions can this firm adjust
                num_dims_firm_changes = np.sum(dims_firm_changes)
                num_firm_options = 3**num_dims_firm_changes # number of possible options given number of dimensions the firm can change
                probability_adjustment_dict[f'{set_type},{j}'] = np.zeros((indices_unraveled.shape[1], num_firm_options, max_T))
    
    if print_msg:
        print(f"Preprocessing finished.", flush=True)
        
    if save_llh_t_i:
        llh_t_i = []
    
    # Go through each year, going backward
    for t in range(max_T - 1, -1, -1):
        
        start = time.time()
        
        # Determine payoffs in period t
        firm_payoff_competitive = profits_competitive[:,t,:] + beta * v_t_competitive # add this period's returns to value function
        firm_payoff_strategic = profits_strategic[:,t,:] + beta * v_t_strategic # add this period's returns to value function
        
        # Determine value and policy functions for the competitive sources
        v_t_competitive, probability_adjustment_competitive_t, firm_payoff_strategic = value_fct(firm_payoff_competitive, adjustment_costs[t,:,:], indices_unraveled, dims, dims_correspondence['competitive'], row_indices['competitive'], col_indices['competitive'], firm_orders['competitive'], pr_firm_orders['competitive'], firm_selection=competitive_firm_selection, take_expectation=firm_payoff_strategic)
        
        # Determine value and policy functions for the strategic firms
        v_t_strategic, probability_adjustment_strategic_t, v_t_competitive = value_fct(firm_payoff_strategic, adjustment_costs[t,:,:], indices_unraveled, dims, dims_correspondence['strategic'], row_indices['strategic'], col_indices['strategic'], firm_orders['strategic'], pr_firm_orders['strategic'], take_expectation=v_t_competitive)
        
        # Put the adjustment probabilities in a dictionary for easy access
        probability_adjustment_dict_t = { # save these in a dict, makes things easier, just a view so no extra computation time
            'competitive': probability_adjustment_competitive_t, 
            'strategic': probability_adjustment_strategic_t
        }

        # Determine probability of a particular observation
        if compute_specific_prob is not None:
            if t < compute_specific_prob['competitive_start_idx'].shape[0]: # max_T may be larger than the number of years we have data for
                for set_type in ["competitive", "strategic"]:
                    for i in range(dims_correspondence[set_type].shape[0]): # go through each competitive fringe source / strategic firm
                        probability_t_i = probability_adjustment_dict_t[set_type][f'{i}'] # select the probability of adjustment in this period t for firm i within the set type
                        probability_adjustment_t_i = probability_t_i[compute_specific_prob[f'{set_type}_start_idx'][t],compute_specific_prob[f'{set_type}_choice_idx'][t,i]] # select the state we're starting from and the correct dimensions to look at
                        llh_obs = llh_obs + np.log(probability_adjustment_t_i)
                        if save_llh_t_i:
                            llh_t_i = llh_t_i + [np.log(probability_adjustment_t_i)]
        
        # Save adjustment probabilities
        if save_probs:
            for set_type in ["competitive", "strategic"]:
                for i in range(dims_correspondence[set_type].shape[0]):
                    probability_adjustment_dict[f'{set_type},{i}'][:,:,t] = probability_adjustment_dict_t[set_type][f'{i}']
            
        del probability_adjustment_dict_t # don't need anymore
        gc.collect()

        if print_msg:
            print(f"\tfinished iteration t={t} in {np.round(time.time() - start, 2)} seconds.", flush=True)
            
    if print_msg:
        print(f"Completed equilibrium computation.", flush=True)
        if compute_specific_prob is not None:
            print(f"\tloglikelihood = {np.round(llh_obs, 3)}", flush=True)

    res = []

    if compute_specific_prob is not None:
        res += [llh_obs]
        
    if save_llh_t_i:
        res += [llh_t_i]

    if save_probs:
        res += [probability_adjustment_dict, v_t_strategic, v_t_competitive]

    if return_sparse_matrix_indexing:
        res += [row_indices, col_indices]

    return tuple(res)
