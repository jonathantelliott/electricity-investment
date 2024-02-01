# %%
# Import packages
import sys
from itertools import product

from multiprocessing import Pool

import time as time

import numpy as np
import scipy.stats as stats
import pandas as pd

import global_vars as gv
import wholesale.wholesale_profits as eqm
import wholesale.capacity_commitment as cc
import wholesale.demand as demand

# %%
# Parameters for generating wholesale profits
num_draws = 1000
running_specification = int(sys.argv[1])
num_cpus = int(sys.argv[2])
print(f"num_cpus: {num_cpus}", flush=True)
generator_groupings = gv.generator_groupings
generators_use = np.concatenate(tuple([value for key, value in generator_groupings.items()]))
groupings_use = np.concatenate(tuple([np.array([key]) for key, value in generator_groupings.items()]))

# %%
# Import data and wholesale market production cost estimates

# Previously-processed data
dates = np.load(gv.dates_file)
facilities = np.load(gv.facilities_file)
participants = np.load(gv.participants_file)
capacities = np.load(gv.capacities_file)
energy_sources = np.load(gv.energy_sources_file)
heat_rates = np.load(gv.heat_rates)
co2_rates = np.load(gv.co2_rates)
transport_charges = np.load(gv.transport_charges)
exited = np.load(gv.exited_file)
loaded = np.load(gv.energy_gen_file)
energy_gen = np.copy(loaded['arr_0'])
loaded.close()
prices = np.load(gv.prices_realtime_file) # np.load(gv.prices_file)
load_curtailed = np.load(gv.load_curtailment_file)
tariffs = np.load(gv.residential_tariff_file)
carbon_taxes = np.load(gv.carbon_taxes_file)
exited_date = np.load(gv.exited_date_file)
entered_date = np.load(gv.entered_date_file)
max_prices = np.load(gv.max_prices_file)
loaded = np.load(gv.outages_file)
outages_exante = np.copy(loaded['arr_0'])
outages_expost = np.copy(loaded['arr_1'])
outages_type = np.copy(loaded['arr_2'])
loaded.close()
balancing_avail = np.load(gv.balancing_avail_file)
gas_prices = np.load(gv.gas_prices_file)
coal_prices = np.load(gv.coal_prices_file)
capacity_price = np.load(gv.capacity_price_file)
cap_date_from = np.load(gv.cap_date_from_file)
cap_date_until = np.load(gv.cap_date_until_file)
print(f"Finished importing data.", flush=True)

# Demand elasticity estimate
demand_elasticity = np.load(gv.arrays_path + "demand_elasticity_estimates.npy")[-1] # we want the last specification
print(f"Finished import demand elasticity estimates.", flush=True)

# Production cost shock estimates
production_cost_est = np.load(gv.arrays_path + "production_cost_estimates.npy")
production_cost_est_sources = np.load(gv.arrays_path + "production_cost_estimates_sources.npy")
print(f"Finished importing production cost estimates.", flush=True)

# %%
# Calibrate fixed price component so residential tariffs = predicted prices

# Select the right dates
select_july1 = (pd.to_datetime(dates).month == 7) & (pd.to_datetime(dates).day == 1)
select_july1[np.isnan(tariffs)] = False
years_use_tariffs = pd.to_datetime(dates).year[select_july1]

# Determine the quantity-weighted average price
quantity_weighted_avg_price = np.zeros(years_use_tariffs.shape)
for i, yr in enumerate(years_use_tariffs):
    yr_arange = np.arange(np.datetime64(str(yr).zfill(4) + "-07-01"), np.datetime64(str(yr + 1).zfill(4) + "-06-30"))
    select_yr = np.isin(dates, yr_arange)
    quantity_weighted_avg_price[i] = np.average(prices[select_yr,:], weights=np.nansum(energy_gen[:,select_yr,:], axis=0))

# Calibrate the fixed component as the average difference    
fixed_tariff_component = np.mean(tariffs[select_july1] - quantity_weighted_avg_price)

# Fill in the tariffs that are NaN b/c before or after sample (useful for sampling later)
first_nonnan_idx = np.min(np.arange(tariffs.shape[0])[~np.isnan(tariffs)])
tariffs[np.arange(tariffs.shape[0]) < first_nonnan_idx] = tariffs[first_nonnan_idx]
last_nonnan_idx = np.max(np.arange(tariffs.shape[0])[~np.isnan(tariffs)])
tariffs[np.arange(tariffs.shape[0]) > last_nonnan_idx] = tariffs[last_nonnan_idx]

print(f"Calibrated the fixed component of tariff", flush=True)

# %%
# Process data

# Determine largest firms
participants_unique = np.unique(participants)
participants_tot_produced = np.zeros(participants_unique.shape)
for participant_idx, participant in enumerate(participants_unique):
    participants_tot_produced[participant_idx] = np.nansum(energy_gen[participants == participant,:,:])
participants_sorted = participants_unique[np.argsort(participants_tot_produced)[::-1]]
print(f"Determined largest firms.", flush=True)

# Relabel all but the largest firms as competitive
participants_originalnames = np.copy(participants)
participants[~np.isin(participants, participants_sorted[:gv.num_strategic_firms])] = gv.competitive_name
print(f"Renamed participants based on whether strategic or not.", flush=True)

# Reshape arrays
num_intervals_in_day = energy_gen.shape[2]
dates = np.repeat(dates, num_intervals_in_day)
energy_gen = np.reshape(energy_gen, (facilities.shape[0], -1))
prices = np.reshape(prices, (-1,))
load_curtailed = np.reshape(load_curtailed, (-1,))
tariffs = np.repeat(tariffs, num_intervals_in_day)
carbon_taxes = np.repeat(carbon_taxes, num_intervals_in_day)
max_prices = np.repeat(max_prices, num_intervals_in_day)
outages_exante = np.reshape(outages_exante, (facilities.shape[0], -1))
outages_expost = np.reshape(outages_expost, (facilities.shape[0], -1))
outages_type = np.reshape(outages_type, (facilities.shape[0], -1))
balancing_avail = np.reshape(balancing_avail, (facilities.shape[0], -1))
gas_prices = np.repeat(gas_prices, num_intervals_in_day)
coal_prices = np.repeat(coal_prices, num_intervals_in_day)
print(f"Reshaped data arrays.", flush=True)

# Total energy
total_load_all = np.nansum(energy_gen, axis=0)
total_load_all = total_load_all - np.nansum(energy_gen[~np.isin(facilities, generators_use),:], axis=0) + load_curtailed # subtract off the production by sources I'm not considering (e.g., landfill gas), this stuff is tiny

# What fraction is not included?
print(f"Fraction not included: {np.round((np.nansum(energy_gen[~np.isin(facilities, generators_use),:]) / np.nansum(energy_gen)) * 100.0, 2)}%.", flush=True)

# Remove the facilities not in gv.use_sources
facilities_orig = np.copy(facilities)
facilities = facilities[np.isin(facilities_orig, generators_use)]
participants = participants[np.isin(facilities_orig, generators_use)]
participants_originalnames = participants_originalnames[np.isin(facilities_orig, generators_use)]
capacities = capacities[np.isin(facilities_orig, generators_use)]
heat_rates = heat_rates[np.isin(facilities_orig, generators_use)]
co2_rates = co2_rates[np.isin(facilities_orig, generators_use)]
transport_charges = transport_charges[np.isin(facilities_orig, generators_use)]
energy_sources = energy_sources[np.isin(facilities_orig, generators_use)]
exited = exited[np.isin(facilities_orig, generators_use)]
energy_gen = energy_gen[np.isin(facilities_orig, generators_use),:]
exited_date = exited_date[np.isin(facilities_orig, generators_use)]
entered_date = entered_date[np.isin(facilities_orig, generators_use)]
outages_exante = outages_exante[np.isin(facilities_orig, generators_use),:]
outages_expost = outages_expost[np.isin(facilities_orig, generators_use),:]
outages_type = outages_type[np.isin(facilities_orig, generators_use),:]
balancing_avail = balancing_avail[np.isin(facilities_orig, generators_use),:]

# Mark gas_cogen as gas_ocgt (b/c all are OCGT)
energy_sources[energy_sources == gv.gas_cogen] = gv.gas_ocgt

# Determine when out of market
out_of_market = np.isnan(energy_gen) | (dates[np.newaxis,:] < entered_date[:,np.newaxis]) | (dates[np.newaxis,:] > exited_date[:,np.newaxis])

# Create capacity factors for each source that we can sample from later
cap_factor = {}
capacities_use_cap_factors = np.tile(capacities[:,np.newaxis], (1,energy_gen.shape[1]))
first_date_expansion_of_greenough_river_pv1 = np.min(dates[(energy_gen[facilities == "GREENOUGH_RIVER_PV1",:][0,:] * 2.0) > (0.3 * capacities[facilities == "GREENOUGH_RIVER_PV1"][0])]) # GREENOUGH_RIVER_PV1 had an expansion, so need to identify the date to do capacity factor properly
select_replacement_region = np.ix_(facilities == "GREENOUGH_RIVER_PV1", dates < first_date_expansion_of_greenough_river_pv1)
capacities_use_cap_factors[select_replacement_region] = np.nanmax(energy_gen[select_replacement_region]) * 2.0 # previous capacity
for source in gv.use_sources:
    select_source = energy_sources == source
    if np.isin(source, gv.intermittent): # intermittent sources have effective capacity based on what produced
        cap_factor[source] = energy_gen[select_source,:] / (capacities_use_cap_factors[select_source,:] / 2.0)
    elif np.isin(source, production_cost_est_sources): # thermal sources have effective capacity based on outage data
        cap_factor[source] = np.maximum(0.0, np.minimum((capacities_use_cap_factors[select_source,:] - outages_expost[select_source,:]) / capacities_use_cap_factors[select_source,:], 1.0))
    if np.sum(select_source) > 0:
        cap_factor[source][out_of_market[select_source,:]] = np.nan # if wasn't in market, make NaN b/c don't want to later sample from it
    
print(f"Finished processing data.", flush=True)

# %%
# Construct dictionary of the possible states

new_ccgt = gv.gas_ccgt
new_wind = gv.wind
new_solar = gv.solar

num_strategic_firms = 3
strategic_firms = participants_sorted[:num_strategic_firms]
all_firms = np.concatenate((strategic_firms, np.array([gv.competitive_name])))

# First list is minimum allowed, second list is maximum, we will fill in the intermediate values in the next step
states_by_dim = {
    f'WPGENER,coal': [[], ["COLLIE_G1", "MUJA_G7-8", "MUJA_G5-6", "KWINANA_G5-6", "KWINANA_G1-4"]], 
    f'WPGENER,gas_add': [["COCKBURN_CCG1"], ["KWINANA_GT2-3", new_ccgt]], 
    f'WPGENER,gas_subtract': [[], ["KEMERTON_GT11-12", "PINJAR_GT1-11", "PPP_KCP_EG1", "SWCJV_WORSLEY_COGEN_COG1"]], 
    f'WPGENER,solar': [], 
    f'WPGENER,wind': [], 
    f'ALINTA,coal': [], 
    f'ALINTA,gas_add': [["ALINTA_PNJ_U1-2"], ["ALINTA_WGP", new_ccgt]], 
    f'ALINTA,gas_subtract': [], 
    f'ALINTA,solar': [], 
    f'ALINTA,wind': [["ALINTA_WWF"], ["BADGINGARRA_WF1", "YANDIN_WF1", new_wind, new_wind]], 
    f'GRIFFINP,coal': [[], ["BW2_BLUEWATERS_G1-2"]], 
    f'GRIFFINP,gas_add': [], 
    f'GRIFFINP,gas_subtract': [], 
    f'GRIFFINP,solar': [], 
    f'GRIFFINP,wind': [], 
    f'{gv.competitive_name},coal': [[], ["MUJA_G1-4"]], 
    f'{gv.competitive_name},gas_add': [[], ["NEWGEN_KWINANA_CCG1", "NEWGEN_NEERABUP_GT1", "PERTHENERGY_KWINANA_GT1", new_ccgt, new_ccgt]], 
    f'{gv.competitive_name},gas_subtract': [], 
    f'{gv.competitive_name},solar': [[], ["GREENOUGH_RIVER_PV1", "MERSOLAR_PV1", new_solar, new_solar]], 
    f'{gv.competitive_name},wind': [["ALBANY_WF1", "EDWFMAN_WF1"], ["INVESTEC_COLLGAR_WF1", "MWF_MUMBIDA_WF1", "WARRADARGE_WF1", new_wind, new_wind]]
}

# Expand along each dimension to add the intermiediate values
for key, value in states_by_dim.items():
    if len(value) > 1:
        states_by_dim[key] = [value[0] + value[1][:i] for i in range(len(value[1]) + 1)] # create the intermediary values one-by-one
print(f"Size of state space: {np.prod([np.maximum(len(value), 1) for key, value in states_by_dim.items()]):,}", flush=True)

# Replace the names with the actual identities of the facilities
for key, value in states_by_dim.items():
    list_states = [] # initialize list
    for state in value:
        if len(state) > 0:
            list_states = list_states + [np.concatenate(tuple([generator_groupings[grouping] if np.isin(grouping, groupings_use) else [grouping] for grouping in state]))]
        else:
            list_states = list_states + [np.array([], dtype="<U1")] # add empty 
    states_by_dim[key] = list_states
    
print(f"Finish constructing the states by firm-source.", flush=True)

# Reverse the order if it's a dimension in which generators are being retired
for key, value in states_by_dim.items():
    if np.isin(key.split(",")[1], np.array(["coal", "gas_subtract"])) and not np.isin(key.split(",")[0], np.array(["GRIFFINP"])):
        states_by_dim[key] = states_by_dim[key][::-1] # reverse order
print(f"Reversed order of dimensions along which we subtract.", flush=True)

# %%
# Add the new (never seen in the data) generators

capacities_add = {
    gv.gas_ccgt: 400.0, 
    gv.solar: 400.0, 
    gv.wind: 400.0
}
energy_sources_add = {
    gv.gas_ccgt: gv.gas_ccgt, 
    gv.solar: gv.solar, 
    gv.wind: gv.wind
}
heat_rates_add = {
    gv.gas_ccgt: 8.0, 
    gv.solar: np.nan, 
    gv.wind: np.nan
}
co2_rates_add = {
    gv.gas_ccgt: 450.0, 
    gv.solar: 0.0, 
    gv.wind: 0.0
}

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
    
if running_specification == 0:
    create_file(f"{gv.stats_path}capacity_new_ccgt.tex", f"{capacities_add[gv.gas_ccgt]:.0f}")
    create_file(f"{gv.stats_path}capacity_new_solar.tex", f"{capacities_add[gv.solar]:.0f}")
    create_file(f"{gv.stats_path}capacity_new_wind.tex", f"{capacities_add[gv.wind]:.0f}")
    create_file(f"{gv.stats_path}heat_rate_new_ccgt.tex", f"{heat_rates_add[gv.gas_ccgt]:.1f}")
    create_file(f"{gv.stats_path}co2_rate_new_ccgt.tex", f"{co2_rates_add[gv.gas_ccgt]:.0f}")

new_sources = np.array([new_ccgt, new_wind, new_solar])
for key, value in states_by_dim.items():
    if len(value) > 1:
        num_new = np.sum(np.isin(value[-1], new_sources))
        firm = key.split(",")[0]
        if num_new > 0:
            source = value[-1][-1]
            for i, state in enumerate(value): # go through each state along that dimension
                isin_new_sources = np.isin(state, new_sources)
                gens_replace_idx = np.arange(state.shape[0])[isin_new_sources]
                if gens_replace_idx.shape[0] > 0:
                    state = np.concatenate((state[~isin_new_sources], np.array([f"new_{state[gens_replace_idx[0]]}_{firm}_{i+1}" for i in range(gens_replace_idx.shape[0])]))) # this works because the new values are always at the end of the array in state
                    value[i] = state
                    
            # add on to the arrays of facilities and their characteristics
            facilities = np.concatenate((facilities, np.array([f"new_{source}_{firm}_{i+1}" for i in range(num_new)])))
            capacities = np.concatenate((capacities, np.array([capacities_add[source] for i in range(num_new)])))
            energy_sources = np.concatenate((energy_sources, np.array([f"{source}" for i in range(num_new)])))
            heat_rates = np.concatenate((heat_rates, np.array([heat_rates_add[source] for i in range(num_new)])))
            co2_rates = np.concatenate((co2_rates, np.array([co2_rates_add[source] for i in range(num_new)])))
            participants = np.concatenate((participants, np.array([firm for i in range(num_new)])))
            participants_originalnames = np.concatenate((participants_originalnames, np.array([firm for i in range(num_new)])))
            
energy_sources_list = np.unique(energy_sources)

# Rename competitive participants to treat each grouping of generators separately
participants = list(participants) # do list instead of array b/c o/w will cut off end of strings, convert later
for i in range(facilities.shape[0]):
    if participants[i] == "c":
        generator_grouping_name = "c_"
        found_in_grouping = False
        for key, value in generator_groupings.items():
            if np.isin(facilities[i], generator_groupings[key]):
                generator_grouping_name = generator_grouping_name + key
                found_in_grouping = True
        if not found_in_grouping: # this means it's one of the new generators
            generator_grouping_name = generator_grouping_name + facilities[i] # this works b/c all the new ones are their own grouping
        participants[i] = generator_grouping_name
participants = np.array(participants) # convert back to array

# %%
# Construct array of state

# Construct list of dimensions
list_firms_state = np.unique([key.split(",")[0] for key in states_by_dim.keys()])
num_firms = list_firms_state.shape[0]
list_sources_state = np.unique([key.split(",")[1] for key in states_by_dim.keys()])
num_sources = list_sources_state.shape[0]
state_shape_list = [np.maximum(1, len(states_by_dim[f"{firm},{source}"])) for firm in list_firms_state for source in list_sources_state] # possible states

# Create dictionary in which for each entry, says whether generator is in or out (i.e., True or False)
#     if not relevant for this dimension (e.g., ALINTA,coal in the WPGENER,gas dimension), list those generators as in
states_by_dim_inout = {}
for key, value in states_by_dim.items():
    facilities_inout_list = []
    if len(value) > 0:
        relevant_facilities = states_by_dim[key][np.argmax([arr.shape[0] for arr in states_by_dim[key]])] # this is an array of the maximal list of generators in this dimension
        for state in states_by_dim[key]:
            facilities_inout_list = facilities_inout_list + [np.isin(facilities, state) | ~np.isin(facilities, relevant_facilities)] # add an entry in which generator is False if a relevant facility and not in this state
    else:
        facilities_inout_list = facilities_inout_list + [np.ones(facilities.shape[0], dtype=bool)] # all in (vacuously because there are no facilities along this dimension in this case)
    facilities_inout_array = np.concatenate(tuple([facilities_inout_array_state[:,np.newaxis] for facilities_inout_array_state in facilities_inout_list]), axis=1)
    states_by_dim_inout[key] = facilities_inout_array

# Take the dictionary and make into one big array with dimension facilities.shape[0] x (all of the dimensions, x_firms x_sources, where x is cross production)
array_state_in = np.ones(tuple([facilities.shape[0]] + state_shape_list), dtype=bool) # initialize size and make all True (will take care of "False"s in loop below)
for f, firm in enumerate(list_firms_state):
    for s, source in enumerate(list_sources_state):
        dim_1 = states_by_dim_inout[f"{firm},{source}"].shape[0] # this should always be the same as facilities.shape[0]
        dim_2_expand = f * num_sources + s # this is the number of dimensions to expand of firm-sources that go before this dimension
        dim_3 = states_by_dim_inout[f"{firm},{source}"].shape[1] # this is the number of states along this dimension
        dim_4_expand = num_firms * num_sources - dim_2_expand - 1 # this is the number of dimensions to expand of firm-sources that go after this dimension
        array_reshape_shape = tuple([dim_1] + [1 for i in range(dim_2_expand)] + [dim_3] + [1 for i in range(dim_4_expand)]) # tuple with new shape that incorporates correct expanded dimensions
        array_state_in = array_state_in * np.reshape(states_by_dim_inout[f"{firm},{source}"], array_reshape_shape) # multiply the broadcasted array
        
# %%
# Create table that describes the choice set / state space

# Begin table
tex_table = f""
tex_table += f"\\begin{{tabular}}{{ lllclc }} \n"
tex_table += f" & & \multicolumn{{1}}{{c}}{{Expand}} / & & & \multicolumn{{1}}{{c}}{{Total}} \\\\ \n"
tex_table += f"\multicolumn{{1}}{{c}}{{Firm}} & \multicolumn{{1}}{{c}}{{Technology}} & \multicolumn{{1}}{{c}}{{Retire}} & \multicolumn{{1}}{{c}}{{State}} & \multicolumn{{1}}{{c}}{{Generators}} & \multicolumn{{1}}{{c}}{{Capacity (MW)}} \\\\ \n \hline \n"

# Fill in table
firm_bf = "" # string to compare firm name to, prevents listing firm name many times
for key, value in states_by_dim.items():
    
    # If not a technology we consider, just ignore
    if len(value) == 0:
        continue
        
    # Determine firm name
    firm = key.split(",")[0]
    if firm == firm_bf: # new firm
        firm = ""
    else:
        firm_bf = firm
        if firm == "c":
            firm = "small firms"
    tex_table += f"{firm} & "
        
    # Determine 
    technology = key.split(",")[1]
    expand_retire = "expand"
    expand_bool = True
    if np.isin(technology, np.array(["coal", "gas_subtract"])):
        if firm != "GRIFFINP": # this one actually expands coal
            expand_retire = "retire"
            expand_bool = False
            value = value[::-1]
    if np.isin(technology, np.array(["gas_add", "gas_subtract"])):
        technology = "gas"
    tex_table += f"{technology} & {expand_retire} & "
        
    prev_gens = np.array([], dtype="<U1")
    for i, list_generators in enumerate(value):
        
        # Add blanks for firm / technology if not first entry
        if i > 0:
            tex_table += f" & & & "
            
        # Add what state we're in
        tex_table += f"{i if expand_bool else len(value) - 1 - i} & "
        
        # Add generators
        if list_generators.shape[0] == 0:
            tex_table += f"$\\emptyset$ & "
        else:
            gens_display = list_generators[~np.isin(list_generators, value[i - 1])] if i > 0 else list_generators
            prev_gens = np.copy(list_generators)
            for j, gen in enumerate(gens_display):
                if "new" in gen: # if it's one of the new ones
                    gens_display[j] = "new " + gen.split("_")[1]
            if gens_display.shape[0] < 4:
                tex_table += (", ").join(gens_display).replace("_", "\\_") + f" & "
            else:
                tex_table += f"{gens_display[0]}--{gens_display[-1]} & ".replace("_", "\\_")
            
        # Add capacity
        capacities_in_out_i = states_by_dim_inout[key][:,i if expand_bool else len(value) - 1 - i] * capacities
        capacity_i = np.sum(capacities_in_out_i[np.isin(facilities, value[-1])])
        tex_table += f"{capacity_i:,.0f} ".replace(",", "\\,")
        
        # End row
        tex_table += f"\\\\ \n"

# End table
tex_table += f"\\hline \n \\end{{tabular}} \n"
    
print(tex_table, flush=True)
    
if running_specification == 0:
    create_file(gv.tables_path + "choice_set.tex", tex_table)

# %%
# Save num draws stat
    
if running_specification == 0:
    create_file(gv.stats_path + "wholesale_profits_num_draws.tex", f"{num_draws:,}".replace(",", "\\,"))

# %%
# Sample from distributions

# Determine years
years = pd.to_datetime(dates).year.values
months = pd.to_datetime(dates).month.values
years[months < 10] = years[months < 10] - 1 # use the same year system as WEM, begins in October
years_unique = np.unique(years)
num_years = years_unique.shape[0]

# Create parameters of skew normal based on estimate array
def corr_matrix(params):
    k = int((1 + np.sqrt(8 * params.shape[0] + 1)) / 2)
    z = np.tanh(params)
    chol_factor = np.zeros((k, k))
    chol_factor[0,0] = 1.0
    z_ctr = 0
    for i in range(1, k):
        for j in range(k):
            if j < i:
                chol_factor[i,j] = z[z_ctr] * np.sqrt(1.0 - np.sum(chol_factor[i,:j]**2.0))
                z_ctr += 1
            elif j == i:
                chol_factor[i,j] = np.sqrt(1.0 - np.sum(chol_factor[i,:j]**2.0))
    matrix_R = chol_factor @ chol_factor.T
    return matrix_R
mu = production_cost_est[0:production_cost_est_sources.shape[0]]
sigma = production_cost_est[production_cost_est_sources.shape[0]:(2*production_cost_est_sources.shape[0])]
alpha = production_cost_est[(2*production_cost_est_sources.shape[0]):(3*production_cost_est_sources.shape[0])]
corr_params = production_cost_est[(3*production_cost_est_sources.shape[0]):]
matrix_R_expanded = corr_matrix(corr_params)
matrix_R = matrix_R_expanded[1:,1:]
for i in range(matrix_R.shape[0]):
    matrix_R[i,i] = matrix_R_expanded[0,i+1]
mu = np.concatenate((mu, np.zeros(gv.intermittent.shape[0])))
sigma = np.concatenate((sigma, np.zeros(gv.intermittent.shape[0])))
alpha = np.concatenate((alpha, np.zeros(gv.intermittent.shape[0])))
matrix_R = np.concatenate((matrix_R, np.zeros((matrix_R.shape[0], gv.intermittent.shape[0]))), axis=1)
matrix_R = np.concatenate((matrix_R, np.zeros((gv.intermittent.shape[0], matrix_R.shape[1]))), axis=0)
mu_gens = np.zeros(facilities.shape[0])
sigma_gens = np.zeros(facilities.shape[0])
alpha_gens = np.zeros(facilities.shape[0])
matrix_R_full = np.zeros((facilities.shape[0], facilities.shape[0]))
for s, source in enumerate(np.concatenate((production_cost_est_sources, gv.intermittent))):
    mu_gens += mu[s] * (energy_sources == source)
    sigma_gens += sigma[s] * (energy_sources == source)
    alpha_gens += alpha[s] * (energy_sources == source)
    for s_p, source_p in enumerate(np.concatenate((production_cost_est_sources, gv.intermittent))):
        matrix_R_full[(energy_sources == source)[:,np.newaxis] & (energy_sources == source_p)[np.newaxis,:]] = matrix_R[s,s_p]
for i in range(matrix_R_full.shape[0]):
    matrix_R_full[i,i] = 1.0

# Draw from skew normal based on estimates of distribution
alpha_cov_alpha = alpha_gens @ matrix_R_full @ alpha_gens
delta = (1.0 / np.sqrt(1.0 + alpha_cov_alpha)) * matrix_R_full @ alpha_gens
covariance_addition = np.block([[np.ones(1), delta], [delta[:,np.newaxis], matrix_R_full]])
covariance_addition_cholesky = np.linalg.cholesky(covariance_addition)
x = stats.multivariate_normal.rvs(mean=np.zeros(energy_sources.shape[0] + 1), cov=np.identity(energy_sources.shape[0] + 1), size=num_draws) @ covariance_addition_cholesky.T
x0, x1 = x[:,0], x[:,1:]
x1 = (x0 > 0.0)[:,np.newaxis] * x1 - (x0 <= 0.0)[:,np.newaxis] * x1
production_cost_shocks = x1.T * sigma_gens[:,np.newaxis] + mu_gens[:,np.newaxis]

# Initialize arrays
cap_factors = np.ones((capacities.shape[0], num_years, num_draws)) * np.nan
production_costs = np.ones((capacities.shape[0], num_years, num_draws)) * np.nan
demand_realizations = np.ones((num_years, num_draws)) * np.nan
tariffs_sample = np.ones((num_years, num_draws)) * np.nan
price_cap = np.ones((num_years, num_draws)) * np.nan
carbon_taxes_rates = np.ones((num_years, num_draws)) * np.nan
idx_all_obs = np.arange(years.shape[0])
np.random.seed(1234567)
for y, year in enumerate(years_unique):
    sample_indices = np.random.choice(idx_all_obs[years == year], size=num_draws)
    
    # Demand
    demand_realizations[y,:] = total_load_all[sample_indices]
    
    # Tariffs
    tariffs_sample[y,:] = tariffs[sample_indices]
    
    # Price cap
    price_cap[y,:] = max_prices[sample_indices]
    
    # Carbon tax rates
    carbon_taxes_rates[y,:] = carbon_taxes[sample_indices]
    
    # Energy source-specific stuff:
    for s, source in enumerate(np.unique(energy_sources)):
        print(f"Computing {source} in year {year}.", flush=True)
        select_source = energy_sources == source
        num_source = np.sum(select_source)
        # Capacity factors
        cap_factor_source = cap_factor[source]
        if np.all(np.isnan(cap_factor_source[:,sample_indices])): # if this source wasn't in the sample in this year
            cap_factors[select_source,y,:] = np.nan # set to NaN, we will use the nearest year later
        else:
            for i in range(num_draws): # sample capacity factors from each date
                min_idx_to_add = np.maximum(0, sample_indices[i] - 1)
                max_idx_to_add = np.minimum(sample_indices[i], idx_all_obs[-1])
                sample_idx_plus_bf_and_af = np.array([min_idx_to_add, sample_indices[i], max_idx_to_add]) # use the interval before and after as well to get sufficient number of observations
                cap_factor_sample_plus_bf_and_af = cap_factor_source[:,sample_idx_plus_bf_and_af]
                cap_factor_sample_plus_bf_and_af_notnan = ~np.isnan(cap_factor_sample_plus_bf_and_af)
                num_obs_in_this_idx = np.sum(cap_factor_sample_plus_bf_and_af_notnan)
                ctr = 0
                while (num_obs_in_this_idx < num_source) and (ctr < 590): # if there are not enough in sample such that each could (in theory, but we sample w/ replacement) be sampled from a different value, expand until there are
                    if (min_idx_to_add > 0) and (max_idx_to_add < idx_all_obs[-1]):
                        min_idx_to_add = min_idx_to_add - 1
                        max_idx_to_add = max_idx_to_add + 1
                        sample_idx_plus_bf_and_af = np.concatenate((np.array([min_idx_to_add]), sample_idx_plus_bf_and_af, np.array([max_idx_to_add])))
                    elif min_idx_to_add == 0: # can't have hit max in this case b/c there is a positive number of observations
                        max_idx_to_add = max_idx_to_add + 1
                        sample_idx_plus_bf_and_af = np.concatenate((sample_idx_plus_bf_and_af, np.array([max_idx_to_add])))
                    elif max_idx_to_add == idx_all_obs[-1]:
                        min_idx_to_add = min_idx_to_add - 1
                        sample_idx_plus_bf_and_af = np.concatenate((np.array([min_idx_to_add]), sample_idx_plus_bf_and_af))
                    cap_factor_sample_plus_bf_and_af = cap_factor_source[:,sample_idx_plus_bf_and_af]
                    cap_factor_sample_plus_bf_and_af_notnan = ~np.isnan(cap_factor_sample_plus_bf_and_af)
                    num_obs_in_this_idx = np.sum(cap_factor_sample_plus_bf_and_af_notnan)
                    ctr = ctr + 1
                if num_obs_in_this_idx >= num_source:
                    cap_factors[select_source,y,i] = np.random.choice(cap_factor_sample_plus_bf_and_af[cap_factor_sample_plus_bf_and_af_notnan], size=num_source)
                else:
                    cap_factors[select_source,y,i] = np.nan # will get replaced later
            
        # Production costs
        if np.isin(source, gv.intermittent):
            # Intermittent sources have cost of zero
            production_costs[select_source,y,:] = 0.0
            
        else:
            # Use heat rates and gas/coal prices to create base price
            production_costs[select_source,y,:] = heat_rates[select_source,np.newaxis] * (np.isin(source, np.array([gv.coal])) * coal_prices[np.newaxis,sample_indices] + np.isin(source, gv.natural_gas) * gas_prices[np.newaxis,sample_indices])
            
            # Add on the cost shock from estimated distribution
            if np.isin(source, production_cost_est_sources):
                production_costs[select_source,y,:] = production_costs[select_source,y,:] + production_cost_shocks[select_source,:]
            
# Fill in the year-sources that didn't have any observations
for s, source in enumerate(np.unique(energy_sources)):
    select_source = energy_sources == source
    nan_yr = np.any(np.isnan(cap_factors[select_source,:,:]), axis=(0,2))
    if np.any(nan_yr): # if there is some year where they are all NaN
        for y, year in enumerate(years_unique):
            if nan_yr[y]:
                non_nan_yr_idx = np.arange(years_unique.shape[0])[~nan_yr]
                closest_non_nan_yr = non_nan_yr_idx[np.argmin(np.abs(non_nan_yr_idx - y))]
                print(f"{source} in year {year} has NaNs, replacing with {years_unique[closest_non_nan_yr]}.", flush=True)
                cap_factors[select_source,y,:] = np.copy(cap_factors[select_source,closest_non_nan_yr,:]) # just use the sample from the closest non-NaN year

# Create available capacities array
available_capacities = cap_factors * capacities[:,np.newaxis,np.newaxis] / 2.0 # / 2.0 to put in half-hour intervals

# %%
# Determine the demand shocks based on demand realizations and prices

# Determine the average price in each year
avg_wholesale_price = np.ones(num_years) * np.nan
xis = np.ones(demand_realizations.shape) * np.nan
for y, year in enumerate(years_unique):
    select_yr = years == year
    avg_wholesale_price[y] = np.average(prices[select_yr], weights=total_load_all[select_yr]) # use the load that includes even the dropped generators
    xis[y,:] = demand.q_demanded_inv(tariffs_sample[y,:], demand_elasticity, demand_realizations[y,:])
print(f"Backed out demand shocks from demand realizations.", flush=True)

# %%
# Reshape array, we will shape it back later
array_state_in = np.reshape(array_state_in, (array_state_in.shape[0], -1)) # collapse down to 1 dimension for the state

# %%
# Create array of which state we are in in the data in each year

# Determine the enter/exit years
enter_dates_years_ = pd.to_datetime(entered_date).year.values
enter_dates_months_ = pd.to_datetime(entered_date).month.values
enter_dates_years_[enter_dates_months_ < 10] = enter_dates_years_[enter_dates_months_ < 10] - 1 # years we're using are WEM year definitions

exit_dates_years_ = pd.to_datetime(exited_date).year.values
exit_dates_months_ = pd.to_datetime(exited_date).month.values
exit_dates_years_[(exit_dates_years_ == np.max(exit_dates_years_)) & (exit_dates_months_ > (np.max(exit_dates_months_[exit_dates_years_ == np.max(exit_dates_years_)]) - 3))] = 10000 # this just needs to be a big number to signal hasn't exited
exit_dates_years_[exit_dates_months_ < 10] = exit_dates_years_[exit_dates_months_ < 10] - 1 # years we're using are WEM year definitions

# Adjust exit dates for MUJA_G5 & MUJA_G6 since MUJA_G5 happens at very end of sample and MUJA_G6 decision made at same time but scheduled for a year later
exit_dates_years_[np.isin(facilities[~pd.Series(facilities).str.contains("new").values], generator_groupings['MUJA_G5-6'])] = 2022
years_unique_data = np.arange(np.min(years_unique), np.maximum(np.max(years_unique), 2022) + 1)

# Adjust enter date for GREENOUGH_RIVER_PV1 b/c that's when the major expansion happened
enter_dates_years_[facilities[~pd.Series(facilities).str.contains("new").values] == "GREENOUGH_RIVER_PV1"] = 2018

# Add 
enter_dates_years = np.ones(facilities.shape, dtype=int) * 9999 # large number b/c new generators will have never entered
exit_dates_years = np.ones(facilities.shape, dtype=int) * 9999 # large number b/c new generators will have never exited
enter_dates_years[~pd.Series(facilities).str.contains("new").values] = enter_dates_years_
exit_dates_years[~pd.Series(facilities).str.contains("new").values] = exit_dates_years_

# Adjust generator entrance/exit that is joint but happens in different years that we want to treat as one
adjust_enter_years = np.zeros((facilities.shape[0],), dtype=bool)
enter_dates_years[np.isin(facilities, generator_groupings['ALINTA_WGP'])] = int(np.ceil(np.mean(enter_dates_years[np.isin(facilities, generator_groupings['ALINTA_WGP'])])))
adjust_enter_years[np.isin(facilities, generator_groupings['ALINTA_WGP'])] = True
adjust_exit_years = np.zeros((facilities.shape[0],), dtype=bool)
exit_dates_years[np.isin(facilities, generator_groupings['KWINANA_G5-6'])] = int(np.ceil(np.mean(exit_dates_years[np.isin(facilities, generator_groupings['KWINANA_G5-6'])])))
adjust_exit_years[np.isin(facilities, generator_groupings['KWINANA_G5-6'])] = True
adjust_exit_years[np.isin(facilities, generator_groupings['MUJA_G5-6'])] = True

# Determine which state we are in in the data
years_data_inout_start = np.zeros((facilities.shape[0], years_unique_data.shape[0]), dtype=bool)
years_data_inout_choice = np.zeros((facilities.shape[0], years_unique_data.shape[0]), dtype=bool)
for y, year in enumerate(years_unique_data):
    years_data_inout_choice[:,y] = (exit_dates_years > year) & (enter_dates_years <= year) # must have exited after end of year and entered within the year or before to be included in state
years_data_inout_start[:,1:] = years_data_inout_choice[:,:-1] # where we are starting in the next period is just where we ended up in the previous one
years_data_inout_start[:,0] = enter_dates_years <= np.min(years_unique_data) # the starting case is any generator that was in the market at the very beginning
data_state_compare_start = np.all(years_data_inout_start[:,np.newaxis,:] == array_state_in[:,:,np.newaxis], axis=0)
data_state_compare_choice = np.all(years_data_inout_choice[:,np.newaxis,:] == array_state_in[:,:,np.newaxis], axis=0)
data_state_idx_start = np.where(data_state_compare_start)[0] # array of the (flattened) state index that we see at start of each year
data_state_idx_choice = np.where(data_state_compare_choice)[0] # array of the (flattened) state index that we see in each year

# Create arrays of strategic firm and competitive participants starting states
data_state_idx_start_strategic = data_state_idx_start # this is very straightforward, it's just whatever was the start of the year
data_state_idx_start_strategic_unraveled = np.array(list(np.unravel_index(data_state_idx_start_strategic, state_shape_list)))
data_state_idx_start_unraveled = np.array(list(np.unravel_index(data_state_idx_start, state_shape_list)))
data_state_idx_choice_unraveled = np.array(list(np.unravel_index(data_state_idx_choice, state_shape_list)))
data_state_idx_start_competitive_unraveled = np.copy(data_state_idx_choice_unraveled) # when competitive firms make their choice, they have already seen what strategic firms choose
competitive_dims = np.zeros(len(state_shape_list), dtype=bool)
competitive_dims[-list_sources_state.shape[0]:] = True
data_state_idx_start_competitive_unraveled[competitive_dims,:] = data_state_idx_start_unraveled[competitive_dims,:] # use the starting state for the competitive firms
data_state_idx_start_competitive = np.ravel_multi_index(data_state_idx_start_competitive_unraveled, state_shape_list)

# Create arrays of strategic firm and competitive participants' choices
data_state_idx_choice_competitive = data_state_idx_choice # this is very straightforward, it's just whatever was the end result of the year
data_state_idx_choice_competitive_unraveled = np.array(list(np.unravel_index(data_state_idx_choice_competitive, state_shape_list)))
data_state_idx_choice_strategic_unraveled = np.copy(data_state_idx_start_unraveled)
strategic_dims = ~competitive_dims
data_state_idx_choice_strategic_unraveled[strategic_dims,:] = data_state_idx_choice_unraveled[strategic_dims,:]
data_state_idx_choice_strategic = np.ravel_multi_index(data_state_idx_choice_strategic_unraveled, state_shape_list)

state_shape_arr = np.array(state_shape_list)
state_shape_arr_gr1 = state_shape_arr[state_shape_arr > 1] # the dimensions that can actually change
indices_unraveled = np.array(list(np.unravel_index(np.arange(np.prod(state_shape_list)), state_shape_arr_gr1)))

# # Determine what the indices are of adjustment with one-step choices for competitive firms
# state_shape_arr_competitive = state_shape_arr[competitive_dims]
# state_shape_arr_competitive = state_shape_arr_competitive[state_shape_arr_competitive > 1]
# num_dims_changes_competitive = state_shape_arr_competitive.shape[0]
# num_options = 3**num_dims_changes_competitive # number of possible combinations of changes in one period
# add_vals = np.zeros((indices_unraveled.shape[0], num_options), dtype=int) # initialize array of what options are in each period
# add_vals_competitive = np.array(list(np.unravel_index(np.arange(num_options), tuple([3 for i in range(num_dims_changes_competitive)])))) - 1 # flattened version of all the possible ways in which can make a one-step adjustment in that direction, -1 gives us -1, 0 and 1 (instead of 0,1,2)
# add_vals[competitive_dims[state_shape_arr > 1],:] = add_vals_competitive
# unraveled_indices_addon = indices_unraveled[:,:,np.newaxis] + add_vals[:,np.newaxis,:]
# adjusted_indices_competitive = np.ravel_multi_index(unraveled_indices_addon, state_shape_arr_gr1, mode="wrap")
# adjusted_indices_competitive[np.any(unraveled_indices_addon >= state_shape_arr_gr1[:,np.newaxis,np.newaxis], axis=0) | np.any(unraveled_indices_addon < 0, axis=0)] = 99999999999 # large number so doesn't correspond to any index
# relevant_adjusted_indices_competitive = np.take_along_axis(adjusted_indices_competitive, data_state_idx_start_competitive[:,np.newaxis], axis=0)
# compare_indices_competitive = relevant_adjusted_indices_competitive == data_state_idx_choice_competitive[:,np.newaxis]
# indices_adjustment_competitive = np.where(compare_indices_competitive)[1]

# # Determine what the indices are of adjustment with one-step choices for strategic firms
# state_shape_arr_strategic = state_shape_arr[strategic_dims]
# state_shape_arr_strategic = state_shape_arr_strategic[state_shape_arr_strategic > 1]
# num_dims_changes_strategic = state_shape_arr_strategic.shape[0]
# num_options = 3**num_dims_changes_strategic # number of possible combinations of changes in one period
# add_vals = np.zeros((indices_unraveled.shape[0], num_options), dtype=int) # initialize array of what options are in each period
# add_vals_strategic = np.array(list(np.unravel_index(np.arange(num_options), tuple([3 for i in range(num_dims_changes_strategic)])))) - 1 # flattened version of all the possible ways in which can make a one-step adjustment in that direction, -1 gives us -1, 0 and 1 (instead of 0,1,2)
# add_vals[strategic_dims[state_shape_arr > 1],:] = add_vals_strategic
# unraveled_indices_addon = indices_unraveled[:,:,np.newaxis] + add_vals[:,np.newaxis,:]
# adjusted_indices_strategic = np.ravel_multi_index(unraveled_indices_addon, state_shape_arr_gr1, mode="wrap")
# adjusted_indices_strategic[np.any(unraveled_indices_addon >= state_shape_arr_gr1[:,np.newaxis,np.newaxis], axis=0) | np.any(unraveled_indices_addon < 0, axis=0)] = 99999999999 # large number so doesn't correspond to any index
# relevant_adjusted_indices_strategic = np.take_along_axis(adjusted_indices_strategic, data_state_idx_start_strategic[:,np.newaxis], axis=0)
# compare_indices_strategic = relevant_adjusted_indices_strategic == data_state_idx_choice_strategic[:,np.newaxis]
# indices_adjustment_strategic = np.where(compare_indices_strategic)[1]

# Determine the choice indices of each individual strategic firm
indices_adjustment_strategic_by_firm = np.zeros((data_state_idx_choice_strategic.shape[0], list_firms_state.shape[0] - 1), dtype=int)
for i in range(list_firms_state.shape[0] - 1): # - 1 b/c not including competitive fringe
    # Select which indices are relevant for firm i
    select_dims_i = (np.arange(state_shape_arr.shape[0]) >= (i * list_sources_state.shape[0])) & (np.arange(state_shape_arr.shape[0]) < ((i + 1) * list_sources_state.shape[0]))
    select_dims_i_gr1 = select_dims_i[state_shape_arr > 1]
    state_shape_arr_i = state_shape_arr[select_dims_i]
    state_shape_arr_i = state_shape_arr_i[state_shape_arr_i > 1]
    num_dims_changes_i = state_shape_arr_i.shape[0]
    num_options = 3**num_dims_changes_i
    add_vals = np.zeros((indices_unraveled.shape[0], num_options), dtype=int) # initialize array of what options are in each period
    add_vals_i = np.array(list(np.unravel_index(np.arange(num_options), tuple([3 for i in range(num_dims_changes_i)])))) - 1
    add_vals[select_dims_i_gr1,:] = add_vals_i
    unraveled_indices_addon = indices_unraveled[:,:,np.newaxis] + add_vals[:,np.newaxis,:]
    adjusted_indices_i = np.ravel_multi_index(unraveled_indices_addon, state_shape_arr_gr1, mode="wrap")
    adjusted_indices_i[np.any(unraveled_indices_addon >= state_shape_arr_gr1[:,np.newaxis,np.newaxis], axis=0) | np.any(unraveled_indices_addon < 0, axis=0)] = 99999999999 # large number so doesn't correspond to any index
    
    # Make it so the starting index includes the choices of the other firms
    data_state_idx_start_i_unraveled = np.copy(data_state_idx_choice_strategic_unraveled[state_shape_arr > 1,:])
    data_state_idx_start_i_unraveled[select_dims_i_gr1] = np.copy(data_state_idx_start_strategic_unraveled[state_shape_arr > 1,:][select_dims_i_gr1,:])
    data_state_idx_start_i = np.ravel_multi_index(data_state_idx_start_i_unraveled, state_shape_arr_gr1)
    
    # Add the choice indices of each firm
    relevant_adjusted_indices_i = np.take_along_axis(adjusted_indices_i, data_state_idx_start_i[:,np.newaxis], axis=0)
    compare_indices_i = relevant_adjusted_indices_i == data_state_idx_choice_strategic[:,np.newaxis]
    print(f"Firm {i} successful?", end=" ", flush=True)
    if np.all(np.sum(compare_indices_i, axis=1) == 1):
        print(f"Yes", flush=True)
    else:
        print(f"No", flush=True)
    indices_adjustment_strategic_by_firm[:,i] = np.where(compare_indices_i)[1]

# Determine the choice indices of each fringe source
competitive_sources_gr1 = list_sources_state[state_shape_arr[competitive_dims] > 1]
indices_adjustment_competitive_by_source = np.zeros((data_state_idx_choice_competitive.shape[0], competitive_sources_gr1.shape[0]), dtype=int)
for i in range(competitive_sources_gr1.shape[0]):
    # Select which indices are relevant for source i
    select_dims_i_gr1 = np.arange(state_shape_arr_gr1.shape[0]) == (state_shape_arr_gr1.shape[0] - competitive_sources_gr1.shape[0] + i)
    state_shape_arr_i = state_shape_arr_gr1[select_dims_i_gr1]
    num_dims_changes_i = state_shape_arr_i.shape[0]
    num_options = 3**num_dims_changes_i
    add_vals = np.zeros((indices_unraveled.shape[0], num_options), dtype=int) # initialize array of what options are in each period
    add_vals_i = np.array(list(np.unravel_index(np.arange(num_options), tuple([3 for i in range(num_dims_changes_i)])))) - 1 # flattened version of all the possible ways in which can make a one-step adjustment in that direction
    add_vals[select_dims_i_gr1,:] = add_vals_i
    unraveled_indices_addon = indices_unraveled[:,:,np.newaxis] + add_vals[:,np.newaxis,:]
    adjusted_indices_i = np.ravel_multi_index(unraveled_indices_addon, state_shape_arr_gr1, mode="wrap")
    adjusted_indices_i[np.any(unraveled_indices_addon >= state_shape_arr_gr1[:,np.newaxis,np.newaxis], axis=0) | np.any(unraveled_indices_addon < 0, axis=0)] = 99999999999 # large number so doesn't correspond to any index
    
    # Make it so the starting index includes the choices of the other sources
    data_state_idx_start_i_unraveled = np.copy(data_state_idx_choice_competitive_unraveled[state_shape_arr > 1,:])
    data_state_idx_start_i_unraveled[select_dims_i_gr1] = np.copy(data_state_idx_start_competitive_unraveled[state_shape_arr > 1,:][select_dims_i_gr1,:])
    data_state_idx_start_i = np.ravel_multi_index(data_state_idx_start_i_unraveled, state_shape_arr_gr1)
    
    # Add the choice indices of each firm
    relevant_adjusted_indices_i = np.take_along_axis(adjusted_indices_i, data_state_idx_start_i[:,np.newaxis], axis=0)
    compare_indices_i = relevant_adjusted_indices_i == data_state_idx_choice_competitive[:,np.newaxis]
    print(f"Source {i} successful?", end=" ", flush=True)
    if np.all(np.sum(compare_indices_i, axis=1) == 1):
        print(f"Yes", flush=True)
    else:
        print(f"No", flush=True)
    indices_adjustment_competitive_by_source[:,i] = np.where(compare_indices_i)[1]

print(f"Determined the state in the data in each year.", flush=True)

# %%
# Check that our state space definition is good

# Check that all facilities are included in the states
print(f"Are all facilities included in at least one state?", end=" ", flush=True)
if np.all(np.max(array_state_in, axis=1)):
    print(f"Yes", flush=True)
else:
    print(f"No", flush=True)

# Check that this definition of the state space incorporates all of the states we observe in the data
print(f"Do the facilities in each year correspond to exactly one state?", end=" ", flush=True)
if np.all(np.sum(data_state_compare_start, axis=0) == 1) and np.all(np.sum(data_state_compare_choice, axis=0) == 1):
    print(f"Yes", flush=True)
else:
    print(f"No", flush=True)
    
# %%
# Create table describing generators in market

info_sources = np.load(gv.info_sources, allow_pickle=True)

# Begin table
tex_table = ""
tex_table += f"\\begin{{tabular}}{{ lllccccc }} \n"
tex_table += f" & & & \multicolumn{{1}}{{c}}{{Capacity}} & \multicolumn{{1}}{{c}}{{Heat Rate}} & \multicolumn{{1}}{{c}}{{Emissions Rate}} & \multicolumn{{1}}{{c}}{{Entered}} & \multicolumn{{1}}{{c}}{{Exit}} \\\\ \n"
tex_table += f"\multicolumn{{1}}{{c}}{{Generator}} & \multicolumn{{1}}{{c}}{{Firm}} & \multicolumn{{1}}{{c}}{{Technology}} & \multicolumn{{1}}{{c}}{{(MW)}} & \multicolumn{{1}}{{c}}{{(GJ/MWh)}} & \multicolumn{{1}}{{c}}{{(kg$\\text{{CO}}_{{2}}$-eq/MWh)}} & \multicolumn{{1}}{{c}}{{Year}} & \multicolumn{{1}}{{c}}{{Year}} \\\\ \n"
tex_table += "\\hline \n"

# Add data
for i in range(facilities.shape[0]):
    if "new" in facilities[i]: # we only want to create the table for generators that actually showed up in the data
        continue
    tex_table += f"{facilities[i]} & ".replace("_", "\\_")
    tex_table += f"{participants_originalnames[i]} & "
    energy_source = energy_sources[i]
    if energy_source == gv.gas_ccgt:
        energy_source = "CCGT"
    if energy_source == gv.gas_ocgt:
        energy_source = "OCGT"
    if energy_source == gv.gas_cogen:
        energy_source = "OCGT"
    tex_table += f"{energy_source}"
    if ("KWINANA" in facilities[i]) and ("KWINANA_GT" not in facilities[i]) and ("CCG" not in facilities[i]):
        tex_table += f"${{}}^{{*}}$"
    tex_table += f" & "
    tex_table += f"{capacities[i]:,.0f} & ".replace(",", "\\,")
    tex_table += f"{heat_rates[i]:,.1f}" if not np.isnan(heat_rates[i]) else "--"
    if np.isin(info_sources[facilities_orig == facilities[i]], np.array(["skm_impute", "skm_confidential"])):
        tex_table += f"${{}}^{{\\dagger}}$"
    tex_table += f" & "
    tex_table += f"{co2_rates[i]:,.1f}".replace(",", "\\,")
    if np.isin(info_sources[facilities_orig == facilities[i]], np.array(["skm_impute", "skm_confidential"])):
        tex_table += f"${{}}^{{\\dagger}}$"
    tex_table += f" & "
    tex_table += f"{enter_dates_years[i]}" if enter_dates_years[i] != np.min(enter_dates_years) else "--"
    if adjust_enter_years[i]:
        tex_table += f"${{}}^{{\\ddagger}}$"
    if facilities[i] == "GREENOUGH_RIVER_PV1":
        tex_table += f"${{}}^{{\\S}}$"
    tex_table += f" & "
    tex_table += f"{exit_dates_years[i]}" if exit_dates_years[i] != np.max(exit_dates_years) else "--"
    if adjust_exit_years[i]:
        tex_table += f"${{}}^{{\\ddagger}}$"
    tex_table += f" \\\\ \n"

# End table
tex_table += f"\\hline \n \\end{{tabular}} \n"
    
print(tex_table, flush=True)

if running_specification == 0:
    create_file(gv.tables_path + "generator_list.tex", tex_table)

# %%
# Are the capacities of the facilities not included below the threshold?

not_included = ~np.isin(facilities_orig, facilities)
facilities_not_included = facilities_orig[not_included]
energy_sources_not_included = np.load(gv.energy_sources_file)[not_included]
capacities_not_included = np.load(gv.capacities_file)[not_included]
energy_sources_not_included_unique = np.unique(energy_sources_not_included)
max_capacity = np.zeros(energy_sources_not_included_unique.shape[0])
for i, source in enumerate(energy_sources_not_included_unique):
    max_capacity[i] = np.max(capacities_not_included[energy_sources_not_included == source])
print(f"maximum capacity for not included solar/wind: {np.round(np.max(max_capacity[np.isin(energy_sources_not_included_unique, gv.intermittent)]), 1)}", flush=True)
print(f"maximum capacity for not included gas/coal: {np.round(np.max(max_capacity[np.isin(energy_sources_not_included_unique, np.concatenate((gv.natural_gas, np.array([gv.coal]))))]), 1)}", flush=True)
    
# %%
# Determine the equilibrium and relevant variables in each state
if running_specification == 0:
    print(f"Starting solving for equilibria...", flush=True)
start_task = time.time()

# Rename facilities, participants, and energy sources to integers (will be quicker for comparisons than with strings)
facilities_unique, facilities_int = np.unique(facilities, return_inverse=True)
participants_unique, participants_int = np.unique(participants, return_inverse=True)
energy_sources_unique, energy_sources_int = np.unique(energy_sources, return_inverse=True)
facilities_int_unique = np.unique(facilities_int)
participants_int_unique = np.unique(participants_int)
energy_sources_int_unique = np.unique(energy_sources_int)

# Initialize arrays
profits = np.zeros((array_state_in.shape[1], num_years, participants_int_unique.shape[0]))
emissions = np.zeros((array_state_in.shape[1], num_years))
blackouts = np.zeros((array_state_in.shape[1], num_years))
frac_by_source = np.zeros((array_state_in.shape[1], num_years, energy_sources_int_unique.shape[0]))
quantity_weighted_avg_price = np.zeros((array_state_in.shape[1], num_years))
total_produced = np.zeros((array_state_in.shape[1], num_years))
misallocated_demand = np.zeros((array_state_in.shape[1], num_years))
consumer_surplus = np.zeros((array_state_in.shape[1], num_years))

# Add on carbon taxes
production_costs_w_carbon_tax = production_costs + (co2_rates[:,np.newaxis,np.newaxis] * carbon_taxes_rates[np.newaxis,:])

# Determine number of half hours that are in a year
num_half_hours = gv.num_intervals_in_day * 365.0

def compute_eqm(indices):
    i, y = indices[0], indices[1]
    index_dimensions = (array_state_in.shape[1],years_unique.shape[0])
    raveled_idx = np.ravel_multi_index(indices, index_dimensions)
    if raveled_idx % 1000000 == 0:
        print(f"\t{raveled_idx} / {np.prod(index_dimensions)}", flush=True)
    select_gens = array_state_in[:,i]
    return eqm.expected_profits(available_capacities[select_gens,y,:], production_costs_w_carbon_tax[select_gens,y,:], production_costs[select_gens,y,:], participants_int[select_gens], participants_int_unique, energy_sources_int[select_gens], energy_sources_int_unique, co2_rates[select_gens], price_cap[y,:], fixed_tariff_component, demand_elasticity, xis[y,:], avg_wholesale_price[y], num_half_hours)
        
# Initialize multiprocessing
if running_specification == 0:
    pool = Pool(num_cpus)
    chunksize = 4
    for ind, res in enumerate(pool.imap(compute_eqm, product(range(array_state_in.shape[1]), range(years_unique.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        profits.flat[idx*participants_int_unique.shape[0]:(idx+1)*participants_int_unique.shape[0]] = res[0]
        emissions.flat[idx] = res[1]
        blackouts.flat[idx] = res[2]
        frac_by_source.flat[idx*energy_sources_int_unique.shape[0]:(idx+1)*energy_sources_int_unique.shape[0]] = res[3]
        quantity_weighted_avg_price.flat[idx] = res[4]
        total_produced.flat[idx] = res[5]
        misallocated_demand.flat[idx] = res[6]
        consumer_surplus.flat[idx] = res[7]
    pool.close()

    print(f"Completed solving for equilibrium in each year in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Determine capacity payments
if running_specification == 0:
    print(f"Starting solving for capacity payments...", flush=True)
start_task = time.time()

# Process capacity prices
cap_years = pd.to_datetime(cap_date_from).year.values
bf_sample = cap_years < np.min(years_unique)
cap_years = cap_years[~bf_sample]
capacity_price = capacity_price[~bf_sample]

# Make cap factors repeat for any years past last year of data
cap_factors_extend = np.concatenate((cap_factors, np.tile(cap_factors[:,-1,:][:,np.newaxis,:], (1,np.max(cap_years) - np.max(years_unique),1))), axis=1)

capacity_payments = np.zeros((array_state_in.shape[1], cap_years.shape[0], participants_int_unique.shape[0]))

# Commitments of each generator
expected_payments_permw_perdollar = 1.0 * ~np.isin(energy_sources_int, np.arange(energy_sources_unique.shape[0])[np.isin(energy_sources_unique, gv.intermittent)])[:,np.newaxis] # np.mean(cap_factors_extend, axis=2)

def compute_payments(indices):
    i = indices[0]
    index_dimensions = (array_state_in.shape[1],)
    raveled_idx = np.ravel_multi_index(indices, index_dimensions)
    if raveled_idx % 1000000 == 0:
        print(f"\t{raveled_idx} / {np.prod(index_dimensions)}", flush=True)
    select_gens = array_state_in[:,i]
    return cc.expected_cap_payment(expected_payments_permw_perdollar[select_gens,:], capacities[select_gens], capacity_price, participants_int[select_gens], participants_int_unique)
        
# Initialize multiprocessing
if running_specification == 0:
    pool = Pool(num_cpus)
    chunksize = 4
    for ind, res in enumerate(pool.imap(compute_payments, product(range(array_state_in.shape[1]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        capacity_payments.flat[idx*(cap_years.shape[0]*participants_int_unique.shape[0]):(idx+1)*(cap_years.shape[0]*participants_int_unique.shape[0])] = res[0]
    pool.close()

    print(f"Completed solving for capacity payments in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Save arrays
if running_specification == 0:
    np.savez_compressed(f"{gv.arrays_path}data_env.npz", 
                        profits=profits, 
                        emissions=emissions, 
                        blackouts=blackouts, 
                        frac_by_source=frac_by_source, 
                        quantity_weighted_avg_price=quantity_weighted_avg_price, 
                        total_produced=total_produced, 
                        misallocated_demand=misallocated_demand, 
                        consumer_surplus=consumer_surplus, 
                        capacity_payments=capacity_payments)
    np.savez_compressed(f"{gv.arrays_path}state_space.npz", 
                        facilities_unique=facilities_unique, 
                        facilities_int=facilities_int, 
                        facilities_int_unique=facilities_int_unique, 
                        participants_unique=participants_unique, 
                        participants_int=participants_int, 
                        participants_int_unique=participants_int_unique, 
                        energy_sources_unique=energy_sources_unique, 
                        energy_sources_int=energy_sources_int, 
                        energy_sources_int_unique=energy_sources_int_unique, 
                        capacities=capacities, 
                        years_unique=years_unique, 
                        cap_years=cap_years, 
                        array_state_in=array_state_in, 
                        state_shape_list=np.array(state_shape_list), 
                        data_state_idx_start_competitive=data_state_idx_start_competitive, 
                        data_state_idx_start_strategic=data_state_idx_start_strategic, 
                        data_state_idx_choice_competitive=data_state_idx_choice_competitive, 
                        data_state_idx_choice_strategic=data_state_idx_choice_strategic, 
                        indices_adjustment_strategic_by_firm=indices_adjustment_strategic_by_firm, 
                        indices_adjustment_competitive_by_source=indices_adjustment_competitive_by_source)

# %%
# Create price caps for counterfactuals
low_price_cap_counterfactuals = 300.0
high_price_cap_counterfactuals = 1000.0
if running_specification == 0:
    create_file(gv.stats_path + "low_price_cap_counterfactuals.tex", f"{int(low_price_cap_counterfactuals):,}".replace(",", "\\,"))
price_cap_counterfactuals = low_price_cap_counterfactuals * np.ones(price_cap.shape)
if running_specification == 3:
    price_cap_counterfactuals = high_price_cap_counterfactuals * np.ones(price_cap.shape)
    create_file(gv.stats_path + "high_price_cap_counterfactuals.tex", f"{int(high_price_cap_counterfactuals):,}".replace(",", "\\,"))

# %%
# Carbon tax counterfactual
if (running_specification == 1) or (running_specification == 3):
    print(f"Starting solving for carbon tax counterfactuals...", flush=True)
start_task = time.time()
carbon_taxes_linspace = np.linspace(0.0, 300.0, 7) # this is in AUD / ton CO2, need to put it in / kgCO2 later when use in equilibrium calculation

# Initialize arrays
profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years, participants_int_unique.shape[0]))
emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years))
blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years))
frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years, energy_sources_int_unique.shape[0]))
quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years))
total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years))
misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years))
consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years))
total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_years))

def compute_eqm(indices):
    c, i, y = indices[0], indices[1], indices[2]
    index_dimensions = (carbon_taxes_linspace.shape[0],array_state_in.shape[1],years_unique.shape[0])
    raveled_idx = np.ravel_multi_index(indices, index_dimensions)
    if raveled_idx % 1000000 == 0:
        print(f"\t{raveled_idx} / {np.prod(index_dimensions)}", flush=True)
    production_costs_w_carbon_tax = production_costs + (co2_rates[:,np.newaxis,np.newaxis] * carbon_taxes_linspace[c] / 1000.0)
    select_gens = array_state_in[:,i]
    return eqm.expected_profits(available_capacities[select_gens,y,:], production_costs_w_carbon_tax[select_gens,y,:], production_costs[select_gens,y,:], participants_int[select_gens], participants_int_unique, energy_sources_int[select_gens], energy_sources_int_unique, co2_rates[select_gens], price_cap_counterfactuals[y,:], fixed_tariff_component, demand_elasticity, xis[y,:], avg_wholesale_price[y], num_half_hours)
            

# Initialize multiprocessing
if (running_specification == 1) or (running_specification == 3):
    pool = Pool(num_cpus)
    chunksize = 4
    for ind, res in enumerate(pool.imap(compute_eqm, product(range(carbon_taxes_linspace.shape[0]), range(array_state_in.shape[1]), range(years_unique.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        profits.flat[idx*participants_int_unique.shape[0]:(idx+1)*participants_int_unique.shape[0]] = res[0]
        emissions.flat[idx] = res[1]
        blackouts.flat[idx] = res[2]
        frac_by_source.flat[idx*energy_sources_int_unique.shape[0]:(idx+1)*energy_sources_int_unique.shape[0]] = res[3]
        quantity_weighted_avg_price.flat[idx] = res[4]
        total_produced.flat[idx] = res[5]
        misallocated_demand.flat[idx] = res[6]
        consumer_surplus.flat[idx] = res[7]
        total_production_cost.flat[idx] = res[9]
    pool.close()

    print(f"Completed the carbon tax counterfactual profits in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Save arrays
if running_specification == 1:
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_co2tax.npz", 
                        carbon_taxes_linspace=carbon_taxes_linspace, 
                        profits=profits, 
                        emissions=emissions, 
                        blackouts=blackouts, 
                        frac_by_source=frac_by_source, 
                        quantity_weighted_avg_price=quantity_weighted_avg_price, 
                        total_produced=total_produced, 
                        misallocated_demand=misallocated_demand, 
                        consumer_surplus=consumer_surplus, 
                        total_production_cost=total_production_cost)
if running_specification == 3:
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_co2tax_highpricecap.npz", 
                        carbon_taxes_linspace=carbon_taxes_linspace, 
                        profits=profits, 
                        emissions=emissions, 
                        blackouts=blackouts, 
                        frac_by_source=frac_by_source, 
                        quantity_weighted_avg_price=quantity_weighted_avg_price, 
                        total_produced=total_produced, 
                        misallocated_demand=misallocated_demand, 
                        consumer_surplus=consumer_surplus, 
                        total_production_cost=total_production_cost)

# %%
# Capacity payment counterfactual
if running_specification == 1:
    print(f"Starting solving for capacity payment counterfactuals...", flush=True)
start_task = time.time()
capacity_payments_linspace = np.linspace(0.0, 200000.0, 5)

# Process capacity payment program parameters
# refund_multipliers = np.ones((facilities.shape[0], years_unique.shape[0])) * (gv.refund_multiplier_intermittent * np.isin(energy_sources, gv.intermittent) + gv.refund_multiplier_scheduled * ~np.isin(energy_sources, gv.intermittent))[:,np.newaxis] # don't need to extension past last year in data b/c the capacity price isn't changing

capacity_payments = np.zeros((capacity_payments_linspace.shape[0],  array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))

# Solve for optimal commitments of each generator
expected_payments_permw_perdollar = 1.0 * ~np.isin(energy_sources_int, np.arange(energy_sources_unique.shape[0])[np.isin(energy_sources_unique, gv.intermittent)])[:,np.newaxis] # np.mean(cap_factors_extend, axis=2)

def compute_payments(indices):
    c, i = indices[0], indices[1]
    index_dimensions = (capacity_payments_linspace.shape[0],array_state_in.shape[1])
    raveled_idx = np.ravel_multi_index(indices, index_dimensions)
    if raveled_idx % 1000000 == 0:
        print(f"\t{raveled_idx} / {np.prod(index_dimensions)}", flush=True)
    select_gens = array_state_in[:,i]
    capacity_price_years = np.ones(years_unique.shape[0]) * capacity_payments_linspace[c]
    return cc.expected_cap_payment(expected_payments_permw_perdollar[select_gens,:], capacities[select_gens], capacity_price_years, participants_int[select_gens], participants_int_unique)
        
# Initialize multiprocessing
if running_specification == 1:
    pool = Pool(num_cpus)
    chunksize = 4
    for ind, res in enumerate(pool.imap(compute_payments, product(range(capacity_payments_linspace.shape[0]), range(array_state_in.shape[1]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        capacity_payments.flat[idx*(years_unique.shape[0]*participants_int_unique.shape[0]):(idx+1)*(years_unique.shape[0]*participants_int_unique.shape[0])] = res[0]
    pool.close()

    print(f"Completed the capacity payment counterfactual payments in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Save arrays
if running_specification == 1:
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz", 
                        capacity_payments_linspace=capacity_payments_linspace, 
                        capacity_payments=capacity_payments)
    
# %%
# Renewable subsidy counterfactual

if running_specification == 2:
    print(f"Starting solving for renewable subsidy counterfactuals...", flush=True)
start_task = time.time()
renewable_subsidies_linspace = np.linspace(0.0, 150.0, 7)

# Initialize arrays
profits = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years, participants_int_unique.shape[0]))
emissions = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years))
blackouts = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years))
frac_by_source = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years, energy_sources_int_unique.shape[0]))
quantity_weighted_avg_price = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years))
total_produced = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years))
misallocated_demand = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years))
consumer_surplus = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years))
renewable_production = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years))
total_production_cost = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_years))

def compute_eqm(indices):
    s, i, y = indices[0], indices[1], indices[2]
    index_dimensions = (renewable_subsidies_linspace.shape[0],array_state_in.shape[1],years_unique.shape[0])
    raveled_idx = np.ravel_multi_index(indices, index_dimensions)
    if raveled_idx % 1000000 == 0:
        print(f"\t{raveled_idx} / {np.prod(index_dimensions)}", flush=True)
    production_costs_w_renewable_subsidies = production_costs - renewable_subsidies_linspace[s] * np.isclose(co2_rates, 0.0)[:,np.newaxis,np.newaxis]
    select_gens = array_state_in[:,i]
    return eqm.expected_profits(available_capacities[select_gens,y,:], production_costs_w_renewable_subsidies[select_gens,y,:], production_costs[select_gens,y,:], participants_int[select_gens], participants_int_unique, energy_sources_int[select_gens], energy_sources_int_unique, co2_rates[select_gens], price_cap_counterfactuals[y,:], fixed_tariff_component, demand_elasticity, xis[y,:], avg_wholesale_price[y], num_half_hours)
            

# Initialize multiprocessing
if running_specification == 2:
    pool = Pool(num_cpus)
    chunksize = 4
    for ind, res in enumerate(pool.imap(compute_eqm, product(range(renewable_subsidies_linspace.shape[0]), range(array_state_in.shape[1]), range(years_unique.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        profits.flat[idx*participants_int_unique.shape[0]:(idx+1)*participants_int_unique.shape[0]] = res[0]
        emissions.flat[idx] = res[1]
        blackouts.flat[idx] = res[2]
        frac_by_source.flat[idx*energy_sources_int_unique.shape[0]:(idx+1)*energy_sources_int_unique.shape[0]] = res[3]
        quantity_weighted_avg_price.flat[idx] = res[4]
        total_produced.flat[idx] = res[5]
        misallocated_demand.flat[idx] = res[6]
        consumer_surplus.flat[idx] = res[7]
        renewable_production.flat[idx] = res[8]
        total_production_cost.flat[idx] = res[9]
    pool.close()

    print(f"Completed the renewable subsidies counterfactual profits in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Save arrays
if running_specification == 2:
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_renewablesubisidies.npz", 
                        renewable_subsidies_linspace=renewable_subsidies_linspace, 
                        profits=profits, 
                        emissions=emissions, 
                        blackouts=blackouts, 
                        frac_by_source=frac_by_source, 
                        quantity_weighted_avg_price=quantity_weighted_avg_price, 
                        total_produced=total_produced, 
                        misallocated_demand=misallocated_demand, 
                        consumer_surplus=consumer_surplus, 
                        renewable_production=renewable_production, 
                        total_production_cost=total_production_cost)
