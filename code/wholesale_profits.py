# %%
# Import packages
import sys
from itertools import product

import time as time
from datetime import datetime

import numpy as np
import scipy.stats as stats
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from scipy.spatial.distance import cdist

import global_vars as gv
import wholesale.wholesale_profits as eqm
import wholesale.capacity_commitment as cc
import wholesale.demand as demand

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
dsp_quantities = np.load(gv.dsp_quantities_file)
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
demand_elasticity = np.load(gv.arrays_path + "demand_elasticity_estimates.npy")[-2] # we want the second-to-last specification
print(f"Finished import demand elasticity estimates.", flush=True)

# Production cost shock estimates
production_cost_est = np.load(gv.arrays_path + "production_cost_estimates.npy")
production_cost_est_sources = np.load(gv.arrays_path + "production_cost_estimates_sources.npy")
print(f"Finished importing production cost estimates.", flush=True)

# Battery parameters
battery_capacity = 2000.0 # total capacity
battery_flow = 1.0 / 8.0 * battery_capacity # how much can be charged/discharged within a half-hour (4-hour battery)
battery_delta = 0.9

# %%
# Save parameters

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()

# %%
# Set up parameters used throughout code

# Obtain SLURM array number
slurm_array_num = int(sys.argv[1])

# Parameters for generating wholesale profits
num_draws = 10
sample_length = 60
drop_before = 5
drop_after = 5
carbon_taxes_linspace = np.linspace(0.0, 300.0, 7) # this is in AUD / ton CO2, need to put it in / kgCO2 later when use in equilibrium calculation
renewable_subsidies_linspace = np.linspace(0.0, 150.0, 7) # this is in AUD / MWh
capacity_payments_linspace = np.linspace(0.0, 200000.0, 5)
capacity_payments_linspace_extended = np.linspace(0.0, 250000.0, 6)
years_list = np.unique(pd.to_datetime(dates).year)
num_years_in_year_grouping = 4
num_year_groupings = int(np.floor(years_list.shape[0] / num_years_in_year_grouping))
if slurm_array_num == 0:
    np.save(f"{gv.arrays_path}counterfactual_carbon_taxes_linspace.npy", carbon_taxes_linspace)
    np.save(f"{gv.arrays_path}counterfactual_renewable_subsidies_linspace.npy", renewable_subsidies_linspace)
    np.save(f"{gv.arrays_path}counterfactual_capacity_payments_linspace.npy", capacity_payments_linspace)
    np.save(f"{gv.arrays_path}counterfactual_capacity_payments_linspace_extended.npy", capacity_payments_linspace_extended)
    np.save(f"{gv.arrays_path}years_list.npy", years_list)
    np.save(f"{gv.arrays_path}num_years_in_year_grouping.npy", np.array([num_years_in_year_grouping]))
    create_file(f"{gv.stats_path}battery_flow.tex", f"{(battery_flow * 2):.0f}") # *2 since it's half-hour capacity/flow
    create_file(f"{gv.stats_path}battery_capacity.tex", f"{(battery_capacity * 2):.0f}")
    create_file(f"{gv.stats_path}battery_delta.tex", f"{(battery_delta * 100.0):.0f}")

# Convert SLURM array number to specification being run
specification_list_slurm_arrays = np.array([0] * num_year_groupings + [1] * num_year_groupings * carbon_taxes_linspace.shape[0] + [2] * num_year_groupings * renewable_subsidies_linspace.shape[0] + [3] * num_year_groupings * carbon_taxes_linspace.shape[0] + [4] * num_year_groupings * carbon_taxes_linspace.shape[0] + [5])
policy_list_slurm_arrays = np.concatenate(([0] * num_year_groupings, np.tile(np.arange(carbon_taxes_linspace.shape[0]), num_year_groupings), np.tile(np.arange(renewable_subsidies_linspace.shape[0]), num_year_groupings), np.tile(np.arange(carbon_taxes_linspace.shape[0]), num_year_groupings), np.tile(np.arange(carbon_taxes_linspace.shape[0]), num_year_groupings), [0]))
year_list_slurm_arrays = np.concatenate((np.tile(np.arange(num_year_groupings), 1 + carbon_taxes_linspace.shape[0] + renewable_subsidies_linspace.shape[0] + carbon_taxes_linspace.shape[0] + carbon_taxes_linspace.shape[0]), np.zeros(1, dtype=int)))
num_computation_groups_per_specification = 16
computation_group_list_slurm_arrays = np.arange(num_computation_groups_per_specification)
computation_group_list_slurm_arrays = np.tile(computation_group_list_slurm_arrays, (specification_list_slurm_arrays.shape[0],))
specification_list_slurm_arrays = np.repeat(specification_list_slurm_arrays, num_computation_groups_per_specification)
policy_list_slurm_arrays = np.repeat(policy_list_slurm_arrays, num_computation_groups_per_specification)
year_list_slurm_arrays = np.repeat(year_list_slurm_arrays, num_computation_groups_per_specification)

# Remove renewable subsidy for which subsidy is 0 b/c same as carbon tax is 0
renewable_subsidy_0 = (specification_list_slurm_arrays == 2) & (policy_list_slurm_arrays == 0)
specification_list_slurm_arrays = specification_list_slurm_arrays[~renewable_subsidy_0]
policy_list_slurm_arrays = policy_list_slurm_arrays[~renewable_subsidy_0]
year_list_slurm_arrays = year_list_slurm_arrays[~renewable_subsidy_0]
computation_group_list_slurm_arrays = computation_group_list_slurm_arrays[~renewable_subsidy_0]

# Determine specifications
if slurm_array_num == 0:
    np.save(f"{gv.arrays_path}specification_list_slurm_arrays.npy", specification_list_slurm_arrays)
    np.save(f"{gv.arrays_path}policy_list_slurm_arrays.npy", policy_list_slurm_arrays)
    np.save(f"{gv.arrays_path}year_list_slurm_arrays.npy", year_list_slurm_arrays)
    np.save(f"{gv.arrays_path}computation_group_list_slurm_arrays.npy", computation_group_list_slurm_arrays)
running_specification, policy_specification, year_specification, computation_group_specification = specification_list_slurm_arrays[slurm_array_num], policy_list_slurm_arrays[slurm_array_num], year_list_slurm_arrays[slurm_array_num], computation_group_list_slurm_arrays[slurm_array_num]

# Generator specifications
generator_groupings = gv.generator_groupings
generators_use = np.concatenate(tuple([value for key, value in generator_groupings.items()]))
groupings_use = np.concatenate(tuple([np.array([key]) for key, value in generator_groupings.items()]))

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
dsp_quantities = np.repeat(dsp_quantities, num_intervals_in_day)
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
if slurm_array_num == 0:
    # Fraction of production
    create_file(gv.stats_path + "frac_drop_gen_size_threshold.tex", f"{np.round((np.nansum(energy_gen[~np.isin(facilities, generators_use),:]) / np.nansum(energy_gen)) * 100.0, 2)}%")
    
    # Fraction of capacity
    ff = np.isin(energy_sources, np.concatenate((gv.natural_gas, np.array([gv.coal]))))
    create_file(gv.stats_path + "frac_drop_gen_cap_size_threshold.tex", f"{np.round((np.nansum(capacities[~np.isin(facilities, generators_use) & ff]) / np.nansum(capacities[ff])) * 100.0, 2)}%")

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
    gv.solar: 200.0, 
    gv.wind: 200.0
}
num_gens_add = {
    gv.gas_ccgt: 1, 
    gv.solar: 2, 
    gv.wind: 2
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
    
if slurm_array_num == 0:
    create_file(f"{gv.stats_path}capacity_new_ccgt.tex", f"{capacities_add[gv.gas_ccgt]:.0f}")
    create_file(f"{gv.stats_path}capacity_new_solar.tex", f"{capacities_add[gv.solar]:.0f}")
    create_file(f"{gv.stats_path}capacity_new_wind.tex", f"{capacities_add[gv.wind]:.0f}")
    create_file(f"{gv.stats_path}heat_rate_new_ccgt.tex", f"{heat_rates_add[gv.gas_ccgt]:.1f}")
    create_file(f"{gv.stats_path}co2_rate_new_ccgt.tex", f"{co2_rates_add[gv.gas_ccgt]:.0f}")

new_sources = np.array([new_ccgt, new_wind, new_solar])
for key, value in states_by_dim.items():
    if len(value) > 1:
        # Change the number of generators based on num_gens_add
        for i in range(len(value)):
            num_repeats = np.ones(value[i].shape, dtype=int)
            for source_ in new_sources:
                num_repeats[np.isin(value[i], source_)] = num_gens_add[source_]
            value[i] = np.repeat(value[i], num_repeats)

        # Replace the generators with new names
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
participants_alt = [p for p in participants] # save extra version of this array, which will use for all competitive version
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

# Rename all participants to be competitive for alternative formulation in which every generator grouping a separate firm
for i in range(facilities.shape[0]):
    generator_grouping_name = "c_"
    found_in_grouping = False
    for key, value in generator_groupings.items():
        if np.isin(facilities[i], generator_groupings[key]):
            generator_grouping_name = generator_grouping_name + key
            found_in_grouping = True
    if not found_in_grouping: # this means it's one of the new generators
        generator_grouping_name = generator_grouping_name + facilities[i] # this works b/c all the new ones are their own grouping
    participants_alt[i] = generator_grouping_name
participants_alt = np.array(participants_alt) # convert back to array

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
tex_table += f"\\begin{{tabular}}{{ llclc }} \n"
tex_table += f" & & & & \\multicolumn{{1}}{{c}}{{Total}} \\\\ \n"
tex_table += f"\\multicolumn{{1}}{{c}}{{Firm}} & \\multicolumn{{1}}{{c}}{{Technology}} & \\multicolumn{{1}}{{c}}{{State}} & \\multicolumn{{1}}{{c}}{{Generators}} & \\multicolumn{{1}}{{c}}{{Capacity (MW)}} \\\\ \n \\hline \n"

# Fill in table
firm_bf = "" # string to compare firm name to, prevents listing firm name many times
for key, value in states_by_dim.items():
    
    # If not a technology we consider, just ignore
    if len(value) == 0:
        continue
        
    # Determine firm name
    firm = key.split(",")[0]
    firm_orig = firm
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
    if firm_orig != "WPGENER":
        if np.isin(technology, np.array(["gas_add", "gas_subtract"])):
            technology = "gas"
    else:
        if technology == "gas_add":
            technology = "gas (modern)"
        if technology == "gas_subtract":
            technology = "gas (old)"
    tex_table += f"{technology} & "
        
    prev_gens = np.array([], dtype="<U1")
    for i, list_generators in enumerate(value):
        
        # Add blanks for firm / technology if not first entry
        if i > 0:
            tex_table += f" & & "

        # Add what state we're in
        tex_table += f"{i} & "
        if i > 0:
            tex_table += f"+ "
        
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

if slurm_array_num == 0:
    create_file(gv.tables_path + "choice_set.tex", tex_table)

# %%
# Save num draws stat
    
if slurm_array_num == 0:
    create_file(gv.stats_path + "wholesale_profits_num_draws.tex", f"{num_draws:,}".replace(",", "\\,"))
    create_file(gv.stats_path + "wholesale_profits_sample_length.tex", f"{sample_length:,}".replace(",", "\\,"))
    create_file(gv.stats_path + "wholesale_profits_drop_before.tex", f"{drop_before:,}".replace(",", "\\,"))
    create_file(gv.stats_path + "wholesale_profits_drop_after.tex", f"{drop_after:,}".replace(",", "\\,"))
    create_file(gv.stats_path + "wholesale_profits_effective_sample_length.tex", f"{(sample_length - drop_before - drop_after):,}".replace(",", "\\,"))

# %%
# Sample from distributions

# Determine years
years = pd.to_datetime(dates).year.values
months = pd.to_datetime(dates).month.values
days = pd.to_datetime(dates).day.values
years[months < 10] = years[months < 10] - 1 # use the same year system as WEM, begins in October
years_unique, years_unique_counts = np.unique(years, return_counts=True)
num_years = years_unique.shape[0]

# Create production cost parameters based on estimate array
ramping_costs = production_cost_est[0:production_cost_est_sources.shape[0]]
mu = production_cost_est[production_cost_est_sources.shape[0]:(2*production_cost_est_sources.shape[0])]
ramping_costs = np.concatenate((ramping_costs, np.zeros(gv.intermittent.shape[0])))
mu = np.concatenate((mu, np.zeros(gv.intermittent.shape[0])))
ramping_costs_gens = np.zeros(facilities.shape[0])
mu_gens = np.zeros(facilities.shape[0])
for s, source in enumerate(np.concatenate((production_cost_est_sources, gv.intermittent))):
    ramping_costs_gens += ramping_costs[s] * (energy_sources == source) / (capacities / 2.0) # scale by capacity
    mu_gens += mu[s] * (energy_sources == source)

# %%
# Draw from normal based on estimates of distribution
production_cost_shocks = np.tile(mu_gens[:,np.newaxis,np.newaxis], (1, sample_length, num_draws))

# %%
# Determine the demand shocks based on demand realizations and prices
xis = np.ones(total_load_all.shape) * np.nan
for y, year in enumerate(years_unique):
    select_yr = years == year
    xis[select_yr] = demand.q_demanded_inv(tariffs[select_yr], demand_elasticity, total_load_all[select_yr])
print(f"Backed out demand shocks from demand realizations.", flush=True)

# %%
# Fill in observations that don't have capacity factor observations with same day in another year

# Create list of intervals if all years had same number
max_intervals_in_year = np.max(years_unique_counts)
interval_date_strings = np.array([f"{years[i]}-{months[i]}-{days[i]}" for i in range(years.shape[0])])
interval_date_strings_full = np.array([f"{years_unique[i]}-{months[j]}-{days[j]}" for i in range(years_unique.shape[0]) for j in np.arange(years.shape[0])[years == years_unique[np.argmax(years_unique_counts)]]]) # as if every year had leap year
interval_exists = np.isin(interval_date_strings_full, interval_date_strings)

# Go through each source and take from different year if missing source in that interval
cap_factors_fill = {}
for s, source in enumerate(np.unique(energy_sources)):
    # Construct array of capacity factors for the source with year as an index
    cap_factor_source = np.ones((cap_factor[source].shape[0], interval_date_strings_full.shape[0]))
    cap_factor_source[:,interval_exists] = cap_factor[source]
    cap_factor_source = np.reshape(cap_factor_source, (cap_factor_source.shape[0], years_unique.shape[0], max_intervals_in_year)) # separate by years

    # If there is no observation in an interval, use values from closest year without missing index
    cap_factor_source_full = np.copy(cap_factor_source)
    cap_factor_source_nan = np.isnan(cap_factor_source)
    missing_idx = np.all(cap_factor_source_nan, axis=0)

    # Loop over each year and interval to replace missing values
    for j in range(cap_factor_source.shape[2]): # loop over intervals
        for i in range(cap_factor_source.shape[1]): # loop over years
            if missing_idx[i,j]:
                # Find the closest valid year (i_replace) for the same interval
                i_replace = None
                for k in range(1, cap_factor_source.shape[1]):
                    # Check previous years
                    if i - k >= 0 and not missing_idx[i - k,j]:
                        i_replace = i - k
                        break
                    # Check next years
                    if i + k < cap_factor_source.shape[1] and not missing_idx[i + k,j]:
                        i_replace = i + k
                        break
                if i_replace is not None:
                    cap_factor_source_full[:,i,j] = cap_factor_source[:,i_replace,j]
                else:
                    print(f"Unable to find a replacement for source {source} in year {i} and interval {j}")

    print(f"{source} needing replacing: {np.sum(missing_idx)}")

    # Remove "extra" leap days
    cap_factor_source = np.reshape(cap_factor_source_full, (cap_factor_source.shape[0], cap_factor_source.shape[1] * cap_factor_source.shape[2]))
    cap_factors_fill[source] = cap_factor_source[:,interval_exists]

# %%
# Construct capacity factors that fill all generators
min_year_idx_years_list = year_specification * num_years_in_year_grouping
max_year_idx_years_list = (year_specification + 1) * num_years_in_year_grouping if year_specification + 1 < num_year_groupings else np.minimum((year_specification + 2) * num_years_in_year_grouping - 1, years_list.shape[0])
select_year = np.isin(years, years_list[min_year_idx_years_list:max_year_idx_years_list])
num_intervals_year = np.sum(select_year)
cap_factors = np.ones((capacities.shape[0], num_intervals_year)) * np.nan
np.random.seed(1234567)
for s, source in enumerate(np.unique(energy_sources)):
    print(f"Computing {source}...", flush=True)
    select_source = energy_sources == source
    num_source = np.sum(select_source)
    cap_factor_source = cap_factors_fill[source][:,select_year] # select the interval 
    for i in range(num_intervals_year): # sample capacity factors from each date
        cap_factor_sample = cap_factor_source[:,i]
        cap_factor_sample_notnan = ~np.isnan(cap_factor_sample)
        num_obs_in_this_idx = np.sum(cap_factor_sample_notnan)
        if num_obs_in_this_idx > 0:
            cap_factors[select_source,i] = np.random.choice(cap_factor_sample[cap_factor_sample_notnan], size=num_source)
        else: # shouldn't happen b/c we filled in previously
            print(f"Problem with source {source} in year {years_list[year_specification]}", flush=True)
            cap_factors[select_source,i] = np.nan # will get replaced later

# %%
# Construct draws

# Standardize each variable group separately
scaler_xis = StandardScaler()
scaler_max_prices = StandardScaler()
scaler_dsp_quantities = StandardScaler()
scaler_dsp_prices = StandardScaler()
scaler_carbon_taxes = StandardScaler()
scaler_cap_factors = StandardScaler()
scaler_coal_prices = StandardScaler()
scaler_gas_prices = StandardScaler()
standardized_xis = np.reshape(scaler_xis.fit_transform(xis[select_year][:,np.newaxis]), (1, -1))
standardized_max_prices = np.reshape(scaler_max_prices.fit_transform(max_prices[select_year][:,np.newaxis]), (1, -1))
standardized_dsp_quantities = np.reshape(scaler_dsp_quantities.fit_transform(dsp_quantities[select_year][:,np.newaxis]), (1, -1))
standardized_carbon_taxes = np.reshape(scaler_carbon_taxes.fit_transform(carbon_taxes[select_year][:,np.newaxis]), (1, -1))
standardized_cap_factors = np.reshape(scaler_cap_factors.fit_transform(np.reshape(cap_factors, (-1, 1))), cap_factors.shape)
standardized_coal_prices = np.reshape(scaler_coal_prices.fit_transform(coal_prices[select_year][:,np.newaxis]), (1, -1))
standardized_gas_prices = np.reshape(scaler_gas_prices.fit_transform(gas_prices[select_year][:,np.newaxis]), (1, -1))

# Create sample with all standardized variables
samples = np.concatenate((standardized_xis, standardized_max_prices, standardized_dsp_quantities, standardized_carbon_taxes, standardized_cap_factors, standardized_coal_prices, standardized_gas_prices), axis=0)
samples = np.reshape(samples[:,:int(np.floor(samples.shape[1] // sample_length) * sample_length)], (samples.shape[0], -1, sample_length)) # drop the last observations that wouldn't fit due to need draws to be of length sample_length
samples = np.moveaxis(samples, 2, 1)
samples = np.reshape(samples, (samples.shape[0] * samples.shape[1], samples.shape[2])) # now everything within the sample_length-long interval is a different variable of the same draw
samples = samples.T # so now num_draws x (num_variables * sample_length)

# Select initial cluster, going to be the one with the largest xi
dimensions_of_interest = np.arange(sample_length) # these are the ones that capture the xis
max_idx = np.argmax(np.max(samples[:,dimensions_of_interest], axis=1)) # index with the largest xi
first_centroid = samples[max_idx,:].reshape(1, -1)

# Determine the other centroids using k-means++
random_state = check_random_state(12345)
distances = np.sum((samples - first_centroid)**2.0, axis=1)
remaining_centroids = []
for _ in range(num_draws - 1):
    # Select the next centroid with a probability proportional to the distance squared
    probs = distances / np.sum(distances)
    cumulative_probs = np.cumsum(probs)
    r = random_state.rand()
    next_centroid = samples[np.searchsorted(cumulative_probs, r)]
    remaining_centroids.append(next_centroid)
    
    # Update the distances with the new centroid
    distances = np.minimum(distances, np.sum((samples - next_centroid)**2.0, axis=1))
clusters = np.vstack([first_centroid] + remaining_centroids) # add the centroids together

# Assign observations to clusters
distances = cdist(samples, clusters, "euclidean")
labels = np.argmin(distances, axis=1)

# Rescale centroids back to original scale for each variable
clusters_xis = np.reshape(scaler_xis.inverse_transform(np.reshape(clusters[:,:sample_length], (num_draws * sample_length, 1))), (num_draws, sample_length))
clusters_max_prices = np.reshape(scaler_max_prices.inverse_transform(np.reshape(clusters[:,sample_length:2*sample_length], (num_draws * sample_length, 1))), (num_draws, sample_length))
clusters_dsp_quantities = np.reshape(scaler_dsp_quantities.inverse_transform(np.reshape(clusters[:,2*sample_length:3*sample_length], (num_draws * sample_length, 1))), (num_draws, sample_length))
clusters_carbon_taxes = np.reshape(scaler_carbon_taxes.inverse_transform(np.reshape(clusters[:,3*sample_length:4*sample_length], (num_draws * sample_length, 1))), (num_draws, sample_length))
clusters_cap_factors = np.reshape(scaler_cap_factors.inverse_transform(np.reshape(clusters[:,4*sample_length:4*sample_length+cap_factors.shape[0]*sample_length], (num_draws * sample_length * cap_factors.shape[0], 1))), (num_draws, cap_factors.shape[0] * sample_length))
clusters_coal_prices = np.reshape(scaler_coal_prices.inverse_transform(np.reshape(clusters[:,4*sample_length+cap_factors.shape[0]*sample_length:5*sample_length+cap_factors.shape[0]*sample_length], (num_draws * sample_length, 1))), (num_draws, sample_length))
clusters_gas_prices = np.reshape(scaler_gas_prices.inverse_transform(np.reshape(clusters[:,5*sample_length+cap_factors.shape[0]*sample_length:], (num_draws * sample_length, 1))), (num_draws, sample_length))
clusters = np.concatenate((clusters_xis, clusters_max_prices, clusters_dsp_quantities, clusters_carbon_taxes, clusters_cap_factors, clusters_coal_prices, clusters_gas_prices), axis=1)

# Compute cluster sizes and probabilities
cluster_sizes = np.bincount(labels)
cluster_probabilities = cluster_sizes / samples.shape[0]

# %%
# Construct variables for computing equilibrium

dsp_quantities_sample = clusters_dsp_quantities.T
available_capacities_sample = np.reshape(clusters_cap_factors, (clusters_cap_factors.shape[0], cap_factors.shape[0], sample_length)) * capacities[np.newaxis,:,np.newaxis] / 2.0 # / 2.0 to put in half-hour intervals
available_capacities_sample = np.moveaxis(np.moveaxis(available_capacities_sample, 1, 0), 2, 1)
production_costs_sample = np.nan_to_num(heat_rates)[np.newaxis,:,np.newaxis] * ((energy_sources == gv.coal)[np.newaxis,:,np.newaxis] * clusters_coal_prices[:,np.newaxis,:] + np.isin(energy_sources, gv.natural_gas)[np.newaxis,:,np.newaxis] * clusters_gas_prices[:,np.newaxis,:]) + np.moveaxis(production_cost_shocks, 2, 0)
production_costs_sample = np.moveaxis(np.moveaxis(production_costs_sample, 1, 0), 2, 1)
production_costs_w_carbon_tax_sample = production_costs_sample + co2_rates[:,np.newaxis,np.newaxis] * clusters_carbon_taxes.T[np.newaxis,:,:]
xis_sample = clusters_xis.T
max_prices_sample = clusters_max_prices.T

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
tex_table += f" & & & \\multicolumn{{1}}{{c}}{{Capacity}} & \\multicolumn{{1}}{{c}}{{Heat Rate}} & \\multicolumn{{1}}{{c}}{{Emissions Rate}} & \\multicolumn{{1}}{{c}}{{Entered}} & \\multicolumn{{1}}{{c}}{{Exit}} \\\\ \n"
tex_table += f"\\multicolumn{{1}}{{c}}{{Generator}} & \\multicolumn{{1}}{{c}}{{Firm}} & \\multicolumn{{1}}{{c}}{{Technology}} & \\multicolumn{{1}}{{c}}{{(MW)}} & \\multicolumn{{1}}{{c}}{{(GJ/MWh)}} & \\multicolumn{{1}}{{c}}{{(kg$\\text{{CO}}_{{2}}$-eq/MWh)}} & \\multicolumn{{1}}{{c}}{{Year}} & \\multicolumn{{1}}{{c}}{{Year}} \\\\ \n"
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
    tex_table += f"{capacities[i]:,.0f}".replace(",", "\\,")
    if "KEMERTON" in facilities[i]:
        tex_table += f"${{}}^{{\\P}}$"
    tex_table += " & "
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

if slurm_array_num == 0:
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
participants_alt_unique, participants_alt_int = np.unique(participants_alt, return_inverse=True)
energy_sources_unique, energy_sources_int = np.unique(energy_sources, return_inverse=True)
facilities_int_unique = np.unique(facilities_int)
participants_int_unique = np.unique(participants_int)
participants_alt_int_unique = np.unique(participants_alt_int)
energy_sources_int_unique = np.unique(energy_sources_int)

# Create sources with limited starts
limited_start_energy_sources = energy_sources_int_unique[np.isin(energy_sources_unique, np.array([gv.coal, gv.gas_ccgt]))]

# Create sources that are intermittent, 0 MC
intermittent_0mc_sources = energy_sources_int_unique[np.isin(energy_sources_unique, gv.intermittent)]

# Initialize arrays
profits = np.zeros((array_state_in.shape[1], participants_int_unique.shape[0]))
emissions = np.zeros((array_state_in.shape[1],))
blackouts = np.zeros((array_state_in.shape[1],))
frac_by_source = np.zeros((array_state_in.shape[1], energy_sources_int_unique.shape[0]))
quantity_weighted_avg_price = np.zeros((array_state_in.shape[1],))
total_produced = np.zeros((array_state_in.shape[1],))
misallocated_demand = np.zeros((array_state_in.shape[1],))
consumer_surplus = np.zeros((array_state_in.shape[1],))
renewable_production = np.zeros((array_state_in.shape[1],))
total_production_cost = np.zeros((array_state_in.shape[1],))

# Determine number of half hours that are in a year
num_half_hours = gv.num_intervals_in_day * 365.0

# Determine which intervals to use in constructing averages
keep_t_sample = np.ones((sample_length,), dtype=bool)
keep_t_sample[:drop_before] = False
keep_t_sample[-drop_after:] = False

# Create an initial candidate price
candidate_avg_price = np.average(prices, weights=total_load_all)

# Initialize ramping costs
ramping_costs = ramping_costs_gens

# Create arrays of "generators" with demand response
blackout_generator_capacity = 9999.9 * np.ones((1, available_capacities_sample.shape[1], available_capacities_sample.shape[2])) # unlimited capacity blackout "generator"
blackout_generator_price = max_prices_sample[np.newaxis,:,:]
dsp_prices_sample = 0.99 * blackout_generator_price # use price cap, just slightly below so JuMP optimizer uses this price rather than blackout price, don't use 0.9999999 b/c JuMP is doing an approximation, and we'll get blackouts rather than DSP
available_capacities_use = np.concatenate((available_capacities_sample, dsp_quantities_sample[np.newaxis,:,:], blackout_generator_capacity), axis=0)
production_costs_use = np.concatenate((production_costs_w_carbon_tax_sample, dsp_prices_sample, blackout_generator_price), axis=0)
production_costs_wo_tax_use = np.concatenate((production_costs_sample, np.zeros((2, dsp_quantities_sample.shape[0], dsp_quantities_sample.shape[1]))), axis=0)
participants_use = np.concatenate((participants_int, np.array([-1, -2]))) # -1 = DSP, -2 = blackout
participants_alt_use = np.concatenate((participants_alt_int, np.array([-1, -2]))) # -1 = DSP, -2 = blackout
energy_sources_use = np.concatenate((energy_sources_int, np.array([-1, -2]))) # -1 = DSP, -2 = blackout
co2_rates_use = np.concatenate((co2_rates, np.zeros((2,))))
ramping_costs_use = np.concatenate((ramping_costs, np.zeros((2,))))
initial_quantities_use = np.concatenate((np.zeros((available_capacities_sample.shape[0], available_capacities_sample.shape[2])), np.zeros((2, dsp_quantities_sample.shape[1]))), axis=0)
generators_w_ramping_costs = ~np.isclose(ramping_costs_use, 0.0)

# Create indices used in this SLURM job
indices_array = np.arange(array_state_in.shape[1])
num_indices_in_computation_grouping = int(np.floor(indices_array.shape[0] / num_computation_groups_per_specification))
min_year_idx_indices_array = computation_group_specification * num_indices_in_computation_grouping
max_year_idx_indices_array = (computation_group_specification + 1) * num_indices_in_computation_grouping if computation_group_specification + 1 < num_indices_in_computation_grouping else np.minimum((computation_group_specification + 2) * num_indices_in_computation_grouping - 1, indices_array.shape[0])
indices_array = indices_array[min_year_idx_indices_array:max_year_idx_indices_array]

# Initialize model
start = time.time()
model, demand_lb, quantity, quantity_diff, market_clearing, quantity_lessthan_capacity = eqm.initialize_model(production_costs_use, ramping_costs_use, initial_quantities_use, generators_w_ramping_costs, print_msg=False)
print(f"\tmodel initialization in {np.round(time.time() - start, 1)} seconds", flush=True)

def compute_eqm(i, model, demand_lb, quantity, market_clearing, index_in_loop_val=None):
    if index_in_loop_val is not None:
        if index_in_loop_val % 100 == 0:
            print(f"Computing index {i} ({index_in_loop_val + 1} / {indices_array.shape[0]}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Provide available capacities of generators in market
    select_gens = np.concatenate((array_state_in[:,i], np.ones((2,), dtype=bool))) # select the ith component, then add on the DSPs and price cap
    true_generators = np.concatenate((np.ones(np.sum(array_state_in[:,i]), dtype=bool), np.zeros((2,), dtype=bool))) # + 1 for blackout "generator"
    model = eqm.update_available_capacities(model, quantity_lessthan_capacity, available_capacities_use, select_gens)
    
    return eqm.expected_profits(model, 
                                demand_lb, 
                                quantity, 
                                market_clearing, 
                                select_gens, 
                                available_capacities_use[select_gens,:,:], 
                                production_costs_use[select_gens,:,:], 
                                production_costs_wo_tax_use[select_gens,:,:], 
                                participants_use[select_gens], 
                                participants_int_unique, 
                                energy_sources_use[select_gens], 
                                energy_sources_int_unique, 
                                co2_rates_use[select_gens], 
                                true_generators, 
                                ramping_costs_use[select_gens], 
                                initial_quantities_use[select_gens,:], 
                                fixed_tariff_component, 
                                demand_elasticity, 
                                xis_sample, 
                                candidate_avg_price, 
                                num_half_hours, 
                                sample_weights=cluster_probabilities, 
                                systemic_blackout_threshold=0.1, 
                                keep_t_sample=None, 
                                threshold_eps=0.01, 
                                max_iter=6, 
                                intermittent_0mc_sources=intermittent_0mc_sources, 
                                print_msg=False)

# Solve for equilibria
if running_specification == 0:
    for index_in_loop_val, idx in enumerate(indices_array):
        res = compute_eqm(idx, model, demand_lb, quantity, market_clearing, index_in_loop_val=index_in_loop_val)
        profits[idx,:] = res[0]
        emissions[idx] = res[1]
        blackouts[idx] = res[2]
        frac_by_source[idx,:] = res[3]
        quantity_weighted_avg_price[idx] = res[4]
        total_produced[idx] = res[5]
        misallocated_demand[idx] = res[6]
        consumer_surplus[idx] = res[7]

    print(f"Completed solving for equilibria in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

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

capacity_payments = np.zeros((array_state_in.shape[1], cap_years.shape[0], participants_int_unique.shape[0]))

# Commitments of each generator
expected_payments_permw_perdollar = 1.0 * ~np.isin(energy_sources_int, np.arange(energy_sources_unique.shape[0])[np.isin(energy_sources_unique, gv.intermittent)])[:,np.newaxis] # np.mean(cap_factors_extend, axis=2)

def compute_payments(i, index_in_loop_val=None):
    if index_in_loop_val is not None:
        if index_in_loop_val % 100 == 0:
            print(f"Computing index {i} ({index_in_loop_val + 1} / {indices_array.shape[0]}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    select_gens = array_state_in[:,i]
    return cc.expected_cap_payment(expected_payments_permw_perdollar[select_gens,:], capacities[select_gens], capacity_price, participants_int[select_gens], participants_int_unique)
        
# Solve for capacity payments
if (running_specification == 0) and (year_specification == 0): # year specification == 0 b/c done all at once
    for index_in_loop_val, idx in enumerate(indices_array):
        capacity_payments[idx,:,:] = compute_payments(idx, index_in_loop_val=index_in_loop_val)

    print(f"Completed solving for capacity payments in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Save arrays
if running_specification == 0:
    np.savez_compressed(f"{gv.arrays_path}data_env_{year_specification}_{computation_group_specification}.npz", 
                        profits=profits[indices_array,:], 
                        emissions=emissions[indices_array], 
                        blackouts=blackouts[indices_array], 
                        frac_by_source=frac_by_source[indices_array,:], 
                        quantity_weighted_avg_price=quantity_weighted_avg_price[indices_array], 
                        total_produced=total_produced[indices_array], 
                        misallocated_demand=misallocated_demand[indices_array], 
                        consumer_surplus=consumer_surplus[indices_array])
if (running_specification == 0) and (year_specification == 0):
    np.savez_compressed(f"{gv.arrays_path}data_env_cap_payments_{computation_group_specification}.npz", 
                        capacity_payments=capacity_payments[indices_array,:,:])
if slurm_array_num == 0:
    np.savez_compressed(f"{gv.arrays_path}state_space.npz", 
                        facilities_unique=facilities_unique, 
                        facilities_int=facilities_int, 
                        facilities_int_unique=facilities_int_unique, 
                        participants_unique=participants_unique, 
                        participants_alt_unique=participants_alt_unique, 
                        participants_int=participants_int, 
                        participants_alt_int=participants_alt_int, 
                        participants_int_unique=participants_int_unique, 
                        participants_alt_int_unique=participants_alt_int_unique, 
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
if slurm_array_num == 0:
    create_file(gv.stats_path + "low_price_cap_counterfactuals.tex", f"{int(low_price_cap_counterfactuals):,}".replace(",", "\\,"))
    create_file(gv.stats_path + "high_price_cap_counterfactuals.tex", f"{int(high_price_cap_counterfactuals):,}".replace(",", "\\,"))
production_costs_use = np.concatenate((production_costs_sample, dsp_prices_sample, blackout_generator_price), axis=0)
production_costs_use[-1,:,:] = low_price_cap_counterfactuals
if running_specification == 3:
    production_costs_use[-1,:,:] = high_price_cap_counterfactuals
production_costs_use[participants_use == -1,:,:] = 0.99 * production_costs_use[-1,:,:][np.newaxis,:,:] # use price cap, just slightly below so JuMP optimizer uses this price rather than blackout price, don't use 0.9999999 b/c JuMP is doing an approximation, and we'll get blackouts rather than DSP

# %%
# Carbon tax counterfactual
if (running_specification == 1) or (running_specification == 3) or (running_specification == 4):
    print(f"Starting solving for carbon tax counterfactuals...", flush=True)
start_task = time.time()

# Initialize arrays
profits = np.zeros((array_state_in.shape[1], participants_int_unique.shape[0]))
profits_alt = np.zeros((array_state_in.shape[1], participants_alt_int_unique.shape[0]))
emissions = np.zeros((array_state_in.shape[1],))
blackouts = np.zeros((array_state_in.shape[1],))
frac_by_source = np.zeros((array_state_in.shape[1], energy_sources_int_unique.shape[0]))
quantity_weighted_avg_price = np.zeros((array_state_in.shape[1],))
total_produced = np.zeros((array_state_in.shape[1],))
misallocated_demand = np.zeros((array_state_in.shape[1],))
consumer_surplus = np.zeros((array_state_in.shape[1],))
total_production_cost = np.zeros((array_state_in.shape[1],))
dsp_profits = np.zeros((array_state_in.shape[1],))
battery_profits = np.zeros((array_state_in.shape[1],))
battery_discharge = np.zeros((array_state_in.shape[1],))

# Determine whether need to sum profits in way in which all generators groupings are separate
return_alternative_participant_specification = False
if running_specification == 1: # just for specification == 1
    return_alternative_participant_specification = True

# Determine production costs including carbon tax
if (running_specification == 1) or (running_specification == 3) or (running_specification == 4):
    production_costs_w_carbon_tax = production_costs_use + (co2_rates_use[:,np.newaxis,np.newaxis] * carbon_taxes_linspace[policy_specification] / 1000.0)
    start = time.time()
    if running_specification == 4:
        battery_dict = {
            'flow': battery_flow, 
            'capacity': battery_capacity, 
            'delta': battery_delta, 
            'initial_stock': np.ones(num_draws) * 0.5 * battery_capacity
        }
        model, demand_lb, quantity, quantity_diff, quantity_charge, quantity_discharge, market_clearing, quantity_lessthan_capacity, battery_constraints = eqm.initialize_model(production_costs_w_carbon_tax, ramping_costs_use, initial_quantities_use, generators_w_ramping_costs, battery_dict=battery_dict, print_msg=False)
        battery_dict['quantity_charge'] = quantity_charge
        battery_dict['quantity_discharge'] = quantity_discharge
    else:
        battery_dict = None
        model, demand_lb, quantity, quantity_diff, market_clearing, quantity_lessthan_capacity = eqm.initialize_model(production_costs_w_carbon_tax, ramping_costs_use, initial_quantities_use, generators_w_ramping_costs, print_msg=False)
    print(f"\tmodel initialization in {np.round(time.time() - start, 1)} seconds", flush=True)

def compute_eqm(i, model, demand_lb, quantity, market_clearing, battery_dict, index_in_loop_val=None):
    if index_in_loop_val is not None:
        if index_in_loop_val % 100 == 0:
            print(f"Computing index {i} ({index_in_loop_val + 1} / {indices_array.shape[0]}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Provide available capacities of generators in market
    select_gens = np.concatenate((array_state_in[:,i], np.ones((2,), dtype=bool))) # select the ith component, then add on the DSPs and price cap
    true_generators = np.concatenate((np.ones(np.sum(array_state_in[:,i]), dtype=bool), np.zeros((2,), dtype=bool))) # + 1 for blackout "generator"
    start = time.time()
    model = eqm.update_available_capacities(model, quantity_lessthan_capacity, available_capacities_use, select_gens)
    
    return eqm.expected_profits(model, 
                                demand_lb, 
                                quantity, 
                                market_clearing, 
                                select_gens, 
                                available_capacities_use[select_gens,:,:], 
                                production_costs_w_carbon_tax[select_gens,:,:], 
                                production_costs_wo_tax_use[select_gens,:,:], 
                                participants_use[select_gens], 
                                participants_int_unique, 
                                energy_sources_use[select_gens], 
                                energy_sources_int_unique, 
                                co2_rates_use[select_gens], 
                                true_generators, 
                                ramping_costs_use[select_gens], 
                                initial_quantities_use[select_gens,:], 
                                fixed_tariff_component, 
                                demand_elasticity, 
                                xis_sample, 
                                candidate_avg_price, 
                                num_half_hours, 
                                battery_dict=battery_dict, 
                                alternative_participants_dict={'participants_unique': participants_alt_int_unique, 'participants': participants_alt_use[select_gens]} if return_alternative_participant_specification else None, 
                                sample_weights=cluster_probabilities, 
                                systemic_blackout_threshold=0.1, 
                                keep_t_sample=None, 
                                threshold_eps=0.01, 
                                max_iter=6, 
                                intermittent_0mc_sources=intermittent_0mc_sources, 
                                print_msg=False)

# Run carbon tax counterfactuals
if (running_specification == 1) or (running_specification == 3) or (running_specification == 4):
    for index_in_loop_val, idx in enumerate(indices_array):
        res = compute_eqm(idx, model, demand_lb, quantity, market_clearing, battery_dict=battery_dict, index_in_loop_val=index_in_loop_val)
        if return_alternative_participant_specification:
            profits[idx,:] = res[0][0]
            profits_alt[idx,:] = res[0][1]
        else:
            profits[idx,:] = res[0]
        emissions[idx] = res[1]
        blackouts[idx] = res[2]
        frac_by_source[idx,:] = res[3]
        quantity_weighted_avg_price[idx] = res[4]
        total_produced[idx] = res[5]
        misallocated_demand[idx] = res[6]
        consumer_surplus[idx] = res[7]
        total_production_cost[idx] = res[9]
        dsp_profits[idx] = res[10]
        if running_specification == 4:
            battery_profits[idx] = res[11]
            battery_discharge[idx] = res[12]

    print(f"Completed the carbon tax counterfactual profits in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Save arrays
save_args = {
    'profits': profits[indices_array,:],
    'emissions': emissions[indices_array],
    'blackouts': blackouts[indices_array],
    'frac_by_source': frac_by_source[indices_array,:],
    'quantity_weighted_avg_price': quantity_weighted_avg_price[indices_array],
    'total_produced': total_produced[indices_array],
    'misallocated_demand': misallocated_demand[indices_array],
    'consumer_surplus': consumer_surplus[indices_array],
    'total_production_cost': total_production_cost[indices_array], 
    'dsp_profits': dsp_profits[indices_array]
}
if return_alternative_participant_specification:
    save_args['profits_alt'] = profits_alt[indices_array,:]
if running_specification == 1:
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_co2tax_{policy_specification}_{year_specification}_{computation_group_specification}.npz", **save_args)
if running_specification == 3:
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_co2tax_highpricecap_{policy_specification}_{year_specification}_{computation_group_specification}.npz", **save_args)
if running_specification == 4:
    save_args['battery_profits'] = battery_profits[indices_array]
    save_args['battery_discharge'] = battery_discharge[indices_array]
    np.savez_compressed(f"{gv.arrays_path}counterfactual_battery_{policy_specification}_{year_specification}_{computation_group_specification}.npz", **save_args)

# %%
# Capacity payment counterfactual
if running_specification == 1:
    print(f"Starting solving for capacity payment counterfactuals...", flush=True)
start_task = time.time()

capacity_payments = np.zeros((capacity_payments_linspace.shape[0],  array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
capacity_payments_alt = np.zeros((capacity_payments_linspace.shape[0],  array_state_in.shape[1], years_unique.shape[0], participants_alt_int_unique.shape[0]))

# Commitments of each generator
expected_payments_permw_perdollar = 1.0 * ~np.isin(energy_sources_int, np.arange(energy_sources_unique.shape[0])[np.isin(energy_sources_unique, gv.intermittent)])[:,np.newaxis]

def compute_payments(c, i, participants_int_unique_use, index_in_loop_val=None):
    if index_in_loop_val is not None:
        if index_in_loop_val % 100 == 0:
            print(f"Computing index ({c}, {i}) ({index_in_loop_val + 1} / {capacity_payments_linspace.shape[0] * array_state_in.shape[1]}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    select_gens = array_state_in[:,i]
    capacity_price_years = np.ones(years_unique.shape[0]) * capacity_payments_linspace[c]
    return cc.expected_cap_payment(expected_payments_permw_perdollar[select_gens,:], capacities[select_gens], capacity_price_years, participants_int[select_gens], participants_int_unique_use)
        
# Initialize multiprocessing
if (running_specification == 1) and (policy_specification == 0) and (year_specification == 0):
    for index_in_loop_val, indices in enumerate(product(range(capacity_payments_linspace.shape[0]), indices_array)):
        c, i = indices[0], indices[1]
        capacity_payments[c,i,:,:] = compute_payments(c, i, participants_int_unique, index_in_loop_val=index_in_loop_val)
        if return_alternative_participant_specification:
            capacity_payments_alt[c,i,:,:] = compute_payments(c, i, participants_alt_int_unique, index_in_loop_val=index_in_loop_val)

    print(f"Completed the capacity payment counterfactual payments in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Save arrays
save_args = {
    'capacity_payments': capacity_payments[:,indices_array,:,:]
}
if return_alternative_participant_specification:
    save_args['capacity_payments_alt'] = capacity_payments_alt[:,indices_array,:,:]
if (running_specification == 1) and (policy_specification == 0) and (year_specification == 0):
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz", **save_args)

# %%
# Expanded capacity payment counterfactual
if running_specification == 5:
    print(f"Starting solving for expanded capacity payment counterfactuals...", flush=True)
    start_task = time.time()
    
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    
    # Commitments of each generator
    expected_payments_permw_perdollar_coal = 1.0 * np.isin(energy_sources_int, np.arange(energy_sources_unique.shape[0])[np.isin(energy_sources_unique, gv.coal)])
    expected_payments_permw_perdollar_gas = 1.0 * np.isin(energy_sources_int, np.arange(energy_sources_unique.shape[0])[np.isin(energy_sources_unique, np.array([gv.gas_ocgt, gv.gas_ccgt, gv.gas_cogen]))])
    
    def compute_payments(c, c_prime, i, participants_int_unique_use, index_in_loop_val=None):
        if index_in_loop_val is not None:
            if index_in_loop_val % 1000 == 0:
                print(f"Computing index ({c}, {c_prime}, {i}) ({index_in_loop_val + 1} / {capacity_payments_linspace.shape[0] * capacity_payments_linspace_extended.shape[0] * array_state_in.shape[1]}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        select_gens = array_state_in[:,i]
        capacity_price_years_coal = np.ones(years_unique.shape[0]) * capacity_payments_linspace[c]
        capacity_price_years_gas = np.ones(years_unique.shape[0]) * capacity_payments_linspace_extended[c_prime]
        capacity_price_years = expected_payments_permw_perdollar_coal[:,np.newaxis] * capacity_price_years_coal[np.newaxis,:] + expected_payments_permw_perdollar_gas[:,np.newaxis] * capacity_price_years_gas[np.newaxis,:]
        expected_payments_permw_perdollar = expected_payments_permw_perdollar_coal + expected_payments_permw_perdollar_gas
        return cc.expected_cap_payment(expected_payments_permw_perdollar[select_gens,np.newaxis], capacities[select_gens], capacity_price_years[select_gens,:], participants_int[select_gens], participants_int_unique_use)
        
    # Initialize multiprocessing
    for index_in_loop_val, indices in enumerate(product(range(capacity_payments_linspace.shape[0]), range(capacity_payments_linspace_extended.shape[0]), indices_array)):
        c, c_prime, i = indices[0], indices[1], indices[2]
        capacity_payments[c,c_prime,i,:,:] = compute_payments(c, c_prime, i, participants_int_unique, index_in_loop_val=index_in_loop_val)

    print(f"Completed the capacity payment counterfactual payments in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

    save_args = {
        'capacity_payments': capacity_payments[:,:,indices_array,:,:]
    }
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_capacitypaymentexpanded_{computation_group_specification}.npz", **save_args)

# %%
# Renewable subsidy counterfactual

if running_specification == 2:
    print(f"Starting solving for renewable subsidy counterfactuals...", flush=True)
start_task = time.time()

# Initialize arrays
profits = np.zeros((array_state_in.shape[1], participants_int_unique.shape[0]))
emissions = np.zeros((array_state_in.shape[1],))
blackouts = np.zeros((array_state_in.shape[1],))
frac_by_source = np.zeros((array_state_in.shape[1], energy_sources_int_unique.shape[0]))
quantity_weighted_avg_price = np.zeros((array_state_in.shape[1],))
total_produced = np.zeros((array_state_in.shape[1],))
misallocated_demand = np.zeros((array_state_in.shape[1],))
consumer_surplus = np.zeros((array_state_in.shape[1],))
renewable_production = np.zeros((array_state_in.shape[1],))
total_production_cost = np.zeros((array_state_in.shape[1],))
dsp_profits = np.zeros((array_state_in.shape[1],))

# Determine production costs including carbon tax
if running_specification == 2:
    renewable_gens = np.isin(energy_sources_use, np.arange(energy_sources_unique.shape[0])[np.isin(energy_sources_unique, gv.intermittent)])
    production_costs_w_renewable_subsidies = production_costs_use - renewable_subsidies_linspace[policy_specification] * renewable_gens[:,np.newaxis,np.newaxis]
    start = time.time()
    model, demand_lb, quantity, quantity_diff, market_clearing, quantity_lessthan_capacity = eqm.initialize_model(production_costs_w_renewable_subsidies, ramping_costs_use, initial_quantities_use, generators_w_ramping_costs, print_msg=False)
    print(f"\tmodel initialization in {np.round(time.time() - start, 1)} seconds", flush=True)

def compute_eqm(i, model, demand_lb, quantity, market_clearing, index_in_loop_val=None):
    if index_in_loop_val is not None:
        if index_in_loop_val % 100 == 0:
            print(f"Computing index {i} ({index_in_loop_val + 1} / {indices_array.shape[0]}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Provide available capacities of generators in market
    select_gens = np.concatenate((array_state_in[:,i], np.ones((2,), dtype=bool))) # select the ith component, then add on the DSPs and price cap
    true_generators = np.concatenate((np.ones(np.sum(array_state_in[:,i]), dtype=bool), np.zeros((2,), dtype=bool))) # + 1 for blackout "generator"
    model = eqm.update_available_capacities(model, quantity_lessthan_capacity, available_capacities_use, select_gens)
    
    return eqm.expected_profits(model, 
                                demand_lb, 
                                quantity, 
                                market_clearing, 
                                select_gens, 
                                available_capacities_use[select_gens,:,:], 
                                production_costs_w_renewable_subsidies[select_gens,:,:], 
                                production_costs_wo_tax_use[select_gens,:,:], 
                                participants_use[select_gens], 
                                participants_int_unique, 
                                energy_sources_use[select_gens], 
                                energy_sources_int_unique, 
                                co2_rates_use[select_gens], 
                                true_generators, 
                                ramping_costs_use[select_gens], 
                                initial_quantities_use[select_gens,:], 
                                fixed_tariff_component, 
                                demand_elasticity, 
                                xis_sample, 
                                candidate_avg_price, 
                                num_half_hours, 
                                sample_weights=cluster_probabilities, 
                                systemic_blackout_threshold=0.1, 
                                keep_t_sample=None, 
                                threshold_eps=0.01, 
                                max_iter=6, 
                                intermittent_0mc_sources=intermittent_0mc_sources, 
                                print_msg=False)

# Run carbon tax counterfactuals
if running_specification == 2:
    for index_in_loop_val, idx in enumerate(indices_array):
        res = compute_eqm(idx, model, demand_lb, quantity, market_clearing, index_in_loop_val=index_in_loop_val)
        profits[idx,:] = res[0]
        emissions[idx] = res[1]
        blackouts[idx] = res[2]
        frac_by_source[idx,:] = res[3]
        quantity_weighted_avg_price[idx] = res[4]
        total_produced[idx] = res[5]
        misallocated_demand[idx] = res[6]
        consumer_surplus[idx] = res[7]
        renewable_production[idx] = res[8]
        total_production_cost[idx] = res[9]
        dsp_profits[idx] = res[10]

    print(f"Completed the renewable subsidies counterfactual profits in {np.round(time.time() - start_task, 1)} seconds.", flush=True)

# %%
# Save arrays
save_args = {
    'profits': profits[indices_array,:],
    'emissions': emissions[indices_array],
    'blackouts': blackouts[indices_array],
    'frac_by_source': frac_by_source[indices_array,:],
    'quantity_weighted_avg_price': quantity_weighted_avg_price[indices_array],
    'total_produced': total_produced[indices_array],
    'misallocated_demand': misallocated_demand[indices_array],
    'consumer_surplus': consumer_surplus[indices_array],
    'renewable_production': renewable_production[indices_array],
    'total_production_cost': total_production_cost[indices_array], 
    'dsp_profits': dsp_profits[indices_array]
}
if running_specification == 2:
    np.savez_compressed(f"{gv.arrays_path}counterfactual_env_renewablesubisidies_{policy_specification}_{year_specification}_{computation_group_specification}.npz", **save_args)


print(f"Everything in this SLURM job is finished!", flush=True)
