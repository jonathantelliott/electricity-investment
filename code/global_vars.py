import numpy as np

# Data location
scratch_loc = "/scratch/jte254/"
home_loc = "/home/jte254/"
scratch_electricity_loc = "/scratch/jte254/electricity/"
home_electricity_loc = "/home/jte254/electricity/"
data_loc = scratch_loc + "electricity/data/"
data_path = data_loc + "raw_data/wem_data/" #"/Volumes/My Passport/wem_data/"
commodity_data_path = data_loc + "raw_data/wa_dmirs/"
western_power_data_path = data_loc + "raw_data/western_power_data/"
temp_data_path = data_loc + "raw_data/bom_data/"
arrays_path = data_loc + "processed_data_v3/"
results_loc = home_loc + "electricity/results/"
tables_path = results_loc + "tables/"
graphs_path = results_loc + "graphs/"
stats_path = results_loc + "stats/"


# Date variables
start_year = 2006
start_month = 10
start_day = 1
end_year = 2022
end_month = 9
end_day = 30
num_intervals_in_day = 48
refyear = 2015 # reference year for inflation (everything is in terms of AUD in this year)


# Technology variables
wind = "wind farm"
solar = "solar pv"
gas_ocgt = "natural gas (open cycle)"
gas_ccgt = "natural gas (combined cycle)"
gas_cogen = "natural gas (cogeneration)"
coal = "coal"
landfill = "landfill gas"
distillate = "distillate"
use_sources = np.array([solar, wind, gas_ocgt, gas_ccgt, gas_cogen, coal]) # np.array([wind, gas_ocgt, gas_ccgt, gas_cogen, coal]) # sources used in analysis
zero_emissions_tech = np.array([wind, solar])
intermittent = np.array([wind, solar])
natural_gas = np.array([gas_ocgt, gas_ccgt])#np.array([gas_ocgt, gas_ccgt, gas_cogen])


# Competition variables
num_strategic_firms = 3
competitive_name = "c"
# refund_multiplier_intermittent = 
# refund_multiplier_scheduled = 


# Outage types
consequential_outage = "Consequential"
forced_outage = "Forced"
planned_outage = "Planned"


# File names
# Facilities-level
dates_file = arrays_path + "dates.npy"
facilities_file = arrays_path + "facilities.npy"
participants_file = arrays_path + "participants.npy"
capacities_file = arrays_path + "capacities.npy"
energy_sources_file = arrays_path + "energy_sources.npy"
exited_file = arrays_path + "exited.npy"
energy_types_file = arrays_path + "energy_types.npy"
intermittent_file = arrays_path + "intermittent.npy"
heat_rates = arrays_path + "heat_rates.npy"
co2_rates = arrays_path + "co2_rates.npy"
transport_charges = arrays_path + "transport_charges.npy"
info_sources = arrays_path + "info_sources.npy"
dsp_names_file = arrays_path + "dsp_names.npy"
dsp_quantities_file = arrays_path + "dsp_quantities.npy"
dsp_prices_file = arrays_path + "dsp_prices.npy"
dsp_dispatch_file = arrays_path + "dsp_dispatch.npy"
energy_gen_file = arrays_path + "energy_gen.npz"
load_curtailment_file = arrays_path + "load_curtailment.npy"
prices_file = arrays_path + "prices.npy"
prices_realtime_file = arrays_path + "prices_realtime.npy"
load_realtime_file = arrays_path + "load_realtime.npy"
carbon_taxes_file = arrays_path + "carbon_taxes.npy"
balancing_avail_file = arrays_path + "balancing_avail.npy"
balancing_arrays_file = arrays_path + "balancing_arrays.npz"
exited_date_file = arrays_path + "exited_date.npy"
entered_date_file = arrays_path + "entered_date.npy"
capacity_price_file = arrays_path + "capacity_price.npy"
cap_date_from_file = arrays_path + "cap_date_from.npy"
cap_date_until_file = arrays_path + "cap_date_until.npy"
capacity_commitments_file = arrays_path + "capacity_commitments.npy"
max_prices_file = arrays_path + "max_prices.npy"
min_prices_file = arrays_path + "min_prices.npy"
outages_file = arrays_path + "outages.npz"
coal_prices_file = arrays_path + "coal_prices.npy"
gas_prices_file = arrays_path + "gas_prices.npy"
residential_tariff_file = arrays_path + "residential_tariff.npy"

# Participant-level
firms_file = arrays_path + "firms.npy"
firm_energy_gen_file = arrays_path + "firm_energy_gen.npy"
firm_capacity_file = arrays_path + "firm_capacity.npy"
firm_effective_capacity_file = arrays_path + "firm_effective_capacity.npy"
firm_outages_exante_matrix_file = arrays_path + "firm_outages_exante_matrix.npy"
firm_outages_expost_matrix_file = arrays_path + "firm_outages_expost_matrix.npy"
largest_by_energy_gen_file = arrays_path + "largest_by_energy_gen.npy"
largest_by_cap_file = arrays_path + "largest_by_cap.npy"

# Interval-level variables
air_temps_file = arrays_path + "air_temps.npy"
wet_bulb_temps_file = arrays_path + "wet_bulb_temps.npy"

# Model-definition aggregated level
strategic_firms_file = arrays_path + "strategic_firms.npy"
firms_processed_file = arrays_path + "firms_processed.npy"
participants_rename_file = arrays_path + "participants_rename.npy"
energy_sources_rename_file = arrays_path + "energy_sources_rename.npy"
energy_types_use_processed_file = arrays_path + "energy_types_use_processed.npy"
dates_processed_file = arrays_path + "dates_processed.npy"
energy_gen_processed_file = arrays_path + "energy_gen_processed.npy"
capacities_processed_file = arrays_path + "capacities_processed.npy"
outages_processed_file = arrays_path + "outages_processed.npy"
all_outages_processed_file = arrays_path + "all_outages_processed.npy"
prices_processed_file = arrays_path + "prices_processed.npy"
dates_from_processed_file = arrays_path + "dates_from_processed.npy"
dates_until_processed_file = arrays_path + "dates_until_processed.npy"
cap_prices_processed_file = arrays_path + "cap_prices_processed.npy"
capacities_yearly_processed_file = arrays_path + "capacities_yearly_processed.npy"
commitment_processed_file = arrays_path + "commitment_processed.npy"
capacity_costs_file = arrays_path + "capacity_costs.npy"
capacity_costs_sources_file = arrays_path + "capacity_costs_sources.npy"
capacity_costs_years_file = arrays_path + "capacity_costs_years_file.npy"
emissions_file = arrays_path + "emissions.npy"

# Generator groupings
generator_groupings = {
    'COLLIE_G1': np.array(["COLLIE_G1"]), 
    'MUJA_G7-8': np.array(["MUJA_G7", "MUJA_G8"]), 
    'MUJA_G5-6': np.array(["MUJA_G5", "MUJA_G6"]), 
    'KWINANA_G5-6': np.array(["KWINANA_G5", "KWINANA_G6"]), 
    'KWINANA_G1-4': np.array(["KWINANA_G1", "KWINANA_G2", "KWINANA_G3", "KWINANA_G4"]), 
    'COCKBURN_CCG1': np.array(["COCKBURN_CCG1"]), 
    'KWINANA_GT2-3': np.array(["KWINANA_GT2", "KWINANA_GT3"]), 
    'KEMERTON_GT11-12': np.array(["KEMERTON_GT11", "KEMERTON_GT12"]), 
    'PINJAR_GT1-11': np.array(["PINJAR_GT1", "PINJAR_GT10", "PINJAR_GT11", "PINJAR_GT2", "PINJAR_GT3", "PINJAR_GT4", "PINJAR_GT5", "PINJAR_GT7", "PINJAR_GT9"]), 
    'PPP_KCP_EG1': np.array(["PPP_KCP_EG1"]), 
    'SWCJV_WORSLEY_COGEN_COG1': np.array(["SWCJV_WORSLEY_COGEN_COG1"]), 
    'ALINTA_PNJ_U1-2': np.array(["ALINTA_PNJ_U1", "ALINTA_PNJ_U2"]), 
    'ALINTA_WGP': np.array(["ALINTA_WGP_GT", "ALINTA_WGP_U2"]), 
    'ALINTA_WWF': np.array(["ALINTA_WWF"]), 
    'BADGINGARRA_WF1': np.array(["BADGINGARRA_WF1"]), 
    'YANDIN_WF1': np.array(["YANDIN_WF1"]), 
    'BW2_BLUEWATERS_G1-2': np.array(["BW2_BLUEWATERS_G1", "BW1_BLUEWATERS_G2"]), 
    'MUJA_G1-4': np.array(["MUJA_G1", "MUJA_G2", "MUJA_G3", "MUJA_G4"]), 
    'NEWGEN_KWINANA_CCG1': np.array(["NEWGEN_KWINANA_CCG1"]), 
    'NEWGEN_NEERABUP_GT1': np.array(["NEWGEN_NEERABUP_GT1"]), 
    'PERTHENERGY_KWINANA_GT1': np.array(["PERTHENERGY_KWINANA_GT1"]), 
    'GREENOUGH_RIVER_PV1': np.array(["GREENOUGH_RIVER_PV1"]), 
    'MERSOLAR_PV1': np.array(["MERSOLAR_PV1"]), 
    'ALBANY_WF1': np.array(["ALBANY_WF1"]), 
    'EDWFMAN_WF1': np.array(["EDWFMAN_WF1"]), 
    'INVESTEC_COLLGAR_WF1': np.array(["INVESTEC_COLLGAR_WF1"]), 
    'MWF_MUMBIDA_WF1': np.array(["MWF_MUMBIDA_WF1"]), 
    'WARRADARGE_WF1': np.array(["WARRADARGE_WF1"])
}

# Generators used in estimating production costs
generators_use_estimation = np.concatenate(tuple([value for key, value in generator_groupings.items()]))
