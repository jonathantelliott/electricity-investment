import numpy as np

# Data location
scratch_loc = "/scratch/jte254/"
home_loc = "/home/jte254/"
data_loc = scratch_loc + "electricity/data/"
data_path = data_loc + "raw_data/wem_data/" #"/Volumes/My Passport/wem_data/"
arrays_path = data_loc + "processed_data/numpy_arrays/"
results_loc = home_loc + "electricity-investment/results/"
tables_path = results_loc + "tables/"
graphs_path = results_loc + "graphs/"
stats_path = results_loc + "stats/"


# Relevant dates
start_year = 2006
start_month = 10
start_day = 1
end_year = 2021
end_month = 6
end_day = 30
first_cap_date = np.datetime64("2007-10-01")
refyear = 2015 # reference year for inflation (everything is in terms of AUD in this year)


# Universal variables
num_intervals_in_day = 48


# Data column names
df_trading_col = "Trading Date"
df_interval_col = "Interval Number"
df_participant_col = "Participant Code"
df_facility_col = "Facility Code"
df_bid_offer_col = "Bid or Offer"
df_stem_price_col = "Price ($/MWh)"
df_stem_q_col = "Quantity (MWh)"
df_energy_source_col = "Energy Source"
df_max_gen_col = "Energy Generated (MWh)"
df_enterred_col = "Registered From"
df_price_col = "Clearing Price ($/MWh)"
df_year_from_col = "Year From"
df_year_until_col = "Year Until"
df_cap_price_col = "Reserve Capacity Price ($ per MW per year)"
df_benchmark_price_col = "BRCP ($ per MW per year)"
df_outage_exante_col = "Ex-Ante Outage (MW)"
df_outage_expost_col = "Ex-Post Outage (MW)"
df_outage_type_col = "Outage Type"


# Variable outcomes
bid_name = "Bid"
offer_name = "Offer"


# Data that has to be replaced by hand
drop_these = np.array(["ALINTA_WGP_AGG", "MHPS"]) # facilities that show up only a few times - I think it's a case of improper naming in EBS
energy_source_replace_dict = { # source: https://en.wikipedia.org/wiki/List_of_power_stations_in_Western_Australia
    "MUJA": "Coal", 
    "MUNGARRA_GT2": "Gas", 
    "KWINANA_G": "Coal", 
    "KWINANA_GT1": "Gas", 
    "KALAMUNDA": "Landfill Gas", 
    "ALCOA": "Gas", 
    "CANNING_MELVILLE": "Landfill Gas", 
    "GERALDTON_GT1": "Gas", 
    "MUNGARRA": "Gas", 
    "WORSLEY": "Gas", 
    "ALINTA_WGP_AGG": "Dual (Gas / Distillate)" # less sure about this, see: https://en.wikipedia.org/wiki/Wagerup_Power_Station
} # these are facilities that I don't have information on in facilities.csv


participant_name_replace_dict = {
    "GRIFFIN2": "GRIFFINP"
}


# How many strategic firms will we consider?
num_strategic_firms = 3


# Should we presave the data we process?
use_saved = True
save_arrays = True


# File names
# Facilities-level
dates_file = arrays_path + "dates.npy"
facilities_file = arrays_path + "facilities.npy"
participants_file = arrays_path + "participants.npy"
capacities_file = arrays_path + "capacities.npy"
energy_sources_file = arrays_path + "energy_sources.npy"
exited_file = arrays_path + "exited.npy"
energy_gen_file = arrays_path + "energy_gen.npz"
prices_file = arrays_path + "prices.npy"
exited_date_file = arrays_path + "exited_date.npy"
entered_date_file = arrays_path + "entered_date.npy"
capacity_price_file = arrays_path + "capacity_price.npy"
cap_date_from_file = arrays_path + "cap_date_from.npy"
cap_date_until_file = arrays_path + "cap_date_until.npy"
capacity_commitments_file = arrays_path + "capacity_commitments.npy"
outages_file = arrays_path + "outages.npz"
coal_price_file = arrays_path + "coal_price.npy"
gas_price_file = arrays_path + "gas_price.npy"

# Participant-level
firms_file = arrays_path + "firms.npy"
firm_energy_gen_file = arrays_path + "firm_energy_gen.npy"
firm_capacity_file = arrays_path + "firm_capacity.npy"
firm_effective_capacity_file = arrays_path + "firm_effective_capacity.npy"
firm_outages_exante_matrix_file = arrays_path + "firm_outages_exante_matrix.npy"
firm_outages_expost_matrix_file = arrays_path + "firm_outages_expost_matrix.npy"
largest_by_energy_gen_file = arrays_path + "largest_by_energy_gen.npy"
largest_by_cap_file = arrays_path + "largest_by_cap.npy"

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
emissions_file = arrays_path + "emissions.npy"

# Wholesale estimation results
N_wholesale = 2500
yr_wholesale = np.array([2015, 2015])
wholesale_specification_array = np.array([0, 1])
include_corr_eps = np.array([False, True])
include_corr_lQbar_dwind = np.array([False, True])
include_indiv_generators = np.array([False, False])
wholesale_specification_use = 1

# Parameters governing wholesale market expected profit simulation
K_rep = { # need a better way to determine these numbers
    "Coal": 100.0, 
    "Gas": 50.0, 
    "Wind": 50.0
}

H = 48.0 * 365.0
lambda_scheduled = 1.0 / 1440.0
lambda_intermittent = 1.0 / 17280.0
rho = 3.0

num_firm_sources = 2 + 2 + 2 + 3 # this is the number of arrays below - we're ignoring sources never used for the sake of tractability
K_1_coal = np.array([400.0, 800.0, 1200.0, 1600.0, 2000.0, 2400.0])
K_1_gas = np.array([1500., 1600., 1700., 1800., 1900., 2000.])
K_2_gas = np.array([500., 700., 900., 1100.])
K_2_wind = np.array([0.0, 200.0, 400.0, 600.0])
K_3_coal = np.array([0., 400., 800.])
K_3_wind = np.array([0.])
K_c_coal = np.array([0., 200., 400.])
K_c_gas = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
K_c_wind = np.array([0.0, 150.0, 300.0, 450.0, 600.0, 750.0, 900.0])

coal_ctrfctl_strat = np.linspace(0.0, 2400.0, 4)
coal_ctrfctl_comp = np.linspace(0.0, 900.0, 4)
gas_ctrfctl_strat = np.linspace(600.0, 2400.0, 4)
gas_ctrfctl_comp = np.linspace(300.0, 1200.0, 4)
wind_ctrfctl_strat = np.linspace(0.0, 1200.0, 4)
wind_ctrfctl_comp = np.linspace(0.0, 1500.0, 4)
K_1_coal_ctrfctl = coal_ctrfctl_strat
K_1_gas_ctrfctl = gas_ctrfctl_strat
K_2_gas_ctrfctl = gas_ctrfctl_strat
K_2_wind_ctrfctl = wind_ctrfctl_strat
K_3_coal_ctrfctl = coal_ctrfctl_strat
K_3_wind_ctrfctl = wind_ctrfctl_strat
K_c_coal_ctrfctl = coal_ctrfctl_comp
K_c_gas_ctrfctl = gas_ctrfctl_comp
K_c_wind_ctrfctl = wind_ctrfctl_comp

num_draws_wholesale_profit = 500
Erho = 6.
voll = 50000.0 # $ / MWh lost
scc = 70.0 / 1000.0 # $ / kg of CO2
cap_payments = np.concatenate((np.linspace(0.0, 50000.0, 10), np.linspace(50000.0, 200000.0, 6)[1:])) # np.linspace(0.0, 200000.0, 10)
emissions_taxes = np.linspace(0.0, 3.0 * scc, 10)
renew_prod_subsidies = np.linspace(0.0, 50.0, 7)[1:] # 1: b/c 0 already captured by emissions_taxes
inv_tax_ff = np.array([0.0]) # $ / kW
inv_tax_r = np.linspace(0.0, 0.9, 10) # % off

price_elast = 0.09#0.27
retail_component = 0.0295 * 1000.0 # from 2014 Residential Electricity Price Trends (Australia Energy Market Commission), convert to MWh
networks_component = 0.1314 * 1000.0 # from 2014 Residential Electricity Price Trends (Australia Energy Market Commission), convert to MWh
fixed_P_component = retail_component + networks_component

# Parameters governing capacity cost data
convert_cap_units = 1000. # need to go from kW (in capacity cost data) to MW (units in AEMO data)


# Parameters governing dynamic game
num_add_years_array = np.array([5, 10, 15]) # number of additional years after data to add before terminal state
include_beta = np.array([False, False, False])
include_F = np.array([False, False, False])
beta_impute = 0.95
scale_profits = 1. / 100000000.

error_band_lb = 0.1
error_band_ub = 0.9

delay_size = 10
