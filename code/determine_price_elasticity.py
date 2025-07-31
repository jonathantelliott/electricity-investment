# %%
# Import packages
import numpy as np
import statsmodels.regression.linear_model as linear_model
import pandas as pd
import global_vars as gv
import wholesale.demand as demand

# %%
# Import data
dates = np.load(gv.dates_file)
hours = np.arange(gv.num_intervals_in_day)
loaded = np.load(gv.energy_gen_file)
energy_gen = np.copy(loaded['arr_0'])
loaded.close()
tariffs = np.load(gv.residential_tariff_file)
load_curtailment = np.load(gv.load_curtailment_file)
air_temps = np.load(gv.air_temps_file)
wet_bulb_temps = np.load(gv.wet_bulb_temps_file)
wholesale_prices = np.load(gv.prices_realtime_file)

# %%
# Reshape arrays
hours = np.reshape(np.tile(hours[np.newaxis,:], (dates.shape[0], 1)), (-1,))
dates = np.repeat(dates, gv.num_intervals_in_day)
energy_gen = np.reshape(energy_gen, (energy_gen.shape[0], -1))
q_satisfied = np.nansum(energy_gen, axis=0)
intervals = np.reshape(np.tile(np.arange(gv.num_intervals_in_day)[np.newaxis,:], (tariffs.shape[0],1)), (-1,))
tariffs = np.repeat(tariffs, gv.num_intervals_in_day)
load_curtailment = np.reshape(load_curtailment, (-1,))
air_temps = np.reshape(air_temps, (-1,))
wet_bulb_temps = np.reshape(wet_bulb_temps, (-1,))
wholesale_prices = np.reshape(wholesale_prices, (-1,))

# %%
# Construct sample definition
buffer_num_days = 15
include_num_days = 30
use_values_for_elast = np.zeros((dates.shape[0],), dtype=bool)
for yr in np.unique(pd.to_datetime(dates).year):
    dates_use_year_bf = np.arange(np.datetime64(str(yr).zfill(4) + "-07-01") - include_num_days, np.datetime64(str(yr).zfill(4) + "-07-01") - buffer_num_days)
    dates_use_year_after = np.arange(np.datetime64(str(yr).zfill(4) + "-07-01") + buffer_num_days, np.datetime64(str(yr).zfill(4) + "-07-01") + include_num_days)
    use_values_for_elast[np.isin(dates, np.concatenate((dates_use_year_bf, dates_use_year_after)))] = True
start_year = 2014
sample = use_values_for_elast & ~np.isnan(tariffs) & (pd.to_datetime(dates).year >= start_year) # only include those after carbon tax
dates = dates[sample]
hours = hours[sample]
q_satisfied = q_satisfied[sample]
intervals = intervals[sample]
tariffs = tariffs[sample]
load_curtailment = load_curtailment[sample]
air_temps = air_temps[sample]
wet_bulb_temps = wet_bulb_temps[sample]
wholesale_prices = wholesale_prices[sample]

# %%
# Construct covariates
const = np.ones(dates.shape[0])
year_dummies = 1.0 * (pd.to_datetime(dates).year.values[:,np.newaxis] == np.unique(pd.to_datetime(dates).year.values)[np.newaxis,:])
tariff_change_dummy = 1.0 * (pd.to_datetime(dates).month.values >= 7) # after tariff change
hours_dummies = 1.0 * (hours[:,np.newaxis] == np.unique(hours)[1:][np.newaxis,:])
base_temp = 18.0
cooling_degree_days = np.maximum(air_temps - base_temp, 0.0)
heating_degree_days = np.maximum(base_temp - air_temps, 0.0)
log_p = np.log(tariffs)
log_qbar = np.log(q_satisfied + load_curtailment)
regression_description = { # covariates, whether has year dummies, whether has tariff change dummy, where has half-hour dummies, whether has temperature
    '1': (np.concatenate((const[:,np.newaxis], log_p[:,np.newaxis]), axis=1), log_qbar, False, False, False, False, False), 
    '2': (np.concatenate((const[:,np.newaxis], log_p[:,np.newaxis], year_dummies[:,1:]), axis=1), log_qbar, True, False, False, False, False), 
    '3': (np.concatenate((const[:,np.newaxis], log_p[:,np.newaxis], year_dummies[:,1:], tariff_change_dummy[:,np.newaxis]), axis=1), log_qbar, True, True, False, False, False), 
    '4': (np.concatenate((const[:,np.newaxis], log_p[:,np.newaxis], year_dummies[:,1:], tariff_change_dummy[:,np.newaxis], hours_dummies), axis=1), log_qbar, True, True, True, False, False), 
    '5': (np.concatenate((const[:,np.newaxis], log_p[:,np.newaxis], cooling_degree_days[:,np.newaxis], heating_degree_days[:,np.newaxis], year_dummies[:,1:], tariff_change_dummy[:,np.newaxis], hours_dummies), axis=1), log_qbar, True, True, True, True, False), 
    '6': (np.concatenate((const[:,np.newaxis], np.where(wholesale_prices <= 150.0, log_p, np.nan)[:,np.newaxis], cooling_degree_days[:,np.newaxis], heating_degree_days[:,np.newaxis], year_dummies[:,1:], tariff_change_dummy[:,np.newaxis], hours_dummies), axis=1), log_qbar, True, True, True, True, True), 
}

# %%
# Remove observations with INFs or NaNs
for spec, (covariates, y, has_year_dummies, has_tariff_change, has_hour_dummies, has_temp, drop_high_price) in regression_description.items():
    # Remove rows with NaN or INF values in the covariates
    valid_rows = np.all(np.isfinite(covariates), axis=1)  # Check each row for finite values (no NaN or INF)
    filtered_covariates = covariates[valid_rows]
    filtered_y = y[valid_rows]  # Filter the dependent variable to match
    
    # Overwrite the entry in regression_description with filtered data
    regression_description[spec] = (filtered_covariates, filtered_y, has_year_dummies, has_tariff_change, has_hour_dummies, has_temp, drop_high_price)

# %%
# Run regressions and create table

# Begin table
tex_table = f""
tex_table += f"\\begin{{tabular}}{{ l" + f"c" * (2 * len(regression_description.keys()) - 1)  + f" }} \n"
tex_table += f"\\hline \n"
tex_table += f" & " + f" & & ".join([f"({key})" for key, item in regression_description.items()]) + f" \\\\ \n"
tex_table += f" ".join([f"\\cline{{{2 + i * 2}-{2 + i * 2}}}" for i in range(len(regression_description.keys()))]) + f" \\\\ \n"

# Add estimates
tex_table += f"\\textit{{Estimates}}" + " &" * (2 * len(regression_description.keys()) - 1) + f" \\\\ \n"
tex_table += f"$\\quad$ $\\hat{{\\epsilon}}$ & "
for key, item in regression_description.items():
    model = linear_model.OLS(item[1], item[0], hasconst=True)
    res = model.fit()
    tex_table += f"{-res.params[1]:.3f}"
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f" & "
for key, item in regression_description.items():
    model = linear_model.OLS(item[1], item[0], hasconst=True)
    res = model.fit()
    tex_table += f"({res.HC3_se[1]:.3f})"
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += " &" * (2 * len(regression_description.keys()) - 1) + f" \\\\ \n"

# Add specification parameters
tex_table += f"\\textit{{Controls}}" + " &" * (2 * len(regression_description.keys()) - 1) + f" \\\\ \n"
tex_table += f"$\\quad$ constant & "
for key, item in regression_description.items():
    tex_table += f"\\checkmark"
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f"$\\quad$ year effects & "
for key, item in regression_description.items():
    tex_table += f"\\checkmark" if item[2] else f""
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f"$\\quad$ month effects & "
for key, item in regression_description.items():
    tex_table += f"\\checkmark" if item[3] else f""
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f"$\\quad$ half-hour dummies & "
for key, item in regression_description.items():
    tex_table += f"\\checkmark" if item[4] else f""
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f"$\\quad$ temperature & "
for key, item in regression_description.items():
    tex_table += f"\\checkmark" if item[5] else f""
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f"$\\quad$ drop high $P_{{h}}$ & "
for key, item in regression_description.items():
    tex_table += f"\\checkmark" if item[6] else f""
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += " &" * (2 * len(regression_description.keys()) - 1) + f" \\\\ \n"
tex_table += f"\\textit{{Num. obs.}} & "
for key, item in regression_description.items():
    tex_table += f"{item[0].shape[0]:,}".replace(",", "\\,")
    if key != list(regression_description.keys())[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f"\\hline \n \\end{{tabular}} \n"

print(tex_table, flush=True)

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
    
create_file(gv.tables_path + "demand_elasticity_estimates.tex", tex_table)
create_file(gv.stats_path + "demand_elasticity_estimates_buffer_num_days.tex", f"{buffer_num_days}")
create_file(gv.stats_path + "demand_elasticity_estimates_include_num_days.tex", f"{include_num_days}")
create_file(gv.stats_path + "demand_elasticity_estimates_start_year.tex", f"{start_year}")
create_file(gv.stats_path + "demand_elasticity_estimate.tex", f"{linear_model.OLS(regression_description['5'][1], regression_description['5'][0], hasconst=True).fit().params[1]:.3f}")

# %%
# Save estimate arrays
demand_elasticity_estimates = np.array([-linear_model.OLS(item[1], item[0], hasconst=True).fit().params[1] for key, item in regression_description.items()])
demand_elasticity_estimates_se = np.array([linear_model.OLS(item[1], item[0], hasconst=True).fit().HC3_se[1] for key, item in regression_description.items()])
np.save(gv.arrays_path + "demand_elasticity_estimates.npy", demand_elasticity_estimates)
np.save(gv.arrays_path + "demand_elasticity_estimates_se.npy", demand_elasticity_estimates_se)
