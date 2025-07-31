# %%
# Import packages
import global_vars as gv

import numpy as np
import pandas as pd
import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm

# %%
# Import data

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
dsp_quantities = np.tile(dsp_quantities[:,np.newaxis], (1,prices.shape[1]))
dsp_quantity_dispacthed = np.load(gv.dsp_dispatch_file)
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

# %%
# What fraction do we capture by only looking at the largest technologies?

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
    
create_file(f"{gv.stats_path}frac_sources_captured.tex", f"{np.nansum(energy_gen[np.isin(energy_sources, gv.use_sources),:,:]) / np.nansum(energy_gen) * 100.0:.1f}")

# %%
# Use only certain generators
generators_use = np.concatenate(tuple([value for key, value in gv.generator_groupings.items()]))
include_facilities = np.isin(facilities, generators_use)
facilities = facilities[include_facilities]
participants = participants[include_facilities]
capacities = capacities[include_facilities]
energy_sources = energy_sources[include_facilities]
energy_sources[energy_sources == gv.gas_cogen] = gv.gas_ocgt
heat_rates = heat_rates[include_facilities]
co2_rates = co2_rates[include_facilities]
transport_charges = transport_charges[include_facilities]
exited = exited[include_facilities]
energy_gen = energy_gen[include_facilities,:,:]
exited_date = exited_date[include_facilities]
entered_date = entered_date[include_facilities]
outages_exante = outages_exante[include_facilities,:,:]
outages_expost = outages_expost[include_facilities,:,:]
outages_type = outages_type[include_facilities,:,:]
balancing_avail = balancing_avail[include_facilities,:,:]

# %%
# Create summary stats

# Begin table
tex_table = f""
tex_table += f"\\begin{{tabular}}{{ lccccccccc }} \n"
tex_table += f"\\hline \n"
tex_table += f" & Mean & & Std. Dev. & & 5th Pctile. & & 95th Pctile. & & Num. Obs. \\\\ \n"
tex_table += f" \\cline{{2-2}} \\cline{{4-4}} \\cline{{6-6}} \\cline{{8-8}} \\cline{{10-10}} \\\\ \n"

# Add half-hourly variables
tex_table += f"\\textit{{Half-hourly variables}} & & & & & & & & & \\\\ \n"
first_date_use = np.datetime64("2012-07-01") # first day of balancing market
dates_use = dates >= first_date_use
prices_use = prices[dates_use,:]
dsp_quantities_use = dsp_quantities[dates_use,:]
dsp_quantity_dispacthed_use = dsp_quantity_dispacthed[dates_use,:]
tex_table += f"$\\quad$ price (A\\$/MWh) & {np.mean(prices_use):,.2f} & & {np.std(prices_use):,.2f} & & {np.percentile(prices_use, 5):,.2f} & & {np.percentile(prices_use, 95):,.2f} & & {np.sum(~np.isnan(prices_use)):,.0f} \\\\ \n".replace(",", "\\,")
total_production = np.nansum(energy_gen, axis=0)
tex_table += f"$\\quad$ total production (MWh) & {np.mean(total_production):,.2f} & & {np.std(total_production):,.2f} & & {np.percentile(total_production, 5):,.2f} & & {np.percentile(total_production, 95):,.2f} & & {np.sum(~np.isnan(total_production)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f"$\\quad$ load curtailed (MWh) & {np.mean(load_curtailed):,.2f} & & {np.std(load_curtailed):,.2f} & & {np.min(load_curtailed):,.2f}${{}}^{{*}}$ & & {np.max(load_curtailed):,.2f}${{}}^{{*}}$ & & {np.sum(~np.isnan(load_curtailed)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f"$\\quad$ demand response dispatched (MWh) & {np.mean(dsp_quantity_dispacthed_use):,.2f} & & {np.std(dsp_quantity_dispacthed_use):,.2f} & & {np.min(dsp_quantity_dispacthed_use):,.2f}${{}}^{{*}}$ & & {np.max(dsp_quantity_dispacthed_use):,.2f}${{}}^{{*}}$ & & {np.sum(~np.isnan(dsp_quantity_dispacthed_use)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f"$\\quad$ demand response quantity available (MWh) & {np.mean(dsp_quantities_use):,.2f} & & {np.std(dsp_quantities_use):,.2f} & & {np.percentile(dsp_quantities_use, 5):,.2f} & & {np.percentile(dsp_quantities_use, 95):,.2f} & & {np.sum(~np.isnan(dsp_quantities_use)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f"$\\quad$ fraction generated by & & & & & & & & & \\\\ \n"
frac_coal = np.nansum(energy_gen[energy_sources == gv.coal], axis=0) / total_production * 100.0
tex_table += f"$\\quad$ $\\quad$ coal (\\%) & {np.mean(frac_coal):,.2f} & & {np.std(frac_coal):,.2f} & & {np.percentile(frac_coal, 5):,.2f} & & {np.percentile(frac_coal, 95):,.2f} & & {np.sum(~np.isnan(frac_coal)):,.0f} \\\\ \n".replace(",", "\\,")
frac_gas = np.nansum(energy_gen[np.isin(energy_sources, gv.natural_gas)], axis=0) / total_production * 100.0
tex_table += f"$\\quad$ $\\quad$ natural gas (\\%) & {np.mean(frac_gas):,.2f} & & {np.std(frac_gas):,.2f} & & {np.percentile(frac_gas, 5):,.2f} & & {np.percentile(frac_gas, 95):,.2f} & & {np.sum(~np.isnan(frac_gas)):,.0f} \\\\ \n".replace(",", "\\,")
frac_solar = np.nansum(energy_gen[energy_sources == gv.solar], axis=0) / total_production * 100.0
tex_table += f"$\\quad$ $\\quad$ solar (\\%) & {np.mean(frac_solar):,.2f} & & {np.std(frac_solar):,.2f} & & {np.percentile(frac_solar, 5):,.2f} & & {np.percentile(frac_solar, 95):,.2f} & & {np.sum(~np.isnan(frac_solar)):,.0f} \\\\ \n".replace(",", "\\,")
frac_wind = np.nansum(energy_gen[energy_sources == gv.wind], axis=0) / total_production * 100.0
tex_table += f"$\\quad$ $\\quad$ wind (\\%) & {np.mean(frac_wind):,.2f} & & {np.std(frac_wind):,.2f} & & {np.percentile(frac_wind, 5):,.2f} & & {np.percentile(frac_wind, 95):,.2f} & & {np.sum(~np.isnan(frac_wind)):,.0f} \\\\ \n".replace(",", "\\,")
cap_factor = np.zeros(energy_gen.shape)
cap_factor[np.isin(energy_sources, gv.intermittent),:,:] = energy_gen[np.isin(energy_sources, gv.intermittent),:,:] / (capacities[np.isin(energy_sources, gv.intermittent),np.newaxis,np.newaxis] / 2.0)
cap_factor[~np.isin(energy_sources, gv.intermittent),:,:] = np.maximum(0.0, np.minimum((capacities[~np.isin(energy_sources, gv.intermittent),np.newaxis,np.newaxis] - outages_expost[~np.isin(energy_sources, gv.intermittent),:,:]) / capacities[~np.isin(energy_sources, gv.intermittent),np.newaxis,np.newaxis], 1.0))
tex_table += f"$\\quad$ fraction capacity available & & & & & & & & & \\\\ \n"
cap_factor_coal = cap_factor[energy_sources == gv.coal,:,:] * 100.0
tex_table += f"$\\quad$ $\\quad$ coal (\\%) & {np.nanmean(cap_factor_coal):,.2f} & & {np.nanstd(cap_factor_coal):,.2f} & & {np.nanpercentile(cap_factor_coal, 5):,.2f} & & {np.nanpercentile(cap_factor_coal, 95):,.2f} & & {np.sum(~np.isnan(cap_factor_coal)):,.0f} \\\\ \n".replace(",", "\\,")
cap_factor_gas = cap_factor[np.isin(energy_sources, gv.natural_gas),:,:] * 100.0
tex_table += f"$\\quad$ $\\quad$ natural gas (\\%) & {np.nanmean(cap_factor_gas):,.2f} & & {np.nanstd(cap_factor_gas):,.2f} & & {np.nanpercentile(cap_factor_gas, 5):,.2f} & & {np.nanpercentile(cap_factor_gas, 95):,.2f} & & {np.sum(~np.isnan(cap_factor_gas)):,.0f} \\\\ \n".replace(",", "\\,")
cap_factor_solar = cap_factor[energy_sources == gv.solar,:,:] * 100.0
tex_table += f"$\\quad$ $\\quad$ solar (\\%) & {np.nanmean(cap_factor_solar):,.2f} & & {np.nanstd(cap_factor_solar):,.2f} & & {np.nanpercentile(cap_factor_solar, 5):,.2f} & & {np.nanpercentile(cap_factor_solar, 95):,.2f} & & {np.sum(~np.isnan(cap_factor_solar)):,.0f} \\\\ \n".replace(",", "\\,")
cap_factor_wind = cap_factor[energy_sources == gv.wind,:,:] * 100.0
tex_table += f"$\\quad$ $\\quad$ wind (\\%) & {np.nanmean(cap_factor_wind):,.2f} & & {np.nanstd(cap_factor_wind):,.2f} & & {np.nanpercentile(cap_factor_wind, 5):,.2f} & & {np.nanpercentile(cap_factor_wind, 95):,.2f} & & {np.sum(~np.isnan(cap_factor_wind)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f" & & & & & & & & & \\\\ \n"

# Add yearly variables
tex_table += f"\\textit{{Yearly variables}} & & & & & & & & & \\\\ \n"
capacity_price_use = capacity_price / 1000.0
tex_table += f"$\\quad$ capacity price (thousand A\$/MW) & {np.mean(capacity_price_use):,.2f} & & {np.std(capacity_price_use):,.2f} & & {np.percentile(capacity_price_use, 5):,.2f} & & {np.percentile(capacity_price_use, 95):,.2f} & & {np.sum(~np.isnan(capacity_price_use)):,.0f} \\\\ \n".replace(",", "\\,")
select_july1 = np.isin(dates, np.array([np.datetime64(f"{yr}-07-01") for yr in np.unique(pd.to_datetime(dates).year)]))
max_prices_use = max_prices[select_july1]
tex_table += f"$\\quad$ price cap (A\$/MWh) & {np.mean(max_prices_use):,.2f} & & {np.std(max_prices_use):,.2f} & & {np.percentile(max_prices_use, 5):,.2f} & & {np.percentile(max_prices_use, 95):,.2f} & & {np.sum(~np.isnan(max_prices_use)):,.0f} \\\\ \n".replace(",", "\\,")
retail_prices_use = tariffs[select_july1]
tex_table += f"$\\quad$ retail price variable component (A\$/MWh) & {np.nanmean(retail_prices_use):,.2f} & & {np.nanstd(retail_prices_use):,.2f} & & {np.nanpercentile(retail_prices_use, 5):,.2f} & & {np.nanpercentile(retail_prices_use, 95):,.2f} & & {np.sum(~np.isnan(retail_prices_use)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f" & & & & & & & & & \\\\ \n"

# Add generator characteristics
tex_table += f"\\textit{{Generator variables}} & & & & & & & & & \\\\ \n"
tex_table += f"$\\quad$ capacity & & & & & & & & & \\\\ \n"
coal_capacity = capacities[energy_sources == gv.coal]
tex_table += f"$\\quad$ $\\quad$ coal (MW) & {np.mean(coal_capacity):,.2f} & & {np.std(coal_capacity):,.2f} & & {np.percentile(coal_capacity, 5):,.2f} & & {np.percentile(coal_capacity, 95):,.2f} & & {np.sum(~np.isnan(coal_capacity)):,.0f} \\\\ \n".replace(",", "\\,")
gas_capacity = capacities[np.isin(energy_sources, gv.natural_gas)]
tex_table += f"$\\quad$ $\\quad$ natural gas (MW) & {np.mean(gas_capacity):,.2f} & & {np.std(gas_capacity):,.2f} & & {np.percentile(gas_capacity, 5):,.2f} & & {np.percentile(gas_capacity, 95):,.2f} & & {np.sum(~np.isnan(gas_capacity)):,.0f} \\\\ \n".replace(",", "\\,")
solar_capacity = capacities[energy_sources == gv.solar]
tex_table += f"$\\quad$ $\\quad$ solar (MW) & {np.mean(solar_capacity):,.2f} & & {np.std(solar_capacity):,.2f} & & {np.percentile(solar_capacity, 5):,.2f} & & {np.percentile(solar_capacity, 95):,.2f} & & {np.sum(~np.isnan(solar_capacity)):,.0f} \\\\ \n".replace(",", "\\,")
wind_capacity = capacities[energy_sources == gv.wind]
tex_table += f"$\\quad$ $\\quad$ wind (MW) & {np.mean(wind_capacity):,.2f} & & {np.std(wind_capacity):,.2f} & & {np.percentile(wind_capacity, 5):,.2f} & & {np.percentile(wind_capacity, 95):,.2f} & & {np.sum(~np.isnan(wind_capacity)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f"$\\quad$ heat rate & & & & & & & & & \\\\ \n"
coal_heat_rates = heat_rates[energy_sources == gv.coal]
tex_table += f"$\\quad$ $\\quad$ coal (GJ/MWh) & {np.mean(coal_heat_rates):,.2f} & & {np.std(coal_heat_rates):,.2f} & & {np.percentile(coal_heat_rates, 5):,.2f} & & {np.percentile(coal_heat_rates, 95):,.2f} & & {np.sum(~np.isnan(coal_heat_rates)):,.0f} \\\\ \n".replace(",", "\\,")
gas_heat_rates = heat_rates[np.isin(energy_sources, gv.natural_gas)]
tex_table += f"$\\quad$ $\\quad$ natural gas (GJ/MWh) & {np.mean(gas_heat_rates):,.2f} & & {np.std(gas_heat_rates):,.2f} & & {np.percentile(gas_heat_rates, 5):,.2f} & & {np.percentile(gas_heat_rates, 95):,.2f} & & {np.sum(~np.isnan(gas_heat_rates)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f"$\\quad$ $\\text{{CO}}_{{2}}$ emissions rate & & & & & & & & & \\\\ \n"
coal_co2_rates = co2_rates[energy_sources == gv.coal]
tex_table += f"$\\quad$ $\\quad$ coal (kg$\\text{{CO}}_{{2}}$-eq/MWh) & {np.mean(coal_co2_rates):,.2f} & & {np.std(coal_co2_rates):,.2f} & & {np.percentile(coal_co2_rates, 5):,.2f} & & {np.percentile(coal_co2_rates, 95):,.2f} & & {np.sum(~np.isnan(coal_co2_rates)):,.0f} \\\\ \n".replace(",", "\\,")
gas_co2_rates = co2_rates[np.isin(energy_sources, gv.natural_gas)]
tex_table += f"$\\quad$ $\\quad$ natural gas (kg$\\text{{CO}}_{{2}}$-eq/MWh) & {np.mean(gas_co2_rates):,.2f} & & {np.std(gas_co2_rates):,.2f} & & {np.percentile(gas_co2_rates, 5):,.2f} & & {np.percentile(gas_co2_rates, 95):,.2f} & & {np.sum(~np.isnan(gas_co2_rates)):,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f" & & & & & & & & & \\\\ \n"

# Finish table
tex_table += f"\\hline \n \\end{{tabular}} \n"

print(tex_table, flush=True)
    
create_file(gv.tables_path + "summary_statistics.tex", tex_table)

# %%
# Plot evolution of variables over time

# Set up figure
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6.0*3,5.0*1))
source_names = ["coal", "natural gas", "solar", "wind"]
source_color = ["#a9a9a9", "#9A6324", "#ffe119", "#3cb44b"]
source_include = np.concatenate((np.isin(energy_sources, np.array([gv.coal]))[:,np.newaxis], np.isin(energy_sources, gv.natural_gas)[:,np.newaxis], np.isin(energy_sources, np.array([gv.solar]))[:,np.newaxis], np.isin(energy_sources, np.array([gv.wind]))[:,np.newaxis]), axis=1)
select_oct1 = np.isin(dates, np.array([np.datetime64(f"{yr}-10-01") for yr in np.unique(pd.to_datetime(dates).year)]))
years_unique = np.unique(pd.to_datetime(dates[select_oct1]).year)
width = 0.775
width_point = 0.5
x = np.arange(years_unique.shape[0])  # the label locations
select_use = np.arange(x.shape[0]) % 3 == 0
x_use = x[select_use]
xlabels_use = years_unique[select_use]
alpha = 0.85
grid_line_alpha = 0.4
grid_line_lw = 1.75
title_size = 18.0
label_size = 13.5

# Capacities
axs[0].set_title("capacities", size=title_size)
capacities_tile = np.tile(capacities[:,np.newaxis], (1,dates.shape[0]))
select_replacement_region = np.ix_(facilities == "GREENOUGH_RIVER_PV1", dates < np.datetime64("2019-06-01"))
capacities_tile[select_replacement_region] = np.nanmax(energy_gen[select_replacement_region]) * 2.0 # previous capacity
capacities_time = capacities_tile * (dates[np.newaxis,:] >= entered_date[:,np.newaxis]) * (dates[np.newaxis,:] <= exited_date[:,np.newaxis])
capacities_tech_time = np.zeros((len(source_names), capacities_time.shape[1]))
for i, source in enumerate(source_names):
    capacities_tech_time[i,:] = np.sum(capacities_time[source_include[:,i],:], axis=0)
start_value = np.zeros((x.shape[0]))
for i, source in enumerate(source_names):
    rects_source = axs[0].bar(x, capacities_tech_time[i,select_oct1], width, bottom=start_value, color=source_color[i], alpha=alpha, label=source)
    start_value = start_value + capacities_tech_time[i,select_oct1]
box = axs[0].get_position()
axs[0].set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=len(source_names))
axs[0].set_xticks(x_use)
axs[0].set_xticklabels(xlabels_use, rotation=45)
axs[0].set_xlabel("year", size=label_size)
axs[0].set_ylabel("MW", size=label_size)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].yaxis.grid(True, color="black", alpha=grid_line_alpha, lw=grid_line_lw)

# Production shares
axs[1].set_title("production by technology", size=title_size)
production_shares = np.zeros((len(source_names), years_unique.shape[0]))
for i, source in enumerate(source_names):
    for y, yr in enumerate(years_unique):
        select_source = source_include[:,i]
        select_yr = (dates >= np.datetime64(f"{yr}-10-01")) & (dates < np.datetime64(f"{yr + 1}-10-01"))
        production_shares[i,y] = np.nansum(energy_gen[np.ix_(select_source,select_yr)])
production_shares = production_shares / np.sum(production_shares, axis=0)
start_value = np.zeros((x.shape[0]))
for i, source in enumerate(source_names):
    rects_source = axs[1].bar(x, production_shares[i,:], width, bottom=start_value, color=source_color[i], alpha=alpha, label=source)
    start_value = start_value + production_shares[i,:]
box = axs[1].get_position()
axs[1].set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=len(source_names))
axs[1].set_xticks(x_use)
axs[1].set_xticklabels(xlabels_use, rotation=45)
axs[1].set_ylabel("share", size=label_size)
axs[1].set_xlabel("year", size=label_size)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[1].yaxis.grid(True, color="black", alpha=grid_line_alpha, lw=grid_line_lw)

# Market shares
axs[2].set_title("market shares", size=title_size)
participants_use = ["WPGENER", "ALINTA", "GRIFFINP", "others"]
participants_names = ["Synergy", "Alinta", "Bluewaters Power", "others"]
participants_color = ["#e6194B", "#f58231", "#ffe119", "#bfef45"]
market_shares = np.zeros((len(participants_use), years_unique.shape[0]))
for i, participant in enumerate(participants_use):
    for y, yr in enumerate(years_unique):
        select_generators = participants == participant
        if participant == "others":
            select_generators = ~np.isin(participants, np.array(participants_use)[:-1])
        select_yr = (dates >= np.datetime64(f"{yr}-10-01")) & (dates < np.datetime64(f"{yr + 1}-10-01"))
        market_shares[i,y] = np.nansum(energy_gen[np.ix_(select_generators,select_yr)])
market_shares = market_shares / np.sum(market_shares, axis=0)
start_value = np.zeros((x.shape[0]))
for i, participant in enumerate(participants_use):
    rects_source = axs[2].bar(x, market_shares[i,:], width, bottom=start_value, color=participants_color[i], alpha=alpha, label=participants_names[i])
    start_value = start_value + market_shares[i,:]
box = axs[2].get_position()
axs[2].set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
axs[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=len(source_names))
axs[2].set_xticks(x_use)
axs[2].set_xticklabels(xlabels_use, rotation=45)
axs[2].set_xlabel("year", size=label_size)
axs[2].set_ylabel("share", size=label_size)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['left'].set_visible(False)
axs[2].yaxis.grid(True, color="black", alpha=grid_line_alpha, lw=grid_line_lw)

# Show plot
fig.tight_layout()
plt.savefig(f"{gv.graphs_path}evolution_over_time.pdf", transparent=True)
# plt.show()

# %%
# Plot evolution of variables over time

# Set up figure
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(4.0*4,4.0*1), sharey=True)
years_unique = np.linspace(2012, 2021, 4, dtype=int)
width = 1.0
width_point = 0.5
x = np.arange(energy_gen.shape[2])
select_use = np.arange(x.shape[0]) % 4 == 0
x_use = x[select_use]
xlabels_use = np.repeat(np.arange(24), 2)[select_use]
am = "am"
pm = "pm"
xlabels_use = [f"{hour % 12} {am if hour // 12 == 0 else pm}" if (hour != 0) and (hour != 12) else f"12 {am if hour // 12 == 0 else pm}" for hour in xlabels_use]

n_ticks = 6
y1_min = 0.0
y1_max = 3000.0
y2_min = 15.0
y2_max = 115.0

# Plot each year
ax2_dict = {}
ever_price = False
for y, yr in enumerate(years_unique):
    axs[y].set_title(f"{yr}", size=title_size)
    production_totals = np.zeros((len(source_names), energy_gen.shape[2]))
    avg_price = np.zeros(energy_gen.shape[2])
    for i, source in enumerate(source_names):
        for j in range(energy_gen.shape[2]):
            select_source = source_include[:,i]
            select_yr = (dates >= np.datetime64(f"{yr}-10-01")) & (dates < np.datetime64(f"{yr + 1}-10-01"))
            select_halfhour = np.arange(energy_gen.shape[2]) == j
            production_totals[i,j] = np.nansum(energy_gen[np.ix_(select_source,select_yr,select_halfhour)])
            avg_price[j] = np.mean(prices[np.ix_(select_yr,select_halfhour)])
    production_totals = np.concatenate((production_totals[:,-(8*2):], production_totals[:,:-(8*2)]), axis=1) # timing starts at 8 am, want to start at midnight
    avg_price = np.concatenate((avg_price[-(8*2):], avg_price[:-(8*2)]))
    production_totals = production_totals / 365.0 # take average
    production_totals = production_totals * 2.0 # scale to hourly
    start_value = np.zeros((x.shape[0]))
    for i, source in enumerate(source_names):
        plot_vals = start_value + production_totals[i,:]
        axs[y].fill_between(x, start_value, plot_vals, color=source_color[i], alpha=alpha, label=source if y == 0 else None)
        start_value = plot_vals
    if yr >= 2012: # start of balancing market
        ax2 = axs[y].twinx()
        ax2_dict[f"{yr}"] = ax2
        ax2.plot(x, avg_price, color="black", lw=2.0, label="average price" if not ever_price else None)
        ever_price = True
    axs[y].set_xticks(x_use)
    axs[y].set_xticklabels(xlabels_use, rotation=45)
    axs[y].set_yticks(np.linspace(y1_min, y1_max, n_ticks))
    axs[y].set_ylabel("avg. production (MWh)", size=label_size)
    axs[y].set_ylim(y1_min, y1_max)
    ax2.set_ylabel("avg. price (A$/MWh)", size=label_size)
    ax2.set_yticks(np.linspace(y2_min, y2_max, n_ticks))
    ax2.set_ylim(y2_min, y2_max)
    if yr != years_unique[-1]:
        ax2.set_yticks([])
    axs[y].spines['top'].set_visible(False)
    axs[y].spines['left'].set_visible(False)
    axs[y].yaxis.grid(True, color="black", alpha=grid_line_alpha, lw=grid_line_lw)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)

fig.legend(loc="lower center", fancybox=True, shadow=True, ncol=len(source_names) + 1)

# Show plot
plt.savefig(f"{gv.graphs_path}evolution_over_time_throughout_day.pdf", transparent=True, bbox_inches="tight")
# plt.show()

# %%
# Explore relationship between capacity price and capacity

# Selection of dates
cap_start_dates = dates[np.isin(dates, cap_date_from)]
cap_until_dates = dates[np.isin(dates, cap_date_until)]
cap_price = capacity_price[np.isin(cap_date_from, cap_start_dates)]

# Total capacity
generator_isin = (exited_date[:,np.newaxis] >= cap_until_dates[np.newaxis,:]) & (entered_date[:,np.newaxis] <= cap_until_dates[np.newaxis,:])
capacity_isin = capacities[:,np.newaxis] * generator_isin
ff_capacity = np.sum(capacity_isin[np.isin(energy_sources, [gv.coal, gv.gas_ccgt, gv.gas_ocgt, gv.gas_cogen]),:], axis=0)
renew_capacity = np.sum(capacity_isin[np.isin(energy_sources, gv.intermittent),:], axis=0)

# Create monthly date range covering full period
monthly_dates = pd.date_range(start=cap_until_dates[0], end=cap_until_dates[-1], freq='MS')  # 'MS' = Month Start
monthly_dates_np = monthly_dates.to_numpy()

# Construct generator capacity
generator_isin_monthly = (exited_date[:,np.newaxis] >= monthly_dates_np[np.newaxis,:]) & (entered_date[:, np.newaxis] <= monthly_dates_np[np.newaxis,:])
capacity_isin_monthly = capacities[:,np.newaxis] * generator_isin_monthly
fossil_mask = np.isin(energy_sources, [gv.coal, gv.gas_ccgt, gv.gas_ocgt, gv.gas_cogen])
fossil_capacity_monthly = np.sum(capacity_isin_monthly[fossil_mask,:], axis=0)
renew_mask = np.isin(energy_sources, gv.intermittent)
renew_capacity_monthly = np.sum(capacity_isin_monthly[renew_mask,:], axis=0)

cap_until_dates_dt = pd.to_datetime(cap_until_dates)
cap_years = cap_until_dates_dt.year
cap_price_by_cap_year = dict(zip(cap_years, cap_price))
monthly_years = monthly_dates.year
monthly_months = monthly_dates.month
monthly_cap_years = np.where(monthly_months >= 10, monthly_years + 1, monthly_years)
monthly_cap_price = np.array([cap_price_by_cap_year[y] for y in monthly_cap_years])

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)

# Fossil capacity
ax1 = axes[0]
ax1.plot(monthly_dates, fossil_capacity_monthly, color='tab:blue', lw=2.0)
ax1.set_ylabel('capacity (MW)', color='tab:blue', size=label_size)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_title('fossil fuel capacity', size=title_size)

# Capacity price (stepwise constant)
ax1b = ax1.twinx()
ax1b.plot(monthly_dates, monthly_cap_price, color='tab:red', linestyle='--', lw=2.0)
ax1b.set_ylabel('capacity price (A$/MW)', color='tab:red', size=label_size)
ax1b.tick_params(axis='y', labelcolor='tab:red')

# Renewable capacity
ax2 = axes[1]
ax2.plot(monthly_dates, renew_capacity_monthly, color='tab:green', lw=2.0)
ax2.set_ylabel('capacity (MW)', color='tab:green', size=label_size)
ax2.tick_params(axis='y', labelcolor='tab:green')
ax2.set_title('renewable capacity', size=title_size)

# Capacity price
ax2b = ax2.twinx()
ax2b.plot(monthly_dates, monthly_cap_price, color='tab:red', linestyle='--', lw=2.0)
ax2b.set_ylabel('capacity price (A$/MW)', color='tab:red', size=label_size)
ax2b.tick_params(axis='y', labelcolor='tab:red')

# Make capacity price y-axes start at 0
ax1b.set_ylim(bottom=0)
ax2b.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f"{gv.graphs_path}capacity_vs_cap_price.pdf", transparent=True, bbox_inches="tight")
# plt.show()

# %%
# Estimate model of whether active

avg_capacity_price = np.copy(cap_price)
avg_capacity_price_3yr = np.zeros(avg_capacity_price.shape)
for i in range(avg_capacity_price_3yr.shape[0]):
    avg_capacity_price_3yr[i] = np.mean(capacity_price[i:(i+3)])
avg_coal_price = np.zeros(avg_capacity_price.shape)
avg_gas_price = np.zeros(avg_capacity_price.shape)
total_load = np.zeros(avg_capacity_price.shape)
avg_wholesale_price = np.zeros(avg_capacity_price.shape)
for i in range(avg_coal_price.shape[0]):
    select_dates = (dates >= cap_start_dates[i]) & (dates <= cap_until_dates[i])
    avg_coal_price[i] = np.mean(coal_prices[select_dates])
    avg_gas_price[i] = np.mean(gas_prices[select_dates])
    total_load[i] = np.nansum(energy_gen[:,select_dates,:])
    avg_wholesale_price[i] = np.nanmean(prices[select_dates,:])

records = []

for i in range(generator_isin.shape[0]):
    fuel = energy_sources[i]
    if fuel == gv.coal:
        fuel_type = "coal"
    elif fuel in [gv.gas_ccgt, gv.gas_ocgt, gv.gas_cogen]:
        fuel_type = "gas"
    elif fuel == gv.solar:
        fuel_type = "solar"
    elif fuel == gv.wind:
        fuel_type = "wind"
    else:
        continue

    for t in range(generator_isin.shape[1]):
        year = cap_start_dates[t]
        cap_price_ = avg_capacity_price[t]
        cap_price_3yr_ = avg_capacity_price_3yr[t]
        if fuel_type == "coal":
            fuel_cost = avg_coal_price[t] * heat_rates[i]
        elif fuel_type == "gas":
            fuel_cost = avg_gas_price[t] * heat_rates[i]
        else:
            fuel_cost = np.nan
        tot_load = total_load[t]
        wholesale_price = avg_wholesale_price[t]

        records.append({
            "generator_id": i,
            "year_idx": t,
            "year": year,
            "active": int(generator_isin[i,t]),
            "capacity_price": cap_price_ / 100000.0,
            "capacity_price_3yr": cap_price_3yr_ / 100000.0,
            "fuel_cost": fuel_cost,
            "fuel_type": fuel_type,
            "tot_load": tot_load,
            "coal": int(fuel_type == "coal"),
            "wind": int(fuel_type == "wind"),
            "wholesale_price": wholesale_price
        })

df = pd.DataFrame(records)
df_ff = df[df["fuel_type"].isin(["coal", "gas"])]
df_ff = pd.get_dummies(df_ff, columns=["generator_id"], drop_first=True, dtype=float)

# Regression 1: fossil fuels, no controls (LPM)
fe_cols = [col for col in df_ff.columns if col.startswith("generator_id_")]
covariates = ["capacity_price"] + fe_cols
X = df_ff[covariates]
X = sm.add_constant(X)
y = df_ff["active"]
model = sm.OLS(y, X)
result1 = model.fit(cov_type='cluster', cov_kwds={'groups': df_ff["year_idx"]})
print(result1.summary())

# Regression 2: fossil fuels, no controls (logit)
model = sm.Logit(y, X)
result2 = model.fit(cov_type='cluster', cov_kwds={'groups': df_ff["year_idx"]})
result2_marginaleffects = result2.get_margeff(method="dydx", at="overall")
print(result2_marginaleffects.summary())

# Regression 3: fossil fuels, w/ controls (LPM)
covariates = ["capacity_price", "wholesale_price", "fuel_cost"] + fe_cols
X = df_ff[covariates]
X = sm.add_constant(X)
y = df_ff["active"]
model = sm.OLS(y, X)
result3 = model.fit(cov_type='cluster', cov_kwds={'groups': df_ff["year_idx"]})
print(result3.summary())

# Regression 4: fossil fuels, w/ controls (logit)
model = sm.Logit(y, X)
result4 = model.fit(cov_type='cluster', cov_kwds={'groups': df_ff["year_idx"]})
result4_marginaleffects = result4.get_margeff(method="dydx", at="overall")
print(result4.summary())

df_r = df[df["fuel_type"].isin(["solar", "wind"])]
df_r = pd.get_dummies(df_r, columns=["generator_id"], drop_first=True, dtype=float)

# Regression 5: renewables, no controls (LPM)
fe_cols = [col for col in df_r.columns if col.startswith("generator_id_")]
covariates = ["capacity_price", "year_idx"] + fe_cols
X = df_r[covariates]
X = sm.add_constant(X)
y = df_r["active"]
model = sm.OLS(y, X)
result5 = model.fit(cov_type='cluster', cov_kwds={'groups': df_r["year_idx"]})
print(result5.summary())

# Regression 6: renewables no controls (logit)
model = sm.Logit(y, X)
result6 = model.fit(cov_type='cluster', cov_kwds={'groups': df_r["year_idx"]})
result6_marginaleffects = result6.get_margeff(method="dydx", at="overall")
print(result6_marginaleffects.summary())

# Regression 7: renewables, w/ controls (LPM)
covariates = ["capacity_price", "year_idx", "wholesale_price"] + fe_cols
X = df_r[covariates]
X = sm.add_constant(X)
model = sm.OLS(y, X)
result7 = model.fit(cov_type='cluster', cov_kwds={'groups': df_r["year_idx"]})
print(result7.summary())

# Regression 8: renewables, w/ controls (logit)
model = sm.Logit(y, X)
result8 = model.fit(cov_type='cluster', cov_kwds={'groups': df_r["year_idx"]})
result8_marginaleffects = result8.get_margeff(method="dydx", at="overall")
print(result8_marginaleffects.summary())

# Organize results
results = [result1, result2_marginaleffects, result3, result4_marginaleffects, result5, result6_marginaleffects, result7, result8_marginaleffects]
model_names1 = ["fossil", "fossil", "fossil", "fossil", "", "", "", ""]
model_names2 = ["fuels", "fuels", "fuels", "fuels", "renew.", "renew.", "renew.", "renew."]
model_names3 = ["(LPM)", "(logit)", "(LPM)", "(logit)", "(LPM)", "(logit)", "(LPM)", "(logit)"]

# Start table
tex_table = ""
tex_table += "\\begin{tabular}{l" + "c" * (len(results)*2-1) + "} \n"
tex_table += "\\hline\n"
tex_table += " & " + " & & ".join(model_names1) + " \\\\ \n"
tex_table += " & " + " & & ".join(model_names2) + " \\\\ \n"
tex_table += " & " + " & & ".join(model_names3) + " \\\\ \n"
tex_table += " \\cline{2-2} \\cline{4-4} \\cline{6-6} \\cline{8-8} \\cline{10-10} \\cline{12-12} \\cline{14-14} \\cline{16-16} \\\\ \n"

# Coefficients
def coef_row(varname, results, decimals=3):
    row = ""
    row_se = ""
    for i, res in enumerate(results):
        ands = " & " if i == 0 else " & & "
        if hasattr(res, "params"):
            # OLS or Logit
            if varname in res.params:
                beta = res.params[varname]
                se = res.bse[varname]
                row += f"{ands}{beta:.{decimals}f}"
                row_se += f"{ands}({se:.{decimals}f})"
            else:
                row += ands
                row_se += ands
        else:
            # DiscreteMargins — use summary_frame for safety
            try:
                frame = res.summary_frame()
                if varname in frame.index:
                    beta = frame.loc[varname, "dy/dx"]
                    se = frame.loc[varname, "Std. Err."]
                    row += f"{ands}{beta:.{decimals}f}"
                    row_se += f"{ands}({se:.{decimals}f})"
                else:
                    row += ands
                    row_se += ands
            except Exception:
                row += ands
                row_se += ands
    return row + " \\\\ \n" + row_se + " \\\\ \n"
tex_table += "$\\text{cap. price}_{t}$" + coef_row("capacity_price", results)
tex_table += "$\\text{avg. price}_{t}$" + coef_row("wholesale_price", results)
tex_table += "avg. fuel cost${}_{gt}$" + coef_row("fuel_cost", results)
tex_table += " \\\\ \n"
tex_table += "technology f.e. & \\checkmark & & \\checkmark & & \\checkmark & & \\checkmark & & \\checkmark & & \\checkmark & & \\checkmark & & \\checkmark \\\\ \n"
tex_table += "linear time trend &  & &  & &  & &  & & \\checkmark & & \\checkmark & & \\checkmark & & \\checkmark \\\\ \n"
tex_table += "\\textit{Num. obs.} & " + " & & ".join([
    f"{int(res.nobs) if hasattr(res, 'nobs') else int(res.results.nobs)}"
    for res in results
]) + " \\\\ \n"
tex_table += " \\\\ \n"
tex_table += "\\hline \n"
tex_table += "\\end{tabular}"

print(tex_table)
create_file(gv.tables_path + "capacity_price_impact.tex", tex_table)
