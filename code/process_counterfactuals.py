# %%
# Import packages
import numpy as np

import global_vars as gv

import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

import scipy.interpolate as interp
import scipy.ndimage as ndimage

# %%
# Variables governing how script is run

show_output = True
save_output = True

# %%
# Create file function

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()

# %%
# Load arrays

# Description of counterfactual equilibria
with np.load(f"{gv.arrays_path}counterfactual_env_co2tax.npz") as loaded:
    carbon_taxes_linspace = loaded['carbon_taxes_linspace']
with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz") as loaded:
    capacity_payments_linspace = loaded['capacity_payments_linspace']
capacity_payments_linspace2 = np.load(f"{gv.arrays_path}counterfactual_capacity_payments_linspace_extended.npy")
with np.load(f"{gv.arrays_path}counterfactual_env_renewablesubisidies.npz") as loaded:
    renewable_subsidies_linspace = loaded['renewable_subsidies_linspace']
with np.load(f"{gv.arrays_path}counterfactual_results_renewableinvestmentsubisidies.npz") as loaded:
    renewable_investment_subsidy_linspace = loaded['renewable_investment_subsidy_linspace']
with np.load(f"{gv.arrays_path}counterfactual_results_delay_smoothed.npz") as loaded:
    delay_linspace = loaded['delay_linspace']
with np.load(f"{gv.arrays_path}counterfactual_results_capacitypaymentspotprice.npz") as loaded:
    spot_price_multiplier = loaded['spot_price_multiplier']

carbon_tax_capacity_payment_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results.npz") as loaded:
    carbon_tax_capacity_payment_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    carbon_tax_capacity_payment_results['expected_emissions'] = loaded['expected_emissions']
    carbon_tax_capacity_payment_results['expected_blackouts'] = loaded['expected_blackouts']
    carbon_tax_capacity_payment_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    carbon_tax_capacity_payment_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    carbon_tax_capacity_payment_results['expected_total_produced'] = loaded['expected_total_produced']
    carbon_tax_capacity_payment_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    carbon_tax_capacity_payment_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus'] + loaded['expected_dsp_profits']
    carbon_tax_capacity_payment_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    carbon_tax_capacity_payment_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    carbon_tax_capacity_payment_results['expected_revenue'] = loaded['expected_revenue']
    carbon_tax_capacity_payment_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum'] + loaded['expected_dsp_profits_sum']
    carbon_tax_capacity_payment_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    carbon_tax_capacity_payment_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    carbon_tax_capacity_payment_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    carbon_tax_capacity_payment_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    carbon_tax_capacity_payment_results['expected_total_production_cost'] = loaded['expected_total_production_cost']
    
renewable_production_subsidies_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results_renewableproductionsubisidies.npz") as loaded:
    renewable_production_subsidies_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    renewable_production_subsidies_results['expected_emissions'] = loaded['expected_emissions']
    renewable_production_subsidies_results['expected_blackouts'] = loaded['expected_blackouts']
    renewable_production_subsidies_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    renewable_production_subsidies_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    renewable_production_subsidies_results['expected_total_produced'] = loaded['expected_total_produced']
    renewable_production_subsidies_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    renewable_production_subsidies_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus'] + loaded['expected_dsp_profits']
    renewable_production_subsidies_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    renewable_production_subsidies_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    renewable_production_subsidies_results['expected_revenue'] = loaded['expected_revenue']
    renewable_production_subsidies_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    renewable_production_subsidies_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum'] + loaded['expected_dsp_profits_sum']
    renewable_production_subsidies_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    renewable_production_subsidies_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    renewable_production_subsidies_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    renewable_production_subsidies_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    
renewable_investment_subsidies_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results_renewableinvestmentsubisidies.npz") as loaded:
    renewable_investment_subsidies_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    renewable_investment_subsidies_results['expected_emissions'] = loaded['expected_emissions']
    renewable_investment_subsidies_results['expected_blackouts'] = loaded['expected_blackouts']
    renewable_investment_subsidies_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    renewable_investment_subsidies_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    renewable_investment_subsidies_results['expected_total_produced'] = loaded['expected_total_produced']
    renewable_investment_subsidies_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    renewable_investment_subsidies_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus'] + loaded['expected_dsp_profits']
    renewable_investment_subsidies_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    renewable_investment_subsidies_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    renewable_investment_subsidies_results['expected_revenue'] = loaded['expected_revenue']
    renewable_investment_subsidies_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    renewable_investment_subsidies_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum'] + loaded['expected_dsp_profits_sum']
    renewable_investment_subsidies_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    renewable_investment_subsidies_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    renewable_investment_subsidies_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    renewable_investment_subsidies_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    
delay_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results_delay_smoothed2.npz") as loaded:
    delay_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    delay_results['expected_emissions'] = loaded['expected_emissions']
    delay_results['expected_blackouts'] = loaded['expected_blackouts']
    delay_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    delay_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    delay_results['expected_total_produced'] = loaded['expected_total_produced']
    delay_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    delay_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus'] + loaded['expected_dsp_profits']
    delay_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    delay_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    delay_results['expected_revenue'] = loaded['expected_revenue']
    delay_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    delay_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum'] + loaded['expected_dsp_profits_sum']
    delay_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    delay_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    delay_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    delay_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    delay_results['expected_total_production_cost'] = loaded['expected_total_production_cost']
    delay_results['expected_consumer_surplus_sum_extra'] = loaded['expected_consumer_surplus_sum_extra'] + loaded['expected_dsp_profits_sum_extra']
    delay_results['expected_revenue_sum_extra'] = loaded['expected_revenue_sum_extra']
    delay_results['expected_emissions_sum_extra'] = loaded['expected_emissions_sum_extra']
    delay_results['expected_blackouts_sum_extra'] = loaded['expected_blackouts_sum_extra']
    delay_results['expected_producer_surplus_sum_extra'] = loaded['expected_producer_surplus_sum_extra']
    
high_price_cap_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results_highpricecap.npz") as loaded:
    high_price_cap_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    high_price_cap_results['expected_emissions'] = loaded['expected_emissions']
    high_price_cap_results['expected_blackouts'] = loaded['expected_blackouts']
    high_price_cap_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    high_price_cap_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    high_price_cap_results['expected_total_produced'] = loaded['expected_total_produced']
    high_price_cap_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    high_price_cap_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus'] + loaded['expected_dsp_profits']
    high_price_cap_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    high_price_cap_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    high_price_cap_results['expected_revenue'] = loaded['expected_revenue']
    high_price_cap_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    high_price_cap_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum'] + loaded['expected_dsp_profits_sum']
    high_price_cap_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    high_price_cap_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    high_price_cap_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    high_price_cap_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    high_price_cap_results['expected_total_production_cost'] = loaded['expected_total_production_cost']

battery_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results_battery.npz") as loaded:
    battery_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    battery_results['expected_emissions'] = loaded['expected_emissions']
    battery_results['expected_blackouts'] = loaded['expected_blackouts']
    battery_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    battery_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    battery_results['expected_total_produced'] = loaded['expected_total_produced']
    battery_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    battery_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus'] + loaded['expected_dsp_profits']
    battery_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    battery_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    battery_results['expected_revenue'] = loaded['expected_revenue']
    battery_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    battery_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum'] + loaded['expected_dsp_profits_sum']
    battery_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    battery_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    battery_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    battery_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    battery_results['expected_total_production_cost'] = loaded['expected_total_production_cost']
    battery_results['expected_battery_discharge'] = loaded['expected_battery_discharge']
    battery_results['expected_battery_profits'] = loaded['expected_battery_profits']

extended_results = {}
with np.load(f"{gv.arrays_path}extended_capacity_payments_results.npz") as loaded:
    extended_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    extended_results['expected_emissions'] = loaded['expected_emissions']
    extended_results['expected_blackouts'] = loaded['expected_blackouts']
    extended_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    extended_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    extended_results['expected_total_produced'] = loaded['expected_total_produced']
    extended_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    extended_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus'] + loaded['expected_dsp_profits']
    extended_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    extended_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    extended_results['expected_revenue'] = loaded['expected_revenue']
    extended_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    extended_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum'] + loaded['expected_dsp_profits_sum']
    extended_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    extended_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    extended_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    extended_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    extended_results['expected_total_production_cost'] = loaded['expected_total_production_cost']

competitive_results = {}
results_idx_arange = np.arange(carbon_taxes_linspace.shape[0] * capacity_payments_linspace.shape[0])
competitive_results['expected_agg_source_capacity'] = np.zeros(battery_results['expected_agg_source_capacity'].shape)
competitive_results['expected_emissions'] = np.zeros(battery_results['expected_emissions'].shape)
competitive_results['expected_blackouts'] = np.zeros(battery_results['expected_blackouts'].shape)
competitive_results['expected_frac_by_source'] = np.zeros(battery_results['expected_frac_by_source'].shape)
competitive_results['expected_quantity_weighted_avg_price'] = np.zeros(battery_results['expected_quantity_weighted_avg_price'].shape)
competitive_results['expected_total_produced'] = np.zeros(battery_results['expected_total_produced'].shape)
competitive_results['expected_misallocated_demand'] = np.zeros(battery_results['expected_misallocated_demand'].shape)
competitive_results['expected_consumer_surplus'] = np.zeros(battery_results['expected_consumer_surplus'].shape)
competitive_results['expected_carbon_tax_revenue'] = np.zeros(battery_results['expected_carbon_tax_revenue'].shape)
competitive_results['expected_capacity_payments'] = np.zeros(battery_results['expected_capacity_payments'].shape)
competitive_results['expected_revenue'] = np.zeros(battery_results['expected_revenue'].shape)
competitive_results['expected_producer_surplus_sum'] = np.zeros(battery_results['expected_producer_surplus_sum'].shape)
competitive_results['expected_consumer_surplus_sum'] = np.zeros(battery_results['expected_consumer_surplus_sum'].shape)
competitive_results['expected_revenue_sum'] = np.zeros(battery_results['expected_revenue_sum'].shape)
competitive_results['expected_product_market_sum'] = np.zeros(battery_results['expected_product_market_sum'].shape)
competitive_results['expected_emissions_sum'] = np.zeros(battery_results['expected_emissions_sum'].shape)
competitive_results['expected_blackouts_sum'] = np.zeros(battery_results['expected_blackouts_sum'].shape)
competitive_results['expected_total_production_cost'] = np.zeros(battery_results['expected_total_production_cost'].shape)

for i in range(3):
    start_idx = (i * results_idx_arange.shape[0]) // 3
    end_idx = ((i + 1) * results_idx_arange.shape[0]) // 3
    start_idx_load = 0
    end_idx_load = start_idx_load + (end_idx - start_idx)
    with np.load(f"{gv.arrays_path}counterfactual_results_competitive_{i}.npz") as loaded:
        num_years_size = loaded['expected_agg_source_capacity'].shape[2]
        energy_sources_size = loaded['expected_agg_source_capacity'].shape[3]
        competitive_results['expected_agg_source_capacity'].flat[(start_idx*(num_years_size*energy_sources_size)):(end_idx*(num_years_size*energy_sources_size))] = loaded['expected_agg_source_capacity'].flat[(start_idx_load*(num_years_size*energy_sources_size)):(end_idx_load*(num_years_size*energy_sources_size))]
        competitive_results['expected_emissions'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_emissions'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_blackouts'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_blackouts'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_frac_by_source'].flat[(start_idx*(num_years_size*energy_sources_size)):(end_idx*(num_years_size*energy_sources_size))] = loaded['expected_frac_by_source'].flat[(start_idx_load*(num_years_size*energy_sources_size)):(end_idx_load*(num_years_size*energy_sources_size))]
        competitive_results['expected_quantity_weighted_avg_price'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_quantity_weighted_avg_price'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_total_produced'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_total_produced'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_misallocated_demand'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_misallocated_demand'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_consumer_surplus'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_consumer_surplus'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))] + loaded['expected_dsp_profits'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_carbon_tax_revenue'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_carbon_tax_revenue'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_capacity_payments'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_capacity_payments'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_revenue'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_revenue'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]
        competitive_results['expected_producer_surplus_sum'].flat[start_idx:end_idx] = loaded['expected_producer_surplus_sum'].flat[start_idx_load:end_idx_load]
        competitive_results['expected_consumer_surplus_sum'].flat[start_idx:end_idx] = loaded['expected_consumer_surplus_sum'].flat[start_idx_load:end_idx_load] + loaded['expected_dsp_profits_sum'].flat[start_idx_load:end_idx_load]
        competitive_results['expected_revenue_sum'].flat[start_idx:end_idx] = loaded['expected_revenue_sum'].flat[start_idx_load:end_idx_load]
        competitive_results['expected_product_market_sum'].flat[start_idx:end_idx] = loaded['expected_product_market_sum'].flat[start_idx_load:end_idx_load]
        competitive_results['expected_emissions_sum'].flat[start_idx:end_idx] = loaded['expected_emissions_sum'].flat[start_idx_load:end_idx_load]
        competitive_results['expected_blackouts_sum'].flat[start_idx:end_idx] = loaded['expected_blackouts_sum'].flat[start_idx_load:end_idx_load]
        competitive_results['expected_total_production_cost'].flat[(start_idx*(num_years_size)):(end_idx*(num_years_size))] = loaded['expected_total_production_cost'].flat[(start_idx_load*(num_years_size)):(end_idx_load*(num_years_size))]

cap_pay_spot_price_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results_capacitypaymentspotprice.npz") as loaded:
    cap_pay_spot_price_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    cap_pay_spot_price_results['expected_emissions'] = loaded['expected_emissions']
    cap_pay_spot_price_results['expected_blackouts'] = loaded['expected_blackouts']
    cap_pay_spot_price_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    cap_pay_spot_price_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    cap_pay_spot_price_results['expected_total_produced'] = loaded['expected_total_produced']
    cap_pay_spot_price_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    cap_pay_spot_price_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus'] + loaded['expected_dsp_profits']
    cap_pay_spot_price_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    cap_pay_spot_price_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    cap_pay_spot_price_results['expected_revenue'] = loaded['expected_revenue']
    cap_pay_spot_price_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    cap_pay_spot_price_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum'] + loaded['expected_dsp_profits_sum']
    cap_pay_spot_price_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    cap_pay_spot_price_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    cap_pay_spot_price_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    cap_pay_spot_price_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    cap_pay_spot_price_results['expected_total_production_cost'] = loaded['expected_total_production_cost']
    cap_pay_spot_price_results['expected_inv_quantity_weighted_avg_price'] = loaded['expected_inv_quantity_weighted_avg_price']

# %%
# Create functions for graphs

# Graph arrays
with np.load(f"{gv.arrays_path}state_space.npz") as loaded:
    sources = loaded['energy_sources_unique']
    years = loaded['years_unique']
num_years = np.load(f"{gv.arrays_path}num_years_investment.npy")[0]
years = np.concatenate((years, np.arange(years[-1] + 1, years[0] + num_years)))

# Rename sources:
source_names = np.copy(sources)
for s, source in enumerate(sources):
    if source == gv.coal:
        source_names[s] = "coal"
    if source == gv.gas_ocgt:
        source_names[s] = "gas"
    if source == gv.wind:
        source_names[s] = "wind"
    if source == gv.solar:
        source_names[s] = "solar"
source_names = source_names[source_names != gv.gas_ccgt]

# Graph properties
lw_paper = 3.0
lw_presentation = 4.0
fontsize_paper = 14
fontsize_presentation = 16
title_fontsize_paper = 17
title_fontsize_presentation = 20
ls_all = ["solid", "solid", "solid", "solid"] #["solid", "dotted", "dashed", "dashdot"]
years_begin = 2006
years_end = 2030
years_use = np.arange(years_begin, years_end + 1)

# Combine
def combine_gas(arr, axis):
    """Combine the natural gas categories into one element"""
    arr_ndims = arr.ndim
    arr_ = np.copy(arr)
    gas_idx = np.isin(sources, gv.natural_gas)
    new_gas_idx = np.where(sources == gv.gas_ocgt)[0][0]
    arr_[tuple([slice(None)] * axis + [new_gas_idx] + [slice(None)] * (arr_ndims - axis - 1))] = np.sum(arr_[tuple([slice(None)] * axis + [gas_idx] + [slice(None)] * (arr_ndims - axis - 1))], axis=axis)
    return_indices = ~gas_idx
    return_indices[new_gas_idx] = True
    return arr_[tuple([slice(None)] * axis + [return_indices] + [slice(None)] * (arr_ndims - axis - 1))]

def select_years(arr, axis):
    """Use only certain years"""
    arr_ndims = arr.ndim
    years_begin_idx = np.where(years == years_begin)[0][0]
    years_end_idx = np.where(years == years_end)[0][0]
    return arr[tuple([slice(None)] * axis + [slice(years_begin_idx, years_end_idx + 1)] + [slice(None)] * (arr_ndims - axis - 1))]

def ton_to_kg(arr):
    """Convert from tons to kg"""
    return arr * 1000.0

def kg_to_ton(arr):
    """Convert from kg to tons"""
    return arr / 1000.0

def plot_capacities(capacities, indices_use, labels, ls, colors, lw, fontsize, title_fontsize, filename):
    fig, axs = plt.subplots(nrows=1, ncols=source_names.shape[0], figsize=(5.0*source_names.shape[0],5.0*1), squeeze=False)
    
    # Plot capacity of each source
    for s, source in enumerate(source_names):
        for i, idx in enumerate(indices_use):
            axs[0,s].plot(years_use, select_years(combine_gas(capacities, 2), 1)[idx,:,s], color=colors[i], label=labels[i] if s == 0 else None, lw=lw, ls=ls[i])
        axs[0,s].set_title(f"{source_names[s]}", fontsize=title_fontsize)
        axs[0,s].set_ylabel("MW", fontsize=fontsize)
        
    # Set limits on capacity
    max_ = np.max(combine_gas(capacities, 2)[indices_use,:,:])
    min_ = 0.0 # np.min(capacities[indices_use,:,:])
    diff_ = max_ - min_
    for s, source in enumerate(source_names):
        axs[0,s].set_ylim((min_ - 0.1 * diff_, max_ + 0.1 * diff_))
    
    # Create figure legend
    if not np.all(np.array(labels) == None):
        fig.legend(loc="lower center", ncol=indices_use.shape[0], fontsize=fontsize)
        fig.subplots_adjust(bottom=0.2)
        
    plt.subplots_adjust(wspace=0.3)
        
    # Save and show plot
    if (filename is not None) and save_output:
        plt.savefig(filename, transparent=True)
    if show_output:
        plt.show()
        
def plot_production(production, indices_use, labels, ls, colors, lw, fontsize, title_fontsize, filename):
    fig, axs = plt.subplots(nrows=1, ncols=source_names.shape[0], figsize=(5.0*source_names.shape[0],5.0*1), squeeze=False)
    
    # Plot capacity of each source
    for s, source in enumerate(source_names):
        for i, idx in enumerate(indices_use):
            axs[0,s].plot(years_use, select_years(combine_gas(production, 2), 1)[idx,:,s], color=colors[i], label=labels[i] if s == 0 else None, lw=lw, ls=ls[i])
        axs[0,s].set_title(f"{source_names[s]}", fontsize=title_fontsize)
        
    # Set limits on production share
    max_ = np.max(combine_gas(production, 2)[indices_use,:,:])
    min_ = 0.0 # np.min(production[indices_use,:,:])
    diff_ = max_ - min_
    for s, source in enumerate(source_names):
        axs[0,s].set_ylim((min_ - 0.1 * diff_, max_ + 0.1 * diff_))
    
    # Create figure legend
    if not np.all(np.array(labels) == None):
        fig.legend(loc="lower center", ncol=indices_use.shape[0], fontsize=fontsize)
        fig.subplots_adjust(bottom=0.2)
        
    # Save and show plot
    if (filename is not None) and save_output:
        plt.savefig(filename, transparent=True)
    if show_output:
        plt.show()
        
def plot_combined_capacities_production(capacities, production, indices_use, labels, ls, colors, lw, fontsize, title_fontsize, filename):
    fig, axs = plt.subplots(nrows=2, ncols=source_names.shape[0], figsize=(5.0*source_names.shape[0],5.0*2), squeeze=False)
    
    # Plot capacity of each source
    for s, source in enumerate(source_names):
        for i, idx in enumerate(indices_use):
            axs[0,s].plot(years_use, select_years(combine_gas(capacities, 2), 1)[idx,:,s], color=colors[i], label=labels[i] if s == 0 else None, lw=lw, ls=ls[i])
            axs[1,s].plot(years_use, select_years(combine_gas(production, 2), 1)[idx,:,s], color=colors[i], lw=lw, ls=ls[i])
        axs[0,s].set_title(f"{source_names[s]} capacity", fontsize=title_fontsize)
        axs[1,s].set_title(f"share produced by {source_names[s]}", fontsize=title_fontsize)
        axs[0,s].set_ylabel("MW", fontsize=fontsize)
        
    # Set limits on capacity and production shares
    max_ = np.max(combine_gas(capacities, 2)[indices_use,:,:])
    min_ = 0.0 # np.min(capacities[indices_use,:,:])
    diff_ = max_ - min_
    for s, source in enumerate(source_names):
        axs[0,s].set_ylim((min_ - 0.1 * diff_, max_ + 0.1 * diff_))
    max_ = np.max(combine_gas(production, 2)[indices_use,:,:])
    min_ = 0.0 # np.min(capacities[indices_use,:,:])
    diff_ = max_ - min_
    for s, source in enumerate(source_names):
        axs[1,s].set_ylim((min_ - 0.1 * diff_, max_ + 0.1 * diff_))
    
    # Create figure legend
    if not np.all(np.array(labels) == None):
        fig.legend(loc="lower center", ncol=indices_use.shape[0], fontsize=fontsize)
        fig.subplots_adjust(bottom=0.11)
        
    plt.subplots_adjust(wspace=0.3)
        
    # Save and show plot
    if (filename is not None) and save_output:
        plt.savefig(filename, transparent=True)
    if show_output:
        plt.show()
        
def plot_welfare(cspsg, emissions, blackouts, x_axis_linspace, x_axis_label, y_axis_label, ls, colors, lw, fontsize, title_fontsize, filename, labels=None):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(5.0*3,5.0*1), squeeze=False)
    
    multiple_lines = cspsg.ndim > 1
    
    x_axis_linspace_fine = np.linspace(np.min(x_axis_linspace), np.max(x_axis_linspace), 1000)
    
    # Plot CS + PS + G
    if multiple_lines:
        for i in range(cspsg.shape[1]):
            axs[0,0].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, cspsg[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i], label=labels[i])
    else:
        axs[0,0].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, cspsg)(x_axis_linspace_fine), color=colors[0], lw=lw, ls=ls[0])
    axs[0,0].set_xlabel(f"{x_axis_label[0]}", fontsize=fontsize)
    axs[0,0].set_ylabel(f"{y_axis_label[0]}", fontsize=fontsize)
    axs[0,0].set_title(f"$CS + PS + G$", fontsize=title_fontsize)
        
    # Plot emissions
    if multiple_lines:
        for i in range(emissions.shape[1]):
            axs[0,1].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, emissions[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i])
    else:
        axs[0,1].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, emissions)(x_axis_linspace_fine), color=colors[1], lw=lw, ls=ls[1])
    axs[0,1].set_xlabel(f"{x_axis_label[1]}", fontsize=fontsize)
    axs[0,1].set_ylabel(f"{y_axis_label[1]}", fontsize=fontsize)
    axs[0,1].set_title(f"emissions", fontsize=title_fontsize)
    
    # Plot blackouts
    if multiple_lines:
        for i in range(blackouts.shape[1]):
            axs[0,2].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, blackouts[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i])
    else:
        axs[0,2].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, blackouts)(x_axis_linspace_fine), color=colors[2], lw=lw, ls=ls[2])
    axs[0,2].set_xlabel(f"{x_axis_label[2]}", fontsize=fontsize)
    axs[0,2].set_ylabel(f"{y_axis_label[2]}", fontsize=fontsize)
    axs[0,2].set_title(f"blackouts", fontsize=title_fontsize)
    
    # Create figure legend
    if multiple_lines:
        fig.legend(loc="lower center", ncol=cspsg.shape[1], fontsize=fontsize)
        fig.subplots_adjust(bottom=0.225)
        
    # Save and show plot
    if (filename is not None) and save_output:
        plt.savefig(filename, transparent=True)
    if show_output:
        plt.show()
        
def plot_limited_welfare(cspsg, emissions, blackouts, x_axis_linspace, x_axis_label, y_axis_label, ls, colors, lw, fontsize, title_fontsize, filename, labels=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5.0*2,5.0*1), squeeze=False)
    
    multiple_lines = cspsg.ndim > 1
    
    x_axis_linspace_fine = np.linspace(np.min(x_axis_linspace), np.max(x_axis_linspace), 1000)
        
    # Plot emissions
    if multiple_lines:
        for i in range(emissions.shape[1]):
            axs[0,0].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, emissions[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i], label=labels[i])
    else:
        axs[0,0].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, emissions)(x_axis_linspace_fine), color=colors[1], lw=lw, ls=ls[1])
    axs[0,0].set_xlabel(f"{x_axis_label[1]}", fontsize=fontsize)
    axs[0,0].set_ylabel(f"{y_axis_label[1]}", fontsize=fontsize)
    axs[0,0].set_title(f"emissions", fontsize=title_fontsize)
    
    # Plot blackouts
    if multiple_lines:
        for i in range(blackouts.shape[1]):
            axs[0,1].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, blackouts[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i])
    else:
        axs[0,1].plot(x_axis_linspace_fine, interp.Akima1DInterpolator(x_axis_linspace, blackouts)(x_axis_linspace_fine), color=colors[2], lw=lw, ls=ls[2])
    axs[0,1].set_xlabel(f"{x_axis_label[2]}", fontsize=fontsize)
    axs[0,1].set_ylabel(f"{y_axis_label[2]}", fontsize=fontsize)
    axs[0,1].set_title(f"blackouts", fontsize=title_fontsize)
    
    # Create figure legend
    if multiple_lines:
        fig.legend(loc="lower center", ncol=cspsg.shape[1], fontsize=fontsize)
        fig.subplots_adjust(bottom=0.225)
        
    # Save and show plot
    if (filename is not None) and save_output:
        plt.savefig(filename, transparent=True)
    if show_output:
        plt.show()
        
def plot_general_over_time(var_arr, indices_use, labels, ls, colors, lw, fontsize, title_fontsize, titles, x_labels, y_labels, filename):
    num_vars = var_arr.shape[0]
    num_lines = var_arr.shape[1]
    fig, axs = plt.subplots(nrows=1, ncols=num_vars, figsize=(5.0*num_vars,5.0*1), squeeze=False)
    
    # Plot each variable of each source
    for i in range(num_vars):
        for j in range(num_lines):
            axs[0,i].plot(years_use, select_years(var_arr[i,j,:], 0), color=colors[j], label=labels[j] if i == 0 else None, lw=lw, ls=ls[j])
        axs[0,i].set_title(f"{titles[i]}", fontsize=title_fontsize)
        axs[0,i].set_xlabel(f"{x_labels[i]}", fontsize=fontsize)
        axs[0,i].set_ylabel(f"{y_labels[i]}", fontsize=fontsize)
    
    # Create figure legend
    if not np.all(np.array(labels) == None):
        fig.legend(loc="lower center", ncol=num_lines, fontsize=fontsize)
        fig.subplots_adjust(bottom=0.2)
        
    # Save and show plot
    if (filename is not None) and save_output:
        plt.savefig(filename, transparent=True)
    if show_output:
        plt.show()

# %%
# Carbon tax counterfactuals

# Capacity (paper version)
tax_idx = np.array([0, 2, 4, 6])#np.array([0, 3, 6, 9])
cap_pay_idx = np.array([0, 0, 0, 0])
indices_use = np.ravel_multi_index(np.vstack((tax_idx, cap_pay_idx)), (carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
labels = [f"$\\tau = $A$\\${int(tau)}$/ton" for tau in carbon_taxes_linspace[tax_idx]]
ls = [ls_all[i] for i in range(tax_idx.shape[0])]
#colors = ["black" for tau in carbon_taxes_linspace[tax_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_carbon_tax.pdf"
plot_capacities(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
ls = [ls_all[0] for i in range(tax_idx.shape[0])]
cmap = cm.get_cmap("Blues")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_carbon_tax_presentation.pdf"
plot_capacities(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
ls = [ls_all[i] for i in range(tax_idx.shape[0])]
#colors = ["black" for tau in carbon_taxes_linspace[tax_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}production_carbon_tax.pdf"
plot_production(np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
ls = [ls_all[0] for i in range(tax_idx.shape[0])]
cmap = cm.get_cmap("Blues")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}production_carbon_tax_presentation.pdf"
plot_production(np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
ls = [ls_all[i] for i in range(tax_idx.shape[0])]
#colors = ["black" for tau in carbon_taxes_linspace[tax_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_carbon_tax.pdf"
plot_combined_capacities_production(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
ls = [ls_all[0] for i in range(tax_idx.shape[0])]
cmap = cm.get_cmap("Blues")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_carbon_tax_presentation.pdf"
plot_combined_capacities_production(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
tax_idx = np.arange(carbon_taxes_linspace.shape[0])
cap_pay_idx = np.zeros(tax_idx.shape, dtype=int)
indices_use = np.ravel_multi_index(np.vstack((tax_idx, cap_pay_idx)), (carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
cspsg = np.reshape(carbon_tax_capacity_payment_results['expected_product_market_sum'], (-1,))[indices_use] / 1000000000.0
emissions = ton_to_kg(np.reshape(carbon_tax_capacity_payment_results['expected_emissions_sum'], (-1,))[indices_use]) / 1000000000.0
blackouts = np.reshape(carbon_tax_capacity_payment_results['expected_blackouts_sum'], (-1,))[indices_use] / 1000000.0
ls = ["solid" for i in range(3)]
colors = ["black" for i in range(3)]
x_axis_labels = [f"$\\tau$ (A\\$/ton)" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
filename = f"{gv.graphs_path}welfare_carbon_tax.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_carbon_tax_presentation.pdf"
colors = [cm.get_cmap("Blues")(0.75) for i in range(3)]
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_carbon_tax_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# %%
# Capacity payment counterfactuals

# Capacity (paper version)
tax_idx = np.array([0, 0, 0, 0])
cap_pay_idx = np.array([0, 1, 2, 3])
indices_use = np.ravel_multi_index(np.vstack((tax_idx, cap_pay_idx)), (carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
labels = [f"$\\kappa = $A$\\${int(kappa):,}$/MW".replace(",", "\\,") for kappa in capacity_payments_linspace[cap_pay_idx]]
ls = [ls_all[i] for i in range(cap_pay_idx.shape[0])]
#colors = ["black" for kappa in capacity_payments_linspace[cap_pay_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, cap_pay_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_capacity_payment.pdf"
plot_capacities(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
ls = [ls_all[0] for i in range(cap_pay_idx.shape[0])]
cmap = cm.get_cmap("Reds")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, cap_pay_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_capacity_payment_presentation.pdf"
plot_capacities(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
ls = [ls_all[i] for i in range(cap_pay_idx.shape[0])]
#colors = ["black" for kappa in capacity_payments_linspace[cap_pay_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, cap_pay_idx.shape[0])]
filename = f"{gv.graphs_path}production_capacity_payment.pdf"
plot_production(np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
ls = [ls_all[0] for i in range(cap_pay_idx.shape[0])]
cmap = cm.get_cmap("Reds")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, cap_pay_idx.shape[0])]
filename = f"{gv.graphs_path}production_capacity_payment_presentation.pdf"
plot_production(np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
ls = [ls_all[i] for i in range(cap_pay_idx.shape[0])]
#colors = ["black" for kappa in capacity_payments_linspace[cap_pay_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, cap_pay_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_capacity_payment.pdf"
plot_combined_capacities_production(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
ls = [ls_all[0] for i in range(cap_pay_idx.shape[0])]
cmap = cm.get_cmap("Reds")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, cap_pay_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_capacity_payment_presentation.pdf"
plot_combined_capacities_production(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
tax_idx = np.zeros(capacity_payments_linspace.shape, dtype=int)
cap_pay_idx = np.arange(tax_idx.shape[0])
indices_use = np.ravel_multi_index(np.vstack((tax_idx, cap_pay_idx)), (carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
cspsg = np.reshape(carbon_tax_capacity_payment_results['expected_product_market_sum'], (-1,))[indices_use] / 1000000000.0
emissions = ton_to_kg(np.reshape(carbon_tax_capacity_payment_results['expected_emissions_sum'], (-1,))[indices_use]) / 1000000000.0
blackouts = np.reshape(carbon_tax_capacity_payment_results['expected_blackouts_sum'], (-1,))[indices_use] / 1000000.0
ls = ["solid" for i in range(3)]
colors = ["black" for i in range(3)]
x_axis_labels = [f"$\\kappa$ (A\\$/MW)" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (millions)", f"MWh (millions)"]
filename = f"{gv.graphs_path}welfare_capacity_payment.pdf"
plot_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_capacity_payment_presentation.pdf"
colors = [cm.get_cmap("Reds")(0.75) for i in range(3)]
plot_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_capacity_payment_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# %%
# Carbon tax + capacity payment counterfactuals

# Capacity (paper version)
tax_idx = np.array([0, 0, 3, 3])
cap_pay_idx = np.array([0, 3, 0, 3])
indices_use = np.ravel_multi_index(np.vstack((tax_idx, cap_pay_idx)), (carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
labels = [f"$\\tau = $A$\\${int(carbon_taxes_linspace[tax_idx[i]])}$, " + f"$\\kappa = $A$\\${int(capacity_payments_linspace[cap_pay_idx[i]]):,}$".replace(",", "\\,") for i in range(tax_idx.shape[0])]
def combined_color_scheme_paper(tax_idx_i, cap_pay_idx_i):
    color_scheme = "Greys"
    if (tax_idx_i == np.min(tax_idx)) and (cap_pay_idx_i == np.min(cap_pay_idx)):
        val = 0.45
    if (tax_idx_i == np.max(tax_idx)) and (cap_pay_idx_i == np.max(cap_pay_idx)):
        val = 1.0
    if (tax_idx_i == np.max(tax_idx)) and (cap_pay_idx_i == np.min(cap_pay_idx)):
        val = 1.0
    if (tax_idx_i == np.min(tax_idx)) and (cap_pay_idx_i == np.max(cap_pay_idx)):
        val = 0.45
    return cm.get_cmap(color_scheme)(val)
ls = ["dashed" if cap_pay_idx[i] == np.max(cap_pay_idx) else "solid" for i in range(tax_idx.shape[0])] #[ls_all[i] for i in range(tax_idx.shape[0])]
colors = [combined_color_scheme_paper(tax_idx[i], cap_pay_idx[i]) for i in range(tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_carbon_tax_capacity_payment.pdf"
plot_capacities(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
def combined_color_scheme_presentation(tax_idx_i, cap_pay_idx_i):
    if (tax_idx_i == np.min(tax_idx)) and (cap_pay_idx_i == np.min(cap_pay_idx)):
        color_scheme = "Purples"
        val = 0.33
    if (tax_idx_i == np.max(tax_idx)) and (cap_pay_idx_i == np.max(cap_pay_idx)):
        color_scheme = "Purples"
        val = 0.75
    if (tax_idx_i == np.max(tax_idx)) and (cap_pay_idx_i == np.min(cap_pay_idx)):
        color_scheme = "Blues"
        val = 0.75
    if (tax_idx_i == np.min(tax_idx)) and (cap_pay_idx_i == np.max(cap_pay_idx)):
        color_scheme = "Reds"
        val = 0.75
    return cm.get_cmap(color_scheme)(val)
ls = [ls_all[0] for i in range(cap_pay_idx.shape[0])]
colors = [combined_color_scheme_presentation(tax_idx[i], cap_pay_idx[i]) for i in range(tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_carbon_tax_capacity_payment_presentation.pdf"
plot_capacities(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
ls = ["dashed" if cap_pay_idx[i] == np.max(cap_pay_idx) else "solid" for i in range(tax_idx.shape[0])] #[ls_all[i] for i in range(tax_idx.shape[0])]
colors = [combined_color_scheme_paper(tax_idx[i], cap_pay_idx[i]) for i in range(tax_idx.shape[0])]
filename = f"{gv.graphs_path}production_carbon_tax_capacity_payment.pdf"
plot_production(np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
ls = [ls_all[0] for i in range(tax_idx.shape[0])]
cmap = cm.get_cmap("Reds")
colors = [combined_color_scheme_presentation(tax_idx[i], cap_pay_idx[i]) for i in range(tax_idx.shape[0])]
filename = f"{gv.graphs_path}production_carbon_tax_capacity_payment_presentation.pdf"
plot_production(np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
ls = ["dashed" if cap_pay_idx[i] == np.max(cap_pay_idx) else "solid" for i in range(tax_idx.shape[0])] #[ls_all[i] for i in range(tax_idx.shape[0])]
colors = [combined_color_scheme_paper(tax_idx[i], cap_pay_idx[i]) for i in range(tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_carbon_tax_capacity_payment.pdf"
plot_combined_capacities_production(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
ls = [ls_all[0] for i in range(tax_idx.shape[0])]
colors = [combined_color_scheme_presentation(tax_idx[i], cap_pay_idx[i]) for i in range(tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_carbon_tax_capacity_payment_presentation.pdf"
plot_combined_capacities_production(np.reshape(carbon_tax_capacity_payment_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), np.reshape(carbon_tax_capacity_payment_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# %%
# Capacity payments + carbon tax graph for presentation

# Welfare (paper version)
tax_idx = np.tile(np.arange(carbon_taxes_linspace.shape[0])[np.newaxis,:], (2,1))
cap_pay_idx = np.vstack((np.zeros(tax_idx.shape[1], dtype=int), np.ones(tax_idx[1].shape, dtype=int) * 3))
indices_use = np.ravel_multi_index(np.vstack((tax_idx.flatten(), cap_pay_idx.flatten())), (carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0])).reshape(2,-1)
cspsg = np.reshape(carbon_tax_capacity_payment_results['expected_product_market_sum'], (-1,))[indices_use] / 1000000000.0
emissions = ton_to_kg(np.reshape(carbon_tax_capacity_payment_results['expected_emissions_sum'], (-1,))[indices_use]) / 1000000000.0
blackouts = np.reshape(carbon_tax_capacity_payment_results['expected_blackouts_sum'], (-1,))[indices_use] / 1000000.0
cspsg = cspsg.T
emissions = emissions.T
blackouts = blackouts.T
x_axis_labels = [f"$\\tau$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"$\\kappa = $A\\$$0$ / MW", f"$\\kappa = $A\\$$150$ / MW"]
colors = ["black", "black"]
ls = ["solid", "dashed"]
filename = f"{gv.graphs_path}welfare_carbon_tax_capacity_payment.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename, labels=labels)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_carbon_tax_capacity_payment_presentation.pdf"
colors = [cm.get_cmap("Blues")(0.75), cm.get_cmap("Purples")(0.9)]
ls = ["solid", "solid"]
filename = f"{gv.graphs_path}welfare_price_cap_capacity_payment_presentation.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_carbon_tax_capacity_payment_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# %%
# High price cap counterfactuals

# Capacity (paper version)
tax_idx_low = 0
tax_idx_high = 6
cap_pay_idx = 0
labels = [f"$\\bar{{P}} = $A\\$$300$ / MWh, $\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_low]))}$ / ton", f"$\\bar{{P}} = $A\\$$1\\,000$ / MWh, $\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_low]))}$", f"$\\bar{{P}} = $A\\$$300$ / MWh, $\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_high]))}$", f"$\\bar{{P}} = $A\\$$1\\,000$ / MWh, $\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_high]))}$"]
ls = [f"solid", f"dashed", f"solid", f"dashed"]
colors = [cm.get_cmap("Greys")(0.45), cm.get_cmap("Greys")(0.45), "black", "black"]
filename = f"{gv.graphs_path}capacity_price_cap.pdf"
capacities = np.concatenate((carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], high_price_cap_results['expected_agg_source_capacity'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:], high_price_cap_results['expected_agg_source_capacity'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:]), axis=0)
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
filename = f"{gv.graphs_path}capacity_price_cap_presentation.pdf"
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
filename = f"{gv.graphs_path}production_price_cap.pdf"
production = np.concatenate((carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], high_price_cap_results['expected_frac_by_source'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:], high_price_cap_results['expected_frac_by_source'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:]), axis=0)
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
filename = f"{gv.graphs_path}production_price_cap_presentation.pdf"
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
filename = f"{gv.graphs_path}capacity_production_price_cap.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
filename = f"{gv.graphs_path}capacity_production_price_cap_presentation.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
cspsg = np.concatenate((carbon_tax_capacity_payment_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis], high_price_cap_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1) / 1000000000.0
emissions = ton_to_kg(np.concatenate((carbon_tax_capacity_payment_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis], high_price_cap_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1)) / 1000000000.0
blackouts = np.concatenate((carbon_tax_capacity_payment_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis], high_price_cap_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1) / 1000000.0
x_axis_labels = [f"$\\tau$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"$\\bar{{P}} = $A\\$$300$ / MWh", f"$\\bar{{P}} = $A\\$$1\\,000$ / MWh"]
colors = ["black", "black"]
filename = f"{gv.graphs_path}welfare_price_cap_carbon_tax.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename, labels=labels)

# Welfare (presentation version)
colors = [cm.get_cmap("Blues")(0.75) for i in range(2)]
filename = f"{gv.graphs_path}welfare_price_cap_carbon_tax_presentation.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_price_cap_carbon_tax_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Capacity (paper version)
tax_idx = 0
cap_pay_idx_low = 0
cap_pay_idx_high = 4
labels = [f"$\\bar{{P}} = $A\\$$300$ / MWh, $\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_low]))}$ / ton", f"$\\bar{{P}} = $A\\$$1\\,000$ / MWh, $\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_low]))}$", f"$\\bar{{P}} = $A\\$$300$ / MWh, $\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_high]))}$", f"$\\bar{{P}} = $A\\$$1\\,000$ / MWh, $\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_high]))}$"]
ls = [f"solid", f"dashed", f"solid", f"dashed"]
colors = [cm.get_cmap("Greys")(0.45), cm.get_cmap("Greys")(0.45), "black", "black"]
filename = f"{gv.graphs_path}capacity_price_capacity_payment_cap.pdf"
capacities = np.concatenate((carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], high_price_cap_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:], high_price_cap_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:]), axis=0)
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
filename = f"{gv.graphs_path}capacity_price_cap_capacity_payment_presentation.pdf"
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
filename = f"{gv.graphs_path}production_price_capacity_payment_cap.pdf"
production = np.concatenate((carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], high_price_cap_results['expected_frac_by_source'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:], high_price_cap_results['expected_frac_by_source'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:]), axis=0)
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
filename = f"{gv.graphs_path}production_price_cap_capacity_payment_presentation.pdf"
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
filename = f"{gv.graphs_path}capacity_production_price_cap_capacity_payment.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
filename = f"{gv.graphs_path}capacity_production_price_cap_capacity_payment_presentation.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
tax_idx = 0
cspsg = np.concatenate((carbon_tax_capacity_payment_results['expected_product_market_sum'][tax_idx,:][:,np.newaxis], high_price_cap_results['expected_product_market_sum'][tax_idx,:][:,np.newaxis]), axis=1) / 1000000000.0
emissions = ton_to_kg(np.concatenate((carbon_tax_capacity_payment_results['expected_emissions_sum'][tax_idx,:][:,np.newaxis], high_price_cap_results['expected_emissions_sum'][tax_idx,:][:,np.newaxis]), axis=1)) / 1000000000.0
blackouts = np.concatenate((carbon_tax_capacity_payment_results['expected_blackouts_sum'][tax_idx,:][:,np.newaxis], high_price_cap_results['expected_blackouts_sum'][tax_idx,:][:,np.newaxis]), axis=1) / 1000000.0
x_axis_labels = [f"$\\kappa$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"$\\bar{{P}} = $A\\$$300$ / MWh", f"$\\bar{{P}} = $A\\$$1\\,000$ / MWh"]
colors = ["black", "black"]
filename = f"{gv.graphs_path}welfare_price_cap_capacity_payment.pdf"
plot_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename, labels=labels)

# Welfare (presentation version)
colors = [cm.get_cmap("Reds")(0.75) for i in range(2)]
filename = f"{gv.graphs_path}welfare_price_cap_capacity_payment_presentation.pdf"
plot_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_price_cap_capacity_payment_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# %%
# Joint policies table

# 
beta = 0.95
beta_power = beta**np.arange(carbon_tax_capacity_payment_results['expected_carbon_tax_revenue'].shape[2] - 1)
beta_repeated = beta**(carbon_tax_capacity_payment_results['expected_carbon_tax_revenue'].shape[2] - 1) / (1.0 - beta)
expected_carbon_tax_revenue_sum = np.sum(beta_power[np.newaxis,np.newaxis,:] * carbon_tax_capacity_payment_results['expected_carbon_tax_revenue'][:,:,:-1], axis=2) + (beta_repeated * carbon_tax_capacity_payment_results['expected_carbon_tax_revenue'][:,:,-1])
expected_carbon_tax_revenue_highpricecap_sum = np.sum(beta_power[np.newaxis,np.newaxis,:] * high_price_cap_results['expected_carbon_tax_revenue'][:,:,:-1], axis=2) + (beta_repeated * high_price_cap_results['expected_carbon_tax_revenue'][:,:,-1])
expected_carbon_tax_revenue_extended_sum = np.sum(beta_power[np.newaxis,np.newaxis,np.newaxis,:] * extended_results['expected_carbon_tax_revenue'][:,:,:,:-1], axis=3) + (beta_repeated * extended_results['expected_carbon_tax_revenue'][:,:,:,-1])
expected_carbon_tax_revenue_spot_sum = np.sum(beta_power[np.newaxis,np.newaxis,:] * cap_pay_spot_price_results['expected_carbon_tax_revenue'][:,:,:-1], axis=2) + (beta_repeated * cap_pay_spot_price_results['expected_carbon_tax_revenue'][:,:,-1])
carbon_tax_capacity_payment_results['expected_carbon_tax_revenue_sum'] = expected_carbon_tax_revenue_sum
high_price_cap_results['expected_carbon_tax_revenue_sum'] = expected_carbon_tax_revenue_highpricecap_sum
extended_results['expected_carbon_tax_revenue_sum'] = expected_carbon_tax_revenue_extended_sum
cap_pay_spot_price_results['expected_carbon_tax_revenue_sum'] = expected_carbon_tax_revenue_spot_sum

tax_idx = np.array([0, 2, 4, 6])#np.array([0, 5, 10])
cap_pay_idx = np.array([0, 1, 2, 3, 4])

format_carbon_tax = lambda x: f"{int(np.round(x))}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")
format_cs = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_ps = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_g = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_emissions = lambda x: "{:,}".format(np.round(ton_to_kg(x) / 1000000000.0, 2)).replace(",","\\,")
format_blackouts = lambda x: "{:,}".format(np.round(x / 1000000.0, 2)).replace(",","\\,")
format_w = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")

to_tex = "\\begin{tabular}{ccccccccccccccccccccccc} \n"
to_tex += "\\hline \n"
to_tex += " & & & \\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\text{CS}_{t}\\right]$} & & \\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\text{PS}_{t}\\right]$} & & \\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\text{T}_{t}\\right]$} & & \\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\text{C}_{t}\\right]$} & & \\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\text{E}_{t}\\right]$} & & \\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\text{B}_{t}\\right]$} & & \\multicolumn{2}{c}{$\\Delta \\mathcal{W}$} \\\\ \n"
to_tex += " & & & \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billions kg $\\text{CO}_{2}$-eq)} & & \\multicolumn{2}{c}{(millions MWh)} & & \\multicolumn{2}{c}{(billions A\\$)} \\\\ \n"
to_tex += " & & & \\multicolumn{2}{c}{baseline: --} & & \\multicolumn{2}{c}{baseline: " + format_ps(carbon_tax_capacity_payment_results['expected_producer_surplus_sum'][0,0]) + "} & & \\multicolumn{2}{c}{baseline: " + format_g(carbon_tax_capacity_payment_results['expected_carbon_tax_revenue_sum'][0,0]) + "} & & \\multicolumn{2}{c}{baseline: " + format_g(carbon_tax_capacity_payment_results['expected_revenue_sum'][0,0] - carbon_tax_capacity_payment_results['expected_carbon_tax_revenue_sum'][0,0]) + "} & & \\multicolumn{2}{c}{baseline: " + format_emissions(carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]) + "} & & \\multicolumn{2}{c}{baseline: " + format_blackouts(carbon_tax_capacity_payment_results['expected_blackouts_sum'][0,0]) + "} & & \\multicolumn{2}{c}{baseline: --} \\\\ \n"
to_tex += " & & & low & high & & low & high & & low & high & & low & high & & low & high & & low & high & & low & high \\\\ \n"
to_tex += "$\\tau$ & $\\kappa$ & & price cap & price cap & & price cap & price cap & & price cap & price cap & & price cap & price cap & & price cap & price cap & & price cap & price cap & & price cap & price cap \\\\ \n"
to_tex += "\\cline{1-1} \\cline{2-2} \\cline{4-5} \\cline{7-8} \\cline{10-11} \\cline{13-14} \\cline{16-17} \\cline{19-20} \\cline{22-23} \n"
to_tex += " & & & & & & & & & & & & & & & & & & & \\\\ \n"

scc = 230.0
voll = 1000.0
default_tax_idx = 0
default_cap_pay_idx = 0
results_use = [carbon_tax_capacity_payment_results, high_price_cap_results]
for i in tax_idx:
    to_tex += format_carbon_tax(carbon_taxes_linspace[i])
    for j in cap_pay_idx:
        to_tex += " & " + format_cap_pay(capacity_payments_linspace[j])
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_cs(results_dict['expected_consumer_surplus_sum'][i,j] - carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][default_tax_idx,default_cap_pay_idx])
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_ps(results_dict['expected_producer_surplus_sum'][i,j] - carbon_tax_capacity_payment_results['expected_producer_surplus_sum'][default_tax_idx,default_cap_pay_idx])
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_g(results_dict['expected_carbon_tax_revenue_sum'][i,j] - carbon_tax_capacity_payment_results['expected_carbon_tax_revenue_sum'][default_tax_idx,default_cap_pay_idx])
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_g((results_dict['expected_revenue_sum'][i,j] - results_dict['expected_carbon_tax_revenue_sum'][i,j]) - (carbon_tax_capacity_payment_results['expected_revenue_sum'][default_tax_idx,default_cap_pay_idx] - carbon_tax_capacity_payment_results['expected_carbon_tax_revenue_sum'][default_tax_idx,default_cap_pay_idx]))
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_emissions(results_dict['expected_emissions_sum'][i,j] - carbon_tax_capacity_payment_results['expected_emissions_sum'][default_tax_idx,default_cap_pay_idx])
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_blackouts(results_dict['expected_blackouts_sum'][i,j] - carbon_tax_capacity_payment_results['expected_blackouts_sum'][default_tax_idx,default_cap_pay_idx])
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_w((results_dict['expected_consumer_surplus_sum'][i,j] + results_dict['expected_producer_surplus_sum'][i,j] + results_dict['expected_revenue_sum'][i,j] - scc * results_dict['expected_emissions_sum'][i,j] - voll * results_dict['expected_blackouts_sum'][i,j]) - (carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][default_tax_idx,default_cap_pay_idx] + carbon_tax_capacity_payment_results['expected_producer_surplus_sum'][default_tax_idx,default_cap_pay_idx] + carbon_tax_capacity_payment_results['expected_revenue_sum'][default_tax_idx,default_cap_pay_idx] - scc * carbon_tax_capacity_payment_results['expected_emissions_sum'][default_tax_idx,default_cap_pay_idx] - voll * carbon_tax_capacity_payment_results['expected_blackouts_sum'][default_tax_idx,default_cap_pay_idx]))
        to_tex += " \\\\ \n"
        if (i != tax_idx[-1]) and (j == cap_pay_idx[-1]):
            to_tex += " & & & & & & & & & & & & & & & & & & & \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

if save_output:
    create_file(gv.tables_path + "policy_welfare.tex", to_tex)
if show_output:
    print(to_tex)

# %%
# Cost / ton of emissions avoided

cost_cs_per_ton_avoided = ((carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'] - carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][0,0]) / -(carbon_tax_capacity_payment_results['expected_emissions_sum'] - carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]))
cost_csg_per_ton_avoided = ((carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'] + carbon_tax_capacity_payment_results['expected_revenue_sum'] - carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][0,0] - carbon_tax_capacity_payment_results['expected_revenue_sum'][0,0]) / -(carbon_tax_capacity_payment_results['expected_emissions_sum'] - carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]))
cost_pmw_per_ton_avoided = ((carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'] + carbon_tax_capacity_payment_results['expected_producer_surplus_sum'] + carbon_tax_capacity_payment_results['expected_revenue_sum'] - carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][0,0] - carbon_tax_capacity_payment_results['expected_producer_surplus_sum'][0,0] - carbon_tax_capacity_payment_results['expected_revenue_sum'][0,0]) / -(carbon_tax_capacity_payment_results['expected_emissions_sum'] - carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]))
cost_cs_per_ton_avoided_highpricecap = ((high_price_cap_results['expected_consumer_surplus_sum'] - high_price_cap_results['expected_consumer_surplus_sum'][0,0]) / -(high_price_cap_results['expected_emissions_sum'] - high_price_cap_results['expected_emissions_sum'][0,0]))
cost_csg_per_ton_avoided_highpricecap = ((high_price_cap_results['expected_consumer_surplus_sum'] + high_price_cap_results['expected_revenue_sum'] - high_price_cap_results['expected_consumer_surplus_sum'][0,0] - high_price_cap_results['expected_revenue_sum'][0,0]) / -(high_price_cap_results['expected_emissions_sum'] - high_price_cap_results['expected_emissions_sum'][0,0]))
cost_pmw_per_ton_avoided_highpricecap = ((high_price_cap_results['expected_consumer_surplus_sum'] + high_price_cap_results['expected_producer_surplus_sum'] + high_price_cap_results['expected_revenue_sum'] - high_price_cap_results['expected_consumer_surplus_sum'][0,0] - high_price_cap_results['expected_producer_surplus_sum'][0,0] - high_price_cap_results['expected_revenue_sum'][0,0]) / -(high_price_cap_results['expected_emissions_sum'] - high_price_cap_results['expected_emissions_sum'][0,0]))
pmw_highpricecap = high_price_cap_results['expected_consumer_surplus_sum'] + high_price_cap_results['expected_producer_surplus_sum'] + high_price_cap_results['expected_revenue_sum'] - carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][0,0] - carbon_tax_capacity_payment_results['expected_producer_surplus_sum'][0,0] - carbon_tax_capacity_payment_results['expected_revenue_sum'][0,0]
pmw_lowpricecap = carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'] + carbon_tax_capacity_payment_results['expected_producer_surplus_sum'] + carbon_tax_capacity_payment_results['expected_revenue_sum'] - carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][0,0] - carbon_tax_capacity_payment_results['expected_producer_surplus_sum'][0,0] - carbon_tax_capacity_payment_results['expected_revenue_sum'][0,0]
if save_output:
    create_file(gv.stats_path + "cost_cs_per_ton_avoided.tex", f"{-cost_cs_per_ton_avoided[2,0]:.2f}")
    create_file(gv.stats_path + "cost_csg_per_ton_avoided.tex", f"{-cost_csg_per_ton_avoided[2,0]:.2f}")
    create_file(gv.stats_path + "cost_pmw_per_ton_avoided.tex", f"{-cost_pmw_per_ton_avoided[2,0]:.2f}")
    create_file(gv.stats_path + "pmw_nopolicy_highpricecap.tex", f"{pmw_highpricecap[0,0]/1000000000.0:.2f}")
    create_file(gv.stats_path + "pmw_modcappay_lowpricecap.tex", f"{pmw_lowpricecap[0,2]/1000000000.0:.2f}")
if show_output:
    print(f"Cost per ton avoided (ΔCS only): ${-cost_cs_per_ton_avoided[2,0]:.2f}")
    print(f"Cost per ton avoided (ΔCS + ΔG): ${-cost_csg_per_ton_avoided[2,0]:.2f}")
    print(f"Cost per ton avoided (ΔCS + ΔPS + ΔG): ${-cost_pmw_per_ton_avoided[2,0]:.2f}")
    print(f"Cost per ton avoided (ΔCS only), high price cap: ${-cost_cs_per_ton_avoided_highpricecap[2,0]:.2f}")
    print(f"Cost per ton avoided (ΔCS + ΔG), high price cap: ${-cost_csg_per_ton_avoided_highpricecap[2,0]:.2f}")
    print(f"Cost per ton avoided (ΔCS + ΔPS + ΔG), high price cap: ${-cost_pmw_per_ton_avoided_highpricecap[2,0]:.2f}")
    print(f"PMW (ΔCS + ΔPS + ΔG), high price cap, no policy: {pmw_highpricecap[0,0]/1000000000.0:.2f}")
    print(f"PMW (ΔCS + ΔPS + ΔG), low price cap, moderate capacity price: {pmw_lowpricecap[0,2]/1000000000.0:.2f}")

# %%
# Additional joint policies table

tax_idx = np.array([0, 2, 4, 6])#np.array([0, 5, 10])
cap_pay_idx = np.array([0, 1, 2, 3, 4])

format_carbon_tax = lambda x: f"{int(np.round(x))}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")
format_cs = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_ps = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_g = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_emissions = lambda x: "{:,}".format(np.round(ton_to_kg(x) / 1000000000.0, 2)).replace(",","\\,")
format_blackouts = lambda x: "{:,}".format(np.round(x / 1000000.0, 2)).replace(",","\\,")
format_w = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")

to_tex = "\\begin{tabular}{cccccccccccccc} \n"
to_tex += "\\hline \n"
to_tex += " & & & "
to_tex += "\\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\left(\\text{CS}_{t} - VOLL \\times \\text{B}_{t}\\right)\\right]$} & & "
to_tex += "\\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\left(\\text{CS}_{t} + C_{t} - VOLL \\times \\text{B}_{t}\\right)\\right]$} & & "
to_tex += "\\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\left(\\text{CS}_{t} + \\text{T}_{t} + \\text{C}_{t} - VOLL \\times \\text{B}_{t}\\right)\\right]$} & & "
to_tex += "\\multicolumn{2}{c}{$\\Delta \\mathbb{E}\\left[\\sum_{t=0}^{\\infty} \\beta^{t} \\left(\\text{CS}_{t} + \\text{T}_{t} + \\text{C}_{t} - SCC \\times \\text{E}_{t} - VOLL \\times \\text{B}_{t}\\right)\\right]$} \\\\ \n"
to_tex += " & & & \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billions A\\$)} \\\\ \n"
to_tex += " & & & low & high & & low & high & & low & high & & low & high \\\\ \n"
to_tex += "$\\tau$ & $\\kappa$ & & price cap & price cap & & price cap & price cap & & price cap & price cap & & price cap & price cap \\\\ \n"
to_tex += "\\cline{1-1} \\cline{2-2} \\cline{4-5} \\cline{7-8} \\cline{10-11} \\cline{13-14} \n"
to_tex += " & & & & & & & & & & & & & \\\\ \n"

scc = 230.0
voll = 1000.0
default_tax_idx = 0
default_cap_pay_idx = 0
results_use = [carbon_tax_capacity_payment_results, high_price_cap_results]
for i in tax_idx:
    to_tex += format_carbon_tax(carbon_taxes_linspace[i])
    for j in cap_pay_idx:
        to_tex += " & " + format_cap_pay(capacity_payments_linspace[j])
        to_tex += " & "
        for results_dict in results_use:
            w_use = results_dict['expected_consumer_surplus_sum'][i,j] - voll * results_dict['expected_blackouts_sum'][i,j]
            w_comp = carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][default_tax_idx,default_cap_pay_idx] - voll * carbon_tax_capacity_payment_results['expected_blackouts_sum'][default_tax_idx,default_cap_pay_idx]
            to_tex += " & " + format_w(w_use - w_comp)
        to_tex += " & "
        for results_dict in results_use:
            w_use = results_dict['expected_consumer_surplus_sum'][i,j] + results_dict['expected_revenue_sum'][i,j] - results_dict['expected_carbon_tax_revenue_sum'][i,j] - voll * results_dict['expected_blackouts_sum'][i,j]
            w_comp = carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][default_tax_idx,default_cap_pay_idx] + carbon_tax_capacity_payment_results['expected_revenue_sum'][default_tax_idx,default_cap_pay_idx] - carbon_tax_capacity_payment_results['expected_carbon_tax_revenue_sum'][default_tax_idx,default_cap_pay_idx] - voll * carbon_tax_capacity_payment_results['expected_blackouts_sum'][default_tax_idx,default_cap_pay_idx]
            to_tex += " & " + format_w(w_use - w_comp)
        to_tex += " & "
        for results_dict in results_use:
            w_use = results_dict['expected_consumer_surplus_sum'][i,j] + results_dict['expected_revenue_sum'][i,j] - voll * results_dict['expected_blackouts_sum'][i,j]
            w_comp = carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][default_tax_idx,default_cap_pay_idx] + carbon_tax_capacity_payment_results['expected_revenue_sum'][default_tax_idx,default_cap_pay_idx] - voll * carbon_tax_capacity_payment_results['expected_blackouts_sum'][default_tax_idx,default_cap_pay_idx]
            to_tex += " & " + format_w(w_use - w_comp)
        to_tex += " & "
        for results_dict in results_use:
            w_use = results_dict['expected_consumer_surplus_sum'][i,j] + results_dict['expected_revenue_sum'][i,j] - voll * results_dict['expected_blackouts_sum'][i,j] - scc * results_dict['expected_emissions_sum'][i,j]
            w_comp = carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][default_tax_idx,default_cap_pay_idx] + carbon_tax_capacity_payment_results['expected_revenue_sum'][default_tax_idx,default_cap_pay_idx] - voll * carbon_tax_capacity_payment_results['expected_blackouts_sum'][default_tax_idx,default_cap_pay_idx] - scc * carbon_tax_capacity_payment_results['expected_emissions_sum'][default_tax_idx,default_cap_pay_idx]
            to_tex += " & " + format_w(w_use - w_comp)
        to_tex += " \\\\ \n"
        if (i != tax_idx[-1]) and (j == cap_pay_idx[-1]):
            to_tex += " & & & & & & & & & & & & & \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

if save_output:
    create_file(gv.tables_path + "policy_welfare_additional.tex", to_tex)
if show_output:
    print(to_tex)

# %%
# Optimal carbon taxes / capacity payments

# Set up the space would like to consider
min_carbon_tax = 0.0
max_carbon_tax = 300.0
num_points = 1000 # need large number
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points)
min_capacity_payment = 0.0
max_capacity_payment = 200000.0
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points)

interp_CS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_T = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_carbon_tax_revenue_sum'], kx=3, ky=3, s=0.0)
interp_C = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_revenue_sum'] - carbon_tax_capacity_payment_results['expected_carbon_tax_revenue_sum'], kx=3, ky=3, s=0.0)
interp_emissions = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
_interp_blackouts = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_blackouts = lambda x, y: np.maximum(0.0, _interp_blackouts(x,y))
interp_CS_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_T_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_carbon_tax_revenue_sum'], kx=3, ky=3, s=0.0)
interp_C_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_revenue_sum'] - high_price_cap_results['expected_carbon_tax_revenue_sum'], kx=3, ky=3, s=0.0)
interp_CSPS_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
_interp_blackouts_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_highpricecap = lambda x, y: np.maximum(0.0, _interp_blackouts_highpricecap(x,y))

scc_vals = np.array([230.0])
num_points_scc = scc_vals.shape[0]
voll_vals = np.array([1000.0, 10000.0])
num_points_voll = voll_vals.shape[0]
num_people = 1100000.0 * 1000.0 # interpretation: thousand A$ per person

# Carbon tax, low price cap
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_T(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_C(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
E_W_compare = E_W[:,:,0,0]
CS_compare = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
PS_compare = interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
T_compare = interp_T(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
C_compare = interp_C(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
emissions_compare = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
blackouts_compare = interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
max_policy = np.argmax(E_W[:,:,:,0], axis=2) # capacity payments = 0
carbon_tax_alone = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone = (np.max(E_W[:,:,:,0], axis=2) - E_W_compare) / num_people
cs_carbon_tax_alone = (np.take_along_axis(interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_carbon_tax_alone = (np.take_along_axis(interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
t_carbon_tax_alone = (np.take_along_axis(interp_T(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - T_compare) / num_people
c_carbon_tax_alone = (np.take_along_axis(interp_C(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - C_compare) / num_people
emissions_carbon_tax_alone = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Carbon tax, high price cap
E_W = interp_CS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_C_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_T_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,:,0], axis=2) # capacity payments = 0
carbon_tax_alone_price_cap = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_price_cap = (np.max(E_W[:,:,:,0], axis=2) - E_W_compare) / num_people
cs_carbon_tax_alone_highpricecap = (np.take_along_axis(interp_CS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_carbon_tax_alone_highpricecap = (np.take_along_axis(interp_PS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
t_carbon_tax_alone_highpricecap = (np.take_along_axis(interp_T_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - T_compare) / num_people
c_carbon_tax_alone_highpricecap = (np.take_along_axis(interp_C_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - C_compare) / num_people
emissions_carbon_tax_alone_highpricecap = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone_highpricecap = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Low price cap
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_T(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_C(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], E_W.shape[1], -1)), axis=2)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[2:]))
carbon_tax_joint = carbon_taxes_fine[max_policy_1]
capacity_payment_joint = capacity_payments_fine[max_policy_2]
max_w_joint = (np.max(np.reshape(E_W, (E_W.shape[0],E_W.shape[1],-1)), axis=2) - E_W_compare) / num_people
cs_joint = (np.take_along_axis(np.reshape(interp_CS(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_joint = (np.take_along_axis(np.reshape(interp_PS(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
t_joint = (np.take_along_axis(np.reshape(interp_T(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - T_compare) / num_people
c_joint = (np.take_along_axis(np.reshape(interp_C(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - C_compare) / num_people
emissions_joint = -scc_vals[:,np.newaxis] * (np.take_along_axis(np.reshape(interp_emissions(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_joint = -voll_vals[np.newaxis,:] * (np.take_along_axis(np.reshape(interp_blackouts(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# High price cap
E_W = interp_CS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_T_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_C_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], E_W.shape[1], -1)), axis=2)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[2:]))
carbon_tax_pricecap = carbon_taxes_fine[max_policy_1]
capacity_payment_pricecap = capacity_payments_fine[max_policy_2]
max_w_pricecap = (np.max(np.reshape(E_W, (E_W.shape[0],E_W.shape[1],-1)), axis=2) - E_W_compare) / num_people
cs_pricecap = (np.take_along_axis(np.reshape(interp_CS_highpricecap(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_pricecap = (np.take_along_axis(np.reshape(interp_PS_highpricecap(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
t_pricecap = (np.take_along_axis(np.reshape(interp_T_highpricecap(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - T_compare) / num_people
c_pricecap = (np.take_along_axis(np.reshape(interp_C_highpricecap(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - C_compare) / num_people
emissions_pricecap = -scc_vals[:,np.newaxis] * (np.take_along_axis(np.reshape(interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_pricecap = -voll_vals[np.newaxis,:] * (np.take_along_axis(np.reshape(interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

format_carbon_tax = lambda x: f"{np.round(x, 1)}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x / 100.0) * 100.0)).replace(",","\\,")
format_W = lambda x: "{:,}".format(np.round(x, 1)).replace(",","\\,") if not np.isclose(x, 0.0) else "0.0"
format_scc = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")
format_voll = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")

to_tex = "\\begin{tabular}{rrccccccccccccc} \n"
to_tex += "\\hline \\\\ \n"
to_tex += " & & \\multicolumn{2}{c}{carbon tax alone, low $\\bar{P}$} & & \\multicolumn{2}{c}{carbon tax alone, high $\\bar{P}$} & & \\multicolumn{3}{c}{joint policies, low $\\bar{P}$} & & \\multicolumn{3}{c}{joint policies, high $\\bar{P}$} \\\\ \n"
to_tex += " \\cline{3-4} \\cline{6-7} \\cline{9-11} \\cline{13-15} \n"
to_tex += "$VOLL$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
to_tex += "\\hline \n"

for i, scc_val in enumerate(scc_vals):
    for j, voll_val in enumerate(voll_vals):
        to_tex += "\\textbf{" + format_voll(voll_val) + "}"
        to_tex += " & & "
        to_tex += "\\textbf{" + format_carbon_tax(carbon_tax_alone[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_carbon_tax_alone[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_alone_price_cap[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_carbon_tax_alone_price_cap[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_joint[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_cap_pay(capacity_payment_joint[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_joint[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_pricecap[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_cap_pay(capacity_payment_pricecap[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_pricecap[i,j]) + "}"
        to_tex += " \\\\ \n"

        # Decomposed welfare result
        to_tex += " & $\\Delta \\text{CS}$"
        to_tex += " & & " + format_W(cs_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(cs_carbon_tax_alone_highpricecap[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(cs_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(cs_pricecap[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $\\Delta \\text{PS}$"
        to_tex += " & & " + format_W(ps_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(ps_carbon_tax_alone_highpricecap[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(ps_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(ps_pricecap[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $\\Delta \\text{T}$"
        to_tex += " & & " + format_W(t_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(t_carbon_tax_alone_highpricecap[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(t_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(t_pricecap[i,j])
        to_tex += " \\\\ \n"
        
        to_tex += " & $\\Delta \\text{C}$"
        to_tex += " & & " + format_W(c_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(c_carbon_tax_alone_highpricecap[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(c_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(c_pricecap[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $-SCC \\times \\Delta \\text{E}$"
        to_tex += " & & " + format_W(emissions_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(emissions_carbon_tax_alone_highpricecap[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(emissions_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(emissions_pricecap[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $-VOLL \\times \\Delta \\text{B}$"
        to_tex += " & & " + format_W(blackouts_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(blackouts_carbon_tax_alone_highpricecap[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(blackouts_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(blackouts_pricecap[i,j])
        to_tex += " \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

if save_output:
    create_file(gv.tables_path + "optimal_policy.tex", to_tex)
    create_file(gv.stats_path + "optimal_policy_scc.tex", format_scc(scc_vals[0]))
    create_file(gv.stats_path + "optimal_policy_voll.tex", format_scc(voll_vals[0]))
    create_file(gv.stats_path + "optimal_policy_voll_large.tex", format_scc(voll_vals[1]))

if show_output:
    print(to_tex)

# %%
# Intro numbers

# Carbon tax alone
carbon_tax_pct_emissions = (interp_emissions(scc_vals[0], 0.0)[0,0] - interp_emissions(0.0, 0.0)[0,0]) / interp_emissions(0.0, 0.0)[0,0]
carbon_tax_pct_blackouts = (interp_blackouts(scc_vals[0], 0.0)[0,0] - interp_blackouts(0.0, 0.0)[0,0]) / interp_blackouts(0.0, 0.0)[0,0]

# Capacity payments alone
capacity_payment_pct_emissions = (interp_emissions(0.0, 150000.0)[0,0] - interp_emissions(0.0, 0.0)[0,0]) / interp_emissions(0.0, 0.0)[0,0]
capacity_payment_pct_blackouts = (interp_blackouts(0.0, 150000.0)[0,0] - interp_blackouts(0.0, 0.0)[0,0]) / interp_blackouts(0.0, 0.0)[0,0]

# Carbon tax + capacity payments
carbon_tax_capacity_payment_pct_emissions = (interp_emissions(scc_vals[0], 150000.0)[0,0] - interp_emissions(0.0, 0.0)[0,0]) / interp_emissions(0.0, 0.0)[0,0]
carbon_tax_capacity_payment_pct_blackouts = (interp_blackouts(scc_vals[0], 150000.0)[0,0] - interp_blackouts(0.0, 0.0)[0,0]) / interp_blackouts(0.0, 0.0)[0,0]

# Save numbers
create_file(f"{gv.stats_path}carbon_tax_pct_emissions.tex", f"{(-carbon_tax_pct_emissions * 100.0):.1f}")
create_file(f"{gv.stats_path}carbon_tax_pct_blackouts.tex", f"{(carbon_tax_pct_blackouts * 100.0):.1f}")
create_file(f"{gv.stats_path}capacity_payment_pct_emissions.tex", f"{(capacity_payment_pct_emissions * 100.0):.1f}")
create_file(f"{gv.stats_path}capacity_payment_pct_blackouts.tex", f"{(-capacity_payment_pct_blackouts * 100.0):.1f}")
create_file(f"{gv.stats_path}carbon_tax_capacity_payment_pct_emissions.tex", f"{(-carbon_tax_capacity_payment_pct_emissions * 100.0):.1f}")
create_file(f"{gv.stats_path}carbon_tax_capacity_payment_pct_blackouts.tex", f"{(-carbon_tax_capacity_payment_pct_blackouts * 100.0):.1f}")

print(f"carbon_tax_pct_emissions: {(-carbon_tax_pct_emissions * 100.0):.1f}")
print(f"carbon_tax_pct_blackouts: {(carbon_tax_pct_blackouts * 100.0):.1f}")
print(f"capacity_payment_pct_emissions: {(capacity_payment_pct_emissions * 100.0):.1f}")
print(f"capacity_payment_pct_blackouts: {(-capacity_payment_pct_blackouts * 100.0):.1f}")
print(f"carbon_tax_capacity_payment_pct_emission: {(-carbon_tax_capacity_payment_pct_emissions * 100.0):.1f}")
print(f"carbon_tax_capacity_payment_pct_blackouts: {(-carbon_tax_capacity_payment_pct_blackouts * 100.0):.1f}")

# %%
# Alternative policies: best could do subject to constraint

# Only use default values
scc_use = 0
voll_use = 0
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine) + interp_PS(carbon_taxes_fine, capacity_payments_fine) + interp_T(carbon_taxes_fine, capacity_payments_fine) + interp_C(carbon_taxes_fine, capacity_payments_fine) - scc_vals[scc_use] * interp_emissions(carbon_taxes_fine, capacity_payments_fine) - voll_vals[voll_use] * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)
E_W_compare = E_W[0,0]
CS_compare = interp_CS(carbon_taxes_fine, capacity_payments_fine)[0,0]
PS_compare = interp_PS(carbon_taxes_fine, capacity_payments_fine)[0,0]
T_compare = interp_T(carbon_taxes_fine, capacity_payments_fine)[0,0]
C_compare = interp_C(carbon_taxes_fine, capacity_payments_fine)[0,0]
emissions_compare = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[0,0]
blackouts_compare = interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[0,0]
feasible_threshold = 0.0

# \max \Delta CS + \Delta G + \Delta E + \Delta B
E_W_consumerstandard = interp_CS(carbon_taxes_fine, capacity_payments_fine) + interp_T(carbon_taxes_fine, capacity_payments_fine) + interp_C(carbon_taxes_fine, capacity_payments_fine) - scc_vals[scc_use] * interp_emissions(carbon_taxes_fine, capacity_payments_fine) - voll_vals[voll_use] * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)
min_idx = np.unravel_index(np.argmax(E_W_consumerstandard), E_W.shape)
best_tax_wcs = carbon_taxes_fine[min_idx[0]]
best_payment_wcs = capacity_payments_fine[min_idx[1]]
emissions_baseline = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[0,0]
emissions_best = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[min_idx]
emissions_frac_wcs = (emissions_best - emissions_baseline) / emissions_baseline
w_wcs = (E_W[min_idx] - E_W_compare) / num_people
cs_wcs = (interp_CS(carbon_taxes_fine, capacity_payments_fine)[min_idx] - CS_compare) / num_people
ps_wcs = (interp_PS(carbon_taxes_fine, capacity_payments_fine)[min_idx] - PS_compare) / num_people
t_wcs = (interp_T(carbon_taxes_fine, capacity_payments_fine)[min_idx] - T_compare) / num_people
c_wcs = (interp_C(carbon_taxes_fine, capacity_payments_fine)[min_idx] - C_compare) / num_people
emissions_wcs = (interp_emissions(carbon_taxes_fine, capacity_payments_fine)[min_idx] - emissions_compare) / 1000000.0
blackouts_wcs = (np.maximum(0.0, interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_wcs = (np.maximum(0.0, interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[min_idx]) - blackouts_compare) / blackouts_compare

# \max \Delta W
min_idx = np.unravel_index(np.argmax(E_W), E_W.shape)
best_tax_w = carbon_taxes_fine[min_idx[0]]
best_payment_w = capacity_payments_fine[min_idx[1]]
emissions_baseline = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[0,0]
emissions_best = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[min_idx]
emissions_frac_w = (emissions_best - emissions_baseline) / emissions_baseline
w_w = (E_W[min_idx] - E_W_compare) / num_people
cs_w = (interp_CS(carbon_taxes_fine, capacity_payments_fine)[min_idx] - CS_compare) / num_people
ps_w = (interp_PS(carbon_taxes_fine, capacity_payments_fine)[min_idx] - PS_compare) / num_people
t_w = (interp_T(carbon_taxes_fine, capacity_payments_fine)[min_idx] - T_compare) / num_people
c_w = (interp_C(carbon_taxes_fine, capacity_payments_fine)[min_idx] - C_compare) / num_people
emissions_w = (interp_emissions(carbon_taxes_fine, capacity_payments_fine)[min_idx] - emissions_compare) / 1000000.0
blackouts_w = (np.maximum(0.0, interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_w = (np.maximum(0.0, interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[min_idx]) - blackouts_compare) / blackouts_compare

# Functions to perform 3D interpolation analogous to the 2D one above using scipy.ndimage
def physical_to_index_coords(x, grid):
    """
    Convert physical values x to fractional array indices on a uniform grid.
    Assumes grid is uniformly spaced.
    """
    dx = grid[1] - grid[0]
    return (np.asarray(x) - grid[0]) / dx
def cubic_spline_interpolator_3d(x_grid, y_grid, z_grid, values):
    """
    Returns a function that interpolates using a 3D cubic spline over a uniform grid.

    Automatically performs broadcasting (like RectBivariateSpline).

    Inputs:
    - x_grid, y_grid, z_grid: 1D uniform grids
    - values: 3D array of shape (len(x_grid), len(y_grid), len(z_grid))

    Returns:
    - interpolator(x_vals, y_vals, z_vals): evaluates interpolated values
    """
    def interpolator(x_vals, y_vals, z_vals):
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)
        z_vals = np.asarray(z_vals)

        # If inputs are 1D, generate meshgrid (like RectBivariateSpline behavior)
        if x_vals.ndim == y_vals.ndim == z_vals.ndim == 1:
            x_vals, y_vals, z_vals = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

        # Convert to fractional index coordinates
        xi = physical_to_index_coords(x_vals, x_grid)
        yi = physical_to_index_coords(y_vals, y_grid)
        zi = physical_to_index_coords(z_vals, z_grid)

        # Interpolate
        coords = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])
        vals = ndimage.map_coordinates(values, coords, order=3, mode="nearest")

        return vals.reshape(x_vals.shape)

    return interpolator

interp_CS_sep = cubic_spline_interpolator_3d(carbon_taxes_linspace, capacity_payments_linspace, capacity_payments_linspace2, extended_results['expected_consumer_surplus_sum'])
interp_PS_sep = cubic_spline_interpolator_3d(carbon_taxes_linspace, capacity_payments_linspace, capacity_payments_linspace2, extended_results['expected_producer_surplus_sum'])
interp_T_sep = cubic_spline_interpolator_3d(carbon_taxes_linspace, capacity_payments_linspace, capacity_payments_linspace2, extended_results['expected_carbon_tax_revenue_sum'])
interp_C_sep = cubic_spline_interpolator_3d(carbon_taxes_linspace, capacity_payments_linspace, capacity_payments_linspace2, extended_results['expected_revenue_sum'] - extended_results['expected_carbon_tax_revenue_sum'])
interp_emissions_sep = cubic_spline_interpolator_3d(carbon_taxes_linspace, capacity_payments_linspace, capacity_payments_linspace2, extended_results['expected_emissions_sum'])
_interp_blackouts_sep = cubic_spline_interpolator_3d(carbon_taxes_linspace, capacity_payments_linspace, capacity_payments_linspace2, extended_results['expected_blackouts_sum'])
interp_blackouts_sep = lambda x, y, z: np.maximum(0.0, _interp_blackouts_sep(x, y, z))

num_points_restricted = 100 # smaller number for the sake of computation time and memory
num_points_restricted_in_num_points = int(np.ceil(num_points / num_points_restricted))
def subset_linspace(min_idx):
    # Find area around min_idx
    carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points)[np.maximum(0, num_points_restricted_in_num_points * (min_idx[0] - 2)):np.minimum(num_points, num_points_restricted_in_num_points * (min_idx[0] + 2))]
    capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points)[np.maximum(0, num_points_restricted_in_num_points * (min_idx[1] - 2)):np.minimum(num_points, num_points_restricted_in_num_points * (min_idx[1] + 2))]
    capacity_payments_fine2 = np.linspace(np.min(capacity_payments_linspace2), np.max(capacity_payments_linspace2), num_points)[np.maximum(0, num_points_restricted_in_num_points * (min_idx[2] - 2)):np.minimum(num_points, num_points_restricted_in_num_points * (min_idx[2] + 2))]

    # Add min values if not included:
    if not np.isin(min_carbon_tax, carbon_taxes_fine):
        carbon_taxes_fine = np.concatenate((np.array([min_carbon_tax]), carbon_taxes_fine))
    if not np.isin(min_capacity_payment, capacity_payments_fine):
        capacity_payments_fine = np.concatenate((np.array([min_capacity_payment]), capacity_payments_fine))
    if not np.isin(np.min(capacity_payments_linspace2), capacity_payments_fine2):
        capacity_payments_fine2 = np.concatenate((np.array([np.min(capacity_payments_linspace2)]), capacity_payments_fine2))
    
    return carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2


# max \Delta E s.t. \Delta CS >= 0 w/ separate capacity prices
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points_restricted)
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points_restricted)
capacity_payments_fine2 = np.copy(capacity_payments_fine)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = (E_W_constrained_change >= feasible_threshold) & (capacity_payments_fine[np.newaxis,:,np.newaxis] == capacity_payments_fine2[np.newaxis,np.newaxis,:])
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2 = subset_linspace(min_idx)
capacity_payments_fine2 = np.copy(capacity_payments_fine)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = (E_W_constrained_change >= feasible_threshold) & (capacity_payments_fine[np.newaxis,:,np.newaxis] == capacity_payments_fine2[np.newaxis,np.newaxis,:])
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_cs = carbon_taxes_fine[min_idx[0]]
best_payment_cs = capacity_payments_fine[min_idx[1]]
emissions_baseline = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[0,0,0]
emissions_best = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]
emissions_frac_cs = (emissions_best - emissions_baseline) / emissions_baseline
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
w_cs = (E_W[min_idx] - E_W_compare) / num_people
cs_cs = (interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - CS_compare) / num_people
ps_cs = (interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - PS_compare) / num_people
t_cs = (interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - T_compare) / num_people
c_cs = (interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - C_compare) / num_people
emissions_cs = (interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - emissions_compare) / 1000000.0
blackouts_cs = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_cs = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta E s.t. \Delta CS + \Delta tax >= 0 w/ separate capacity prices
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points_restricted)
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points_restricted)
capacity_payments_fine2 = np.copy(capacity_payments_fine)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = (E_W_constrained_change >= feasible_threshold) & (capacity_payments_fine[np.newaxis,:,np.newaxis] == capacity_payments_fine2[np.newaxis,np.newaxis,:])
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2 = subset_linspace(min_idx)
capacity_payments_fine2 = np.copy(capacity_payments_fine)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = (E_W_constrained_change >= feasible_threshold) & (capacity_payments_fine[np.newaxis,:,np.newaxis] == capacity_payments_fine2[np.newaxis,np.newaxis,:])
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_cstax = carbon_taxes_fine[min_idx[0]]
best_payment_cstax = capacity_payments_fine[min_idx[1]]
emissions_baseline = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[0,0,0]
emissions_best = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]
emissions_frac_cstax = (emissions_best - emissions_baseline) / emissions_baseline
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
w_cstax = (E_W[min_idx] - E_W_compare) / num_people
cs_cstax = (interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - CS_compare) / num_people
ps_cstax = (interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - PS_compare) / num_people
t_cstax = (interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - T_compare) / num_people
c_cstax = (interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - C_compare) / num_people
emissions_cstax = (interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - emissions_compare) / 1000000.0
blackouts_cstax = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_cstax = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta E s.t. \Delta CS + \Delta G >= 0 w/ separate capacity prices
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points_restricted)
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points_restricted)
capacity_payments_fine2 = np.copy(capacity_payments_fine)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = (E_W_constrained_change >= feasible_threshold) & (capacity_payments_fine[np.newaxis,:,np.newaxis] == capacity_payments_fine2[np.newaxis,np.newaxis,:])
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2 = subset_linspace(min_idx)
capacity_payments_fine2 = np.copy(capacity_payments_fine)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = (E_W_constrained_change >= feasible_threshold) & (capacity_payments_fine[np.newaxis,:,np.newaxis] == capacity_payments_fine2[np.newaxis,np.newaxis,:])
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_csg = carbon_taxes_fine[min_idx[0]]
best_payment_csg = capacity_payments_fine[min_idx[1]]
emissions_baseline = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[0,0,0]
emissions_best = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]
emissions_frac_csg = (emissions_best - emissions_baseline) / emissions_baseline
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
w_csg = (E_W[min_idx] - E_W_compare) / num_people
cs_csg = (interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - CS_compare) / num_people
ps_csg = (interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - PS_compare) / num_people
t_csg = (interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - T_compare) / num_people
c_csg = (interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - C_compare) / num_people
emissions_csg = (interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - emissions_compare) / 1000000.0
blackouts_csg = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_csg = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta E s.t. \Delta CS >= 0 w/ separate capacity prices
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points_restricted)
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points_restricted)
capacity_payments_fine2 = np.linspace(np.min(capacity_payments_linspace2), np.max(capacity_payments_linspace2), num_points_restricted)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2 = subset_linspace(min_idx)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_cs_sep = carbon_taxes_fine[min_idx[0]]
best_payment1_cs_sep = capacity_payments_fine[min_idx[1]]
best_payment2_cs_sep = capacity_payments_fine2[min_idx[2]]
emissions_baseline = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[0,0,0]
emissions_best = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]
emissions_frac_cs_sep = (emissions_best - emissions_baseline) / emissions_baseline
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
w_cs_sep = (E_W[min_idx] - E_W_compare) / num_people
cs_cs_sep = (interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - CS_compare) / num_people
ps_cs_sep = (interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - PS_compare) / num_people
t_cs_sep = (interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - T_compare) / num_people
c_cs_sep = (interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - C_compare) / num_people
emissions_cs_sep = (interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - emissions_compare) / 1000000.0
blackouts_cs_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_cs_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta E s.t. \Delta CS + \Delta tax >= 0 w/ separate capacity prices
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points_restricted)
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points_restricted)
capacity_payments_fine2 = np.linspace(np.min(capacity_payments_linspace2), np.max(capacity_payments_linspace2), num_points_restricted)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2 = subset_linspace(min_idx)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_cstax_sep = carbon_taxes_fine[min_idx[0]]
best_payment1_cstax_sep = capacity_payments_fine[min_idx[1]]
best_payment2_cstax_sep = capacity_payments_fine2[min_idx[2]]
emissions_baseline = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[0,0,0]
emissions_best = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]
emissions_frac_cstax_sep = (emissions_best - emissions_baseline) / emissions_baseline
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
w_cstax_sep = (E_W[min_idx] - E_W_compare) / num_people
cs_cstax_sep = (interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - CS_compare) / num_people
ps_cstax_sep = (interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - PS_compare) / num_people
t_cstax_sep = (interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - T_compare) / num_people
c_cstax_sep = (interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - C_compare) / num_people
emissions_cstax_sep = (interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - emissions_compare) / 1000000.0
blackouts_cstax_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_cstax_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta E s.t. \Delta CS + \Delta G >= 0 w/ separate capacity prices
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points_restricted)
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points_restricted)
capacity_payments_fine2 = np.linspace(np.min(capacity_payments_linspace2), np.max(capacity_payments_linspace2), num_points_restricted)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2 = subset_linspace(min_idx)
E_W_constrained = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_csg_sep = carbon_taxes_fine[min_idx[0]]
best_payment1_csg_sep = capacity_payments_fine[min_idx[1]]
best_payment2_csg_sep = capacity_payments_fine2[min_idx[2]]
emissions_baseline = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[0,0,0]
emissions_best = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]
emissions_frac_csg_sep = (emissions_best - emissions_baseline) / emissions_baseline
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
w_csg_sep = (E_W[min_idx] - E_W_compare) / num_people
cs_csg_sep = (interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - CS_compare) / num_people
ps_csg_sep = (interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - PS_compare) / num_people
t_csg_sep = (interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - T_compare) / num_people
c_csg_sep = (interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - C_compare) / num_people
emissions_csg_sep = (interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - emissions_compare) / 1000000.0
blackouts_csg_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_csg_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / blackouts_compare

# \max \Delta CS + \Delta G + \Delta E + \Delta B
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points_restricted)
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points_restricted)
capacity_payments_fine2 = np.linspace(np.min(capacity_payments_linspace2), np.max(capacity_payments_linspace2), num_points_restricted)
E_W_consumerstandard = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
min_idx = np.unravel_index(np.argmax(E_W_consumerstandard), E_W_consumerstandard.shape)
carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2 = subset_linspace(min_idx)
E_W_consumerstandard = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
min_idx = np.unravel_index(np.argmax(E_W_consumerstandard), E_W_consumerstandard.shape)
best_tax_wcs_sep = carbon_taxes_fine[min_idx[0]]
best_payment1_wcs_sep = capacity_payments_fine[min_idx[1]]
best_payment2_wcs_sep = capacity_payments_fine2[min_idx[2]]
emissions_baseline = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[0,0,0]
emissions_best = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]
emissions_frac_wcs_sep = (emissions_best - emissions_baseline) / emissions_baseline
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
w_wcs_sep = (E_W[min_idx] - E_W_compare) / num_people
cs_wcs_sep = (interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - CS_compare) / num_people
ps_wcs_sep = (interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - PS_compare) / num_people
t_wcs_sep = (interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - T_compare) / num_people
c_wcs_sep = (interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - C_compare) / num_people
emissions_wcs_sep = (interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - emissions_compare) / 1000000.0
blackouts_wcs_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_wcs_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta W w/ separate capacity prices
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points_restricted)
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points_restricted)
capacity_payments_fine2 = np.linspace(np.min(capacity_payments_linspace2), np.max(capacity_payments_linspace2), num_points_restricted)
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
min_idx = np.unravel_index(np.argmax(E_W), E_W.shape)
carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2 = subset_linspace(min_idx)
E_W = interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) + interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - scc_vals[scc_use] * interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2) - voll_vals[voll_use] * interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)
min_idx = np.unravel_index(np.argmax(E_W), E_W.shape)
best_tax_w_sep = carbon_taxes_fine[min_idx[0]]
best_payment1_w_sep = capacity_payments_fine[min_idx[1]]
best_payment2_w_sep = capacity_payments_fine2[min_idx[2]]
emissions_baseline = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[0,0,0]
emissions_best = interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]
emissions_frac_w_sep = (emissions_best - emissions_baseline) / emissions_baseline
w_w_sep = (E_W[min_idx] - E_W_compare) / num_people
cs_w_sep = (interp_CS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - CS_compare) / num_people
ps_w_sep = (interp_PS_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - PS_compare) / num_people
t_w_sep = (interp_T_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - T_compare) / num_people
c_w_sep = (interp_C_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - C_compare) / num_people
emissions_w_sep = (interp_emissions_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx] - emissions_compare) / 1000000.0
blackouts_w_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_w_sep = (np.maximum(0.0, interp_blackouts_sep(carbon_taxes_fine, capacity_payments_fine, capacity_payments_fine2)[min_idx]) - blackouts_compare) / blackouts_compare

# Tied to spot market price
capacity_payments_linspace3 = np.copy(spot_price_multiplier)
interp_CS_spot = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace3, cap_pay_spot_price_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS_spot = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace3, cap_pay_spot_price_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_T_spot = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace3, cap_pay_spot_price_results['expected_carbon_tax_revenue_sum'], kx=3, ky=3, s=0.0)
interp_C_spot = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace3, cap_pay_spot_price_results['expected_revenue_sum'] - cap_pay_spot_price_results['expected_carbon_tax_revenue_sum'], kx=3, ky=3, s=0.0)
interp_emissions_spot = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace3, cap_pay_spot_price_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
_interp_blackouts_spot = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace3, cap_pay_spot_price_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_spot = lambda x, y: np.maximum(0.0, _interp_blackouts_spot(x,y))
interp_inv_price_spot = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace3, np.mean(cap_pay_spot_price_results['expected_inv_quantity_weighted_avg_price'], axis=2), kx=3, ky=3, s=0.0)

# max \Delta E s.t. \Delta CS >= 0 w/ separate capacity prices
num_points = 1000
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points)
capacity_payments_fine3 = np.linspace(np.min(spot_price_multiplier), np.max(spot_price_multiplier), num_points)
E_W = interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3) + interp_PS_spot(carbon_taxes_fine, capacity_payments_fine3) + interp_T_spot(carbon_taxes_fine, capacity_payments_fine3) + interp_C_spot(carbon_taxes_fine, capacity_payments_fine3) - scc_vals[scc_use] * interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3) - voll_vals[voll_use] * interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)
E_W_constrained = interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_cs_spot = carbon_taxes_fine[min_idx[0]]
best_payment_cs_spot = capacity_payments_fine3[min_idx[1]]
best_policy_avg_cap_pay_cs_spot = capacity_payments_fine3[min_idx[1]] * interp_inv_price_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_baseline = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[0,0]
emissions_best = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_frac_cs_spot = (emissions_best - emissions_baseline) / emissions_baseline
w_cs_spot = (E_W[min_idx] - E_W_compare) / num_people
cs_cs_spot = (interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - CS_compare) / num_people
ps_cs_spot = (interp_PS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - PS_compare) / num_people
t_cs_spot = (interp_T_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - T_compare) / num_people
c_cs_spot = (interp_C_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - C_compare) / num_people
emissions_cs_spot = (interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - emissions_compare) / 1000000.0
blackouts_cs_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_cs_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta E s.t. \Delta CS + \Delta tax >= 0 w/ separate capacity prices
E_W_constrained = interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3) + interp_C_spot(carbon_taxes_fine, capacity_payments_fine3) - voll_vals[voll_use] * interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_cstax_spot = carbon_taxes_fine[min_idx[0]]
best_payment_cstax_spot = capacity_payments_fine3[min_idx[1]]
best_policy_avg_cap_pay_cstax_spot = capacity_payments_fine3[min_idx[1]] * interp_inv_price_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_baseline = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[0,0]
emissions_best = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_frac_cstax_spot = (emissions_best - emissions_baseline) / emissions_baseline
w_cstax_spot = (E_W[min_idx] - E_W_compare) / num_people
cs_cstax_spot = (interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - CS_compare) / num_people
ps_cstax_spot = (interp_PS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - PS_compare) / num_people
t_cstax_spot = (interp_T_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - T_compare) / num_people
c_cstax_spot = (interp_C_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - C_compare) / num_people
emissions_cstax_spot = (interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - emissions_compare) / 1000000.0
blackouts_cstax_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_cstax_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta E s.t. \Delta CS + \Delta G >= 0 w/ separate capacity prices
E_W_constrained = interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3) + interp_T_spot(carbon_taxes_fine, capacity_payments_fine3) + interp_C_spot(carbon_taxes_fine, capacity_payments_fine3) - voll_vals[voll_use] * interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)
E_W_constrained_change = E_W_constrained - E_W_constrained[0,0]
feasible_mask = E_W_constrained_change >= feasible_threshold
emissions_feasible = np.where(feasible_mask, interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3), np.inf)
min_idx = np.unravel_index(np.argmin(emissions_feasible), emissions_feasible.shape)
best_tax_csg_spot = carbon_taxes_fine[min_idx[0]]
best_payment_csg_spot = capacity_payments_fine3[min_idx[1]]
best_policy_avg_cap_pay_csg_spot = capacity_payments_fine3[min_idx[1]] * interp_inv_price_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_baseline = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[0,0]
emissions_best = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_frac_csg_spot = (emissions_best - emissions_baseline) / emissions_baseline
w_csg_spot = (E_W[min_idx] - E_W_compare) / num_people
cs_csg_spot = (interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - CS_compare) / num_people
ps_csg_spot = (interp_PS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - PS_compare) / num_people
t_csg_spot = (interp_T_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - T_compare) / num_people
c_csg_spot = (interp_C_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - C_compare) / num_people
emissions_csg_spot = (interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - emissions_compare) / 1000000.0
blackouts_csg_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_csg_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / blackouts_compare

# \max \Delta CS + \Delta G + \Delta E + \Delta B
E_W_consumerstandard = interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3) + interp_T_spot(carbon_taxes_fine, capacity_payments_fine3) + interp_C_spot(carbon_taxes_fine, capacity_payments_fine3) - scc_vals[scc_use] * interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3) - voll_vals[voll_use] * interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)
min_idx = np.unravel_index(np.argmax(E_W_consumerstandard), E_W.shape)
best_tax_wcs_spot = carbon_taxes_fine[min_idx[0]]
best_payment_wcs_spot = capacity_payments_fine3[min_idx[1]]
best_policy_avg_cap_pay_wcs_spot = capacity_payments_fine3[min_idx[1]] * interp_inv_price_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_baseline = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[0,0]
emissions_best = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_frac_wcs_spot = (emissions_best - emissions_baseline) / emissions_baseline
w_wcs_spot = (E_W[min_idx] - E_W_compare) / num_people
cs_wcs_spot = (interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - CS_compare) / num_people
ps_wcs_spot = (interp_PS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - PS_compare) / num_people
t_wcs_spot = (interp_T_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - T_compare) / num_people
c_wcs_spot = (interp_C_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - C_compare) / num_people
emissions_wcs_spot = (interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - emissions_compare) / 1000000.0
blackouts_wcs_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_wcs_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / blackouts_compare

# max \Delta W w/ separate capacity prices
min_idx = np.unravel_index(np.argmax(E_W), E_W.shape)
best_tax_w_spot = carbon_taxes_fine[min_idx[0]]
best_payment_w_spot = capacity_payments_fine3[min_idx[1]]
best_policy_avg_cap_pay_w_spot = capacity_payments_fine3[min_idx[1]] * interp_inv_price_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_baseline = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[0,0]
emissions_best = interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]
emissions_frac_w_spot = (emissions_best - emissions_baseline) / emissions_baseline
w_w_spot = (E_W[min_idx] - E_W_compare) / num_people
cs_w_spot = (interp_CS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - CS_compare) / num_people
ps_w_spot = (interp_PS_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - PS_compare) / num_people
t_w_spot = (interp_T_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - T_compare) / num_people
c_w_spot = (interp_C_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - C_compare) / num_people
emissions_w_spot = (interp_emissions_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx] - emissions_compare) / 1000000.0
blackouts_w_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / 1000000.0
blackouts_frac_w_spot = (np.maximum(0.0, interp_blackouts_spot(carbon_taxes_fine, capacity_payments_fine3)[min_idx]) - blackouts_compare) / blackouts_compare

format_pct = lambda x: "{:,}\\%".format(int(np.round(x * 100.0))).replace(",","\\,") if not np.isclose(x, 0.0) else "0\\%"

# Create table
to_tex = "\\begin{tabular}{rcccccccccc} \n"
to_tex += "\\hline \\\\ \n"
to_tex += " & &  & & $\\max$ & &$\\max$  & & $\\max$ & &  \\\\ \n"
to_tex += " & & $\\max$ & & $-\\Delta\\text{E}$ s.t. & & $-\\Delta\\text{E}$ s.t. & & $\\Delta\\text{CS} + \\Delta\\text{T} + \\Delta\\text{C}$ & &  \\\\ \n"
to_tex += " & & $-\\Delta\\text{E}$ s.t. & & $\\Delta\\text{CS} + \\Delta\\text{C}$ & & $\\Delta\\text{CS} + \\Delta\\text{T} + \\Delta\\text{C}$ & &  $-SCC \\times \\Delta\\text{E}$ & & $\\max$ \\\\ \n"
to_tex += " & & $\\Delta\\text{CS} \\geq 0$ & & $-VOLL \\times \\Delta\\text{B} \\geq 0$ & & $-VOLL \\times \\Delta\\text{B} \\geq 0$ & & $-VOLL \\times \\Delta\\text{B}$ & & $\\Delta\\mathcal{W}$ \\\\ \n"
to_tex += " \\cline{3-3} \\cline{5-5} \\cline{7-7} \\cline{9-9} \\cline{11-11} \n"
to_tex += " & & & & & & & & & & \\\\ \n"
to_tex += " & & \\multicolumn{9}{c}{$\\kappa_{\\text{coal}} = \\kappa_{\\text{gas}}$} \\\\ \n"
to_tex += " \\cline{3-11} \n"

to_tex += "policy"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_cs)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_cstax)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_csg)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_wcs)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_w)}$,"
to_tex += " \\\\ \n"

to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}} = {format_cap_pay(best_payment_cs)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}} = {format_cap_pay(best_payment_cstax)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}} = {format_cap_pay(best_payment_csg)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}} = {format_cap_pay(best_payment_wcs)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}} = {format_cap_pay(best_payment_w)}$"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{CS}$"
to_tex += " & "
to_tex += " & " + format_W(cs_cs)
to_tex += " & "
to_tex += " & " + format_W(cs_cstax)
to_tex += " & "
to_tex += " & " + format_W(cs_csg)
to_tex += " & "
to_tex += " & " + format_W(cs_wcs)
to_tex += " & "
to_tex += " & " + format_W(cs_w)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{PS}$"
to_tex += " & "
to_tex += " & " + format_W(ps_cs)
to_tex += " & "
to_tex += " & " + format_W(ps_cstax)
to_tex += " & "
to_tex += " & " + format_W(ps_csg)
to_tex += " & "
to_tex += " & " + format_W(ps_wcs)
to_tex += " & "
to_tex += " & " + format_W(ps_w)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{T}$"
to_tex += " & "
to_tex += " & " + format_W(t_cs)
to_tex += " & "
to_tex += " & " + format_W(t_cstax)
to_tex += " & "
to_tex += " & " + format_W(t_csg)
to_tex += " & "
to_tex += " & " + format_W(t_wcs)
to_tex += " & "
to_tex += " & " + format_W(t_w)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{C}$"
to_tex += " & "
to_tex += " & " + format_W(c_cs)
to_tex += " & "
to_tex += " & " + format_W(c_cstax)
to_tex += " & "
to_tex += " & " + format_W(c_csg)
to_tex += " & "
to_tex += " & " + format_W(c_wcs)
to_tex += " & "
to_tex += " & " + format_W(c_w)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{E}$"
to_tex += " & "
to_tex += " & " + format_W(emissions_cs) + f" ({format_pct(emissions_frac_cs)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_cstax) + f" ({format_pct(emissions_frac_cstax)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_csg) + f" ({format_pct(emissions_frac_csg)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_wcs) + f" ({format_pct(emissions_frac_wcs)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_w) + f" ({format_pct(emissions_frac_w)})"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{B}$"
to_tex += " & "
to_tex += " & " + format_W(blackouts_cs) + f" ({format_pct(blackouts_frac_cs)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_cstax) + f" ({format_pct(blackouts_frac_cstax)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_csg) + f" ({format_pct(blackouts_frac_csg)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_wcs) + f" ({format_pct(blackouts_frac_wcs)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_w) + f" ({format_pct(blackouts_frac_w)})"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\mathcal{W}$"
to_tex += " & "
to_tex += " & " + format_W(w_cs)
to_tex += " & "
to_tex += " & " + format_W(w_cstax)
to_tex += " & "
to_tex += " & " + format_W(w_csg)
to_tex += " & "
to_tex += " & " + format_W(w_wcs)
to_tex += " & "
to_tex += " & " + format_W(w_w)
to_tex += " \\\\ \n"

to_tex += " & & & & & & & & \\\\ \n"

to_tex += " & & \\multicolumn{9}{c}{separate $\\kappa_{\\text{coal}}$, $\\kappa_{\\text{gas}}$} \\\\ \n"
to_tex += " \\cline{3-11} \n"

to_tex += "policy"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_cs_sep)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_cstax_sep)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_csg_sep)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_wcs_sep)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_w_sep)}$,"
to_tex += " \\\\ \n"

to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{coal}}}} = {format_cap_pay(best_payment1_cs_sep)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{coal}}}} = {format_cap_pay(best_payment1_cstax_sep)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{coal}}}} = {format_cap_pay(best_payment1_csg_sep)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{coal}}}} = {format_cap_pay(best_payment1_wcs_sep)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{coal}}}} = {format_cap_pay(best_payment1_w_sep)}$"
to_tex += " \\\\ \n"

to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{gas}}}} = {format_cap_pay(best_payment2_cs_sep)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{gas}}}} = {format_cap_pay(best_payment2_cstax_sep)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{gas}}}} = {format_cap_pay(best_payment2_csg_sep)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{gas}}}} = {format_cap_pay(best_payment2_wcs_sep)}$"
to_tex += " & "
to_tex += " & " + f"$\\kappa^{{*}}_{{\\text{{gas}}}} = {format_cap_pay(best_payment2_w_sep)}$"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{CS}$"
to_tex += " & "
to_tex += " & " + format_W(cs_cs_sep)
to_tex += " & "
to_tex += " & " + format_W(cs_cstax_sep)
to_tex += " & "
to_tex += " & " + format_W(cs_csg_sep)
to_tex += " & "
to_tex += " & " + format_W(cs_wcs_sep)
to_tex += " & "
to_tex += " & " + format_W(cs_w_sep)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{PS}$"
to_tex += " & "
to_tex += " & " + format_W(ps_cs_sep)
to_tex += " & "
to_tex += " & " + format_W(ps_cstax_sep)
to_tex += " & "
to_tex += " & " + format_W(ps_csg_sep)
to_tex += " & "
to_tex += " & " + format_W(ps_wcs_sep)
to_tex += " & "
to_tex += " & " + format_W(ps_w_sep)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{T}$"
to_tex += " & "
to_tex += " & " + format_W(t_cs_sep)
to_tex += " & "
to_tex += " & " + format_W(t_cstax_sep)
to_tex += " & "
to_tex += " & " + format_W(t_csg_sep)
to_tex += " & "
to_tex += " & " + format_W(t_wcs_sep)
to_tex += " & "
to_tex += " & " + format_W(t_w_sep)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{C}$"
to_tex += " & "
to_tex += " & " + format_W(c_cs_sep)
to_tex += " & "
to_tex += " & " + format_W(c_cstax_sep)
to_tex += " & "
to_tex += " & " + format_W(c_csg_sep)
to_tex += " & "
to_tex += " & " + format_W(c_wcs_sep)
to_tex += " & "
to_tex += " & " + format_W(c_w_sep)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{E}$"
to_tex += " & "
to_tex += " & " + format_W(emissions_cs_sep) + f" ({format_pct(emissions_frac_cs_sep)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_cstax_sep) + f" ({format_pct(emissions_frac_cstax_sep)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_csg_sep) + f" ({format_pct(emissions_frac_csg_sep)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_wcs_sep) + f" ({format_pct(emissions_frac_wcs_sep)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_w_sep) + f" ({format_pct(emissions_frac_w_sep)})"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{B}$"
to_tex += " & "
to_tex += " & " + format_W(blackouts_cs_sep) + f" ({format_pct(blackouts_frac_cs_sep)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_cstax_sep) + f" ({format_pct(blackouts_frac_cstax_sep)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_csg_sep) + f" ({format_pct(blackouts_frac_csg_sep)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_wcs_sep) + f" ({format_pct(blackouts_frac_wcs_sep)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_w_sep) + f" ({format_pct(blackouts_frac_w_sep)})"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\mathcal{W}$"
to_tex += " & "
to_tex += " & " + format_W(w_cs_sep)
to_tex += " & "
to_tex += " & " + format_W(w_cstax_sep)
to_tex += " & "
to_tex += " & " + format_W(w_csg_sep)
to_tex += " & "
to_tex += " & " + format_W(w_wcs_sep)
to_tex += " & "
to_tex += " & " + format_W(w_w_sep)
to_tex += " \\\\ \n"

to_tex += " & & & & & & & & \\\\ \n"

to_tex += " & & \\multicolumn{9}{c}{$\\kappa = \\alpha / P_{t}^{avg}$} \\\\ \n"
to_tex += " \\cline{3-11} \n"

format_alpha = lambda x: "{:,}".format(int(np.round(x / 100000.0) * 100000.0)).replace(",","\\,")

to_tex += "policy"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_cs_spot)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_cstax_spot)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_csg_spot)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_wcs_spot)}$,"
to_tex += " & "
to_tex += " & " + f"$\\tau^{{*}} = {format_carbon_tax(best_tax_w_spot)}$,"
to_tex += " \\\\ \n"

to_tex += " & "
to_tex += " & " + f"$\\alpha^{{*}} = {format_alpha(best_payment_cs_spot)}$"
to_tex += " & "
to_tex += " & " + f"$\\alpha^{{*}} = {format_alpha(best_payment_cstax_spot)}$"
to_tex += " & "
to_tex += " & " + f"$\\alpha^{{*}} = {format_alpha(best_payment_csg_spot)}$"
to_tex += " & "
to_tex += " & " + f"$\\alpha^{{*}} = {format_alpha(best_payment_wcs_spot)}$"
to_tex += " & "
to_tex += " & " + f"$\\alpha^{{*}} = {format_alpha(best_payment_w_spot)}$"
to_tex += " \\\\ \n"

to_tex += " & "
to_tex += " & " + f"($\\bar{{\\kappa}} = {format_cap_pay(best_policy_avg_cap_pay_cs_spot)}$)"
to_tex += " & "
to_tex += " & " + f"($\\bar{{\\kappa}} = {format_cap_pay(best_policy_avg_cap_pay_cstax_spot)}$)"
to_tex += " & "
to_tex += " & " + f"($\\bar{{\\kappa}} = {format_cap_pay(best_policy_avg_cap_pay_csg_spot)}$)"
to_tex += " & "
to_tex += " & " + f"($\\bar{{\\kappa}} = {format_cap_pay(best_policy_avg_cap_pay_wcs_spot)}$)"
to_tex += " & "
to_tex += " & " + f"($\\bar{{\\kappa}} = {format_cap_pay(best_policy_avg_cap_pay_w_spot)}$)"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{CS}$"
to_tex += " & "
to_tex += " & " + format_W(cs_cs_spot)
to_tex += " & "
to_tex += " & " + format_W(cs_cstax_spot)
to_tex += " & "
to_tex += " & " + format_W(cs_csg_spot)
to_tex += " & "
to_tex += " & " + format_W(cs_wcs_spot)
to_tex += " & "
to_tex += " & " + format_W(cs_w_spot)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{PS}$"
to_tex += " & "
to_tex += " & " + format_W(ps_cs_spot)
to_tex += " & "
to_tex += " & " + format_W(ps_cstax_spot)
to_tex += " & "
to_tex += " & " + format_W(ps_csg_spot)
to_tex += " & "
to_tex += " & " + format_W(ps_wcs_spot)
to_tex += " & "
to_tex += " & " + format_W(ps_w_spot)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{T}$"
to_tex += " & "
to_tex += " & " + format_W(t_cs_spot)
to_tex += " & "
to_tex += " & " + format_W(t_cstax_spot)
to_tex += " & "
to_tex += " & " + format_W(t_csg_spot)
to_tex += " & "
to_tex += " & " + format_W(t_wcs_spot)
to_tex += " & "
to_tex += " & " + format_W(t_w_spot)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{C}$"
to_tex += " & "
to_tex += " & " + format_W(c_cs_spot)
to_tex += " & "
to_tex += " & " + format_W(c_cstax_spot)
to_tex += " & "
to_tex += " & " + format_W(c_csg_spot)
to_tex += " & "
to_tex += " & " + format_W(c_wcs_spot)
to_tex += " & "
to_tex += " & " + format_W(c_w_spot)
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{E}$"
to_tex += " & "
to_tex += " & " + format_W(emissions_cs_spot) + f" ({format_pct(emissions_frac_cs_spot)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_cstax_spot) + f" ({format_pct(emissions_frac_cstax_spot)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_csg_spot) + f" ({format_pct(emissions_frac_csg_spot)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_wcs_spot) + f" ({format_pct(emissions_frac_wcs_spot)})"
to_tex += " & "
to_tex += " & " + format_W(emissions_w_spot) + f" ({format_pct(emissions_frac_w_spot)})"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\text{B}$"
to_tex += " & "
to_tex += " & " + format_W(blackouts_cs_spot) + f" ({format_pct(blackouts_frac_cs_spot)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_cstax_spot) + f" ({format_pct(blackouts_frac_cstax_spot)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_csg_spot) + f" ({format_pct(blackouts_frac_csg_spot)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_wcs_spot) + f" ({format_pct(blackouts_frac_wcs_spot)})"
to_tex += " & "
to_tex += " & " + format_W(blackouts_w_spot) + f" ({format_pct(blackouts_frac_w_spot)})"
to_tex += " \\\\ \n"

to_tex += " $\\Delta \\mathcal{W}$"
to_tex += " & "
to_tex += " & " + format_W(w_cs_spot)
to_tex += " & "
to_tex += " & " + format_W(w_cstax_spot)
to_tex += " & "
to_tex += " & " + format_W(w_csg_spot)
to_tex += " & "
to_tex += " & " + format_W(w_wcs_spot)
to_tex += " & "
to_tex += " & " + format_W(w_w_spot)
to_tex += " \\\\ \n"

to_tex += " & & & & & & & & \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

if save_output:
    create_file(gv.tables_path + "alternative_policies.tex", to_tex)

if show_output:
    print(to_tex)

# Save values from above exericse

if save_output:
    create_file(gv.stats_path + "emissions_frac_reduction_cs_spot.tex", format_pct(-emissions_frac_cs_spot))
    create_file(gv.stats_path + "sep_welfare_frac_increase.tex", f"{int(np.round((w_w_sep - w_w) / w_w * 100.0))}")

# %%
# Carbon tax delay counterfactuals

# Capacity (paper version)
tax_idx = np.array([4, 4, 4, 4])
delay_idx = np.array([0, 3, 6, 9]) #np.array([0, 2, 4, 6])
indices_use = np.ravel_multi_index(np.vstack((tax_idx, delay_idx)), (carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
labels = [f"years delayed = {delay}" for delay in delay_linspace[delay_idx]]
ls = [ls_all[i] for i in range(delay_idx.shape[0])]
#colors = ["black" for tau in delay_linspace[delay_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_delay.pdf"
plot_capacities(np.reshape(delay_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
ls = [ls_all[0] for i in range(delay_idx.shape[0])]
cmap = cm.get_cmap("Purples")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, delay_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_delay_presentation.pdf"
plot_capacities(np.reshape(delay_results['expected_agg_source_capacity'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
ls = [ls_all[i] for i in range(delay_idx.shape[0])]
#colors = ["black" for tau in delay_linspace[delay_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}production_delay.pdf"
plot_production(np.reshape(delay_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
ls = [ls_all[0] for i in range(delay_idx.shape[0])]
cmap = cm.get_cmap("Purples")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, delay_idx.shape[0])]
filename = f"{gv.graphs_path}production_delay_presentation.pdf"
plot_production(np.reshape(delay_results['expected_frac_by_source'], (-1, num_years, sources.shape[0])), indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
tax_idx = np.array([0, 2, 4, 6])
cspsg = delay_results['expected_product_market_sum'][tax_idx,:].T / 1000000000.0
emissions = ton_to_kg(delay_results['expected_emissions_sum'][tax_idx,:].T) / 1000000000.0
blackouts = delay_results['expected_blackouts_sum'][tax_idx,:].T / 1000000.0
ls = [ls_all[i] for i in range(tax_idx.shape[0])]
#colors = ["black" for tau in tax_idx]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
x_axis_labels = [f"years policy delayed" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"$\\tau = $A$\\${int(tau)}$/ton" for tau in carbon_taxes_linspace[tax_idx]]
filename = f"{gv.graphs_path}welfare_delay.pdf"
plot_welfare(cspsg, emissions, blackouts, delay_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename, labels=labels)

# Welfare (presentation version)
ls = [ls_all[0] for i in range(tax_idx.shape[0])]
cmap = cm.get_cmap("Purples")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}welfare_delay_presentation.pdf"
plot_welfare(cspsg, emissions, blackouts, delay_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# \Delta Prices and \Delta CS over time
delay_idx = np.array([1, 5, 9]) #np.array([0, 1, 2])
tax_idx_use = 4
create_file(gv.stats_path + "counterfactuals_delay_graph_carbon_tax.tex", f"{int(carbon_taxes_linspace[tax_idx_use]):,}".replace(",","\\,"))
ls = [ls_all[i] for i in range(delay_idx.shape[0])]
#colors = ["black" for i in range(delay_idx.shape[0])]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
labels = [f"years delayed = {delay}" for delay in delay_linspace[delay_idx]]
titles = [f"$\\Delta$ avg. $P_{{h}}$", f"$\\Delta$ CS${{}}_{{t}}$"]
x_labels = [f"year", f"year"]
y_labels = [f"A\\$ / MWh", f"A\\$ (billions)"]
delta_p = delay_results['expected_quantity_weighted_avg_price'][tax_idx_use,:,:][delay_idx,:] - delay_results['expected_quantity_weighted_avg_price'][0,0,:][np.newaxis,:]
delta_cs = (delay_results['expected_consumer_surplus'][tax_idx_use,:,:][delay_idx,:] - delay_results['expected_consumer_surplus'][0,0,:][np.newaxis,:]) / 1000000000.0
delta_production_cost = (delay_results['expected_total_production_cost'][tax_idx_use,:,:][delay_idx,:] - delay_results['expected_total_production_cost'][0,0,:][np.newaxis,:]) / 1000.0
var_arr = np.concatenate((delta_p[np.newaxis,:,:], delta_cs[np.newaxis,:,:]), axis=0)
filename = f"{gv.graphs_path}cs_delay.pdf"
plot_general_over_time(var_arr, indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, titles, x_labels, y_labels, filename)

# \Delta Prices and \Delta CS over time (presentation)
cmap = cm.get_cmap("Purples")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}cs_delay_presentation.pdf"
plot_general_over_time(var_arr, indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, titles, x_labels, y_labels, filename)

# Welfare table
tax_idx = np.array([2, 4, 6])
delay_idx = np.array([1, 5, 9])
for table_idx, (delay_idx_baseline, table_name, key_name_suffix) in enumerate(zip([0, 6], ["", "_later"], ["", "_extra"])):
    to_tex = "\\begin{tabular}{ccccccccccccccccc} \n"
    to_tex += "\\hline \n"
    to_tex += " & & & \\multicolumn{2}{c}{$\\Delta \\text{CS}$} & & \\multicolumn{2}{c}{$\\Delta \\text{PS}$} & & \\multicolumn{2}{c}{$\\Delta \\text{G}$} & & \\multicolumn{2}{c}{$\\Delta$ emissions} & & \\multicolumn{2}{c}{$\\Delta$ blackouts} \\\\ \n"
    to_tex += " & & &  \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billions A\\$)} & & \\multicolumn{2}{c}{(billons A\\$)} & & \\multicolumn{2}{c}{(billions kg $\\text{CO}_{2}$-eq)} & & \\multicolumn{2}{c}{(millions MWh)} \\\\ \n"
    # to_tex += " & & & \\multicolumn{2}{c}{baseline: --} & & \\multicolumn{2}{c}{baseline: " + format_ps(delay_results['expected_producer_surplus_sum'][0,0]) + "} & & \\multicolumn{2}{c}{baseline: " + format_g(delay_results['expected_revenue_sum'][0,0]) + "} & & \\multicolumn{2}{c}{baseline: " + format_emissions(delay_results['expected_emissions_sum'][0,0]) + "} & & \\multicolumn{2}{c}{baseline: " + format_blackouts(delay_results['expected_blackouts_sum'][0,0]) + "} \\\\ \n"
    to_tex += f" &  & & from & from no & & from & from no & & from & from no & & from & from no & & from & from no \\\\ \n"
    to_tex += f"$\\tau$ & delay & & baseline & delay & & baseline & delay & & baseline & delay & & baseline & delay & & baseline & delay \\\\ \n"
    to_tex += "\\cline{1-1} \\cline{2-2} \\cline{4-5} \\cline{7-8} \\cline{10-11} \\cline{13-14} \\cline{16-17} \n"
    to_tex += " & & & & & & & & & & & \\\\ \n"
    for i in tax_idx:
        to_tex += format_carbon_tax(carbon_taxes_linspace[i])
        for j in delay_idx:
            to_tex += " & " + f"{delay_linspace[j]}"
            to_tex += " & "
            to_tex += " & " + format_cs(delay_results['expected_consumer_surplus_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_consumer_surplus_sum' + key_name_suffix][0,0])
            to_tex += " & " + format_cs(delay_results['expected_consumer_surplus_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_consumer_surplus_sum' + key_name_suffix][i,delay_idx_baseline])
            to_tex += " & "
            to_tex += " & " + format_ps(delay_results['expected_producer_surplus_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_producer_surplus_sum' + key_name_suffix][0,0])
            to_tex += " & " + format_ps(delay_results['expected_producer_surplus_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_producer_surplus_sum' + key_name_suffix][i,delay_idx_baseline])
            to_tex += " & "
            to_tex += " & " + format_g(delay_results['expected_revenue_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_revenue_sum' + key_name_suffix][0,0])
            to_tex += " & " + format_g(delay_results['expected_revenue_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_revenue_sum' + key_name_suffix][i,delay_idx_baseline])
            to_tex += " & "
            to_tex += " & " + format_emissions(delay_results['expected_emissions_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_emissions_sum' + key_name_suffix][0,0])
            to_tex += " & " + format_emissions(delay_results['expected_emissions_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_emissions_sum' + key_name_suffix][i,delay_idx_baseline])
            to_tex += " & "
            to_tex += " & " + format_blackouts(delay_results['expected_blackouts_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_blackouts_sum' + key_name_suffix][0,0])
            to_tex += " & " + format_blackouts(delay_results['expected_blackouts_sum' + key_name_suffix][i,j + delay_idx_baseline] - delay_results['expected_blackouts_sum' + key_name_suffix][i,delay_idx_baseline])
            to_tex += " \\\\ \n"
            if (i != tax_idx[-1]) and (j == delay_idx[-1]):
                to_tex += " & & & & & & & & & & & \\\\ \n"
    
    to_tex += "\\hline \n"
    to_tex += "\\end{tabular}"
    if show_output:
        print(to_tex)
    if save_output:
        create_file(gv.tables_path + f"delay_welfare_table{table_name}.tex", to_tex)

# %%
# Optimal carbon taxes / capacity payments

# Set up the space would like to consider
min_carbon_tax = 0.0
max_carbon_tax = 300.0
num_points = 500 # need large number
voll = 10000.0
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points)

interp_CSPS = [interp.CubicSpline(carbon_taxes_linspace, delay_results['expected_product_market_sum'][:,i]) for i in range(delay_linspace.shape[0])]
interp_emissions = [interp.CubicSpline(carbon_taxes_linspace, delay_results['expected_emissions_sum'][:,i]) for i in range(delay_linspace.shape[0])]
interp_blackouts = [interp.CubicSpline(carbon_taxes_linspace, delay_results['expected_blackouts_sum'][:,i]) for i in range(delay_linspace.shape[0])]
carbon_taxes_fine = np.linspace(np.min(carbon_taxes_linspace), np.max(carbon_taxes_linspace), 1000)
capacity_payments_fine = np.linspace(np.min(capacity_payments_linspace), np.max(capacity_payments_linspace), 1000)

scc_vals = np.linspace(0.0, 300.0, 1000)

# E_W
E_W = np.concatenate(tuple([(interp_CSPS[i](carbon_taxes_fine)[np.newaxis,:] - scc_vals[:,np.newaxis] * interp_emissions[i](carbon_taxes_fine)[np.newaxis,:] - voll * interp_blackouts[i](carbon_taxes_fine)[np.newaxis,:])[:,:,np.newaxis] for i in range(delay_linspace.shape[0])]), axis=2)
E_W_compare = E_W[:,0,0]

# w/o delay
max_policy = np.argmax(E_W[:,:,0], axis=1) # delay = 0
carbon_tax_no_delay = carbon_taxes_fine[max_policy]
max_w_carbon_tax_no_delay = (np.max(E_W[:,:,0], axis=1) - E_W_compare) / num_people

# w/ delay
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_delay = carbon_taxes_fine[max_policy_1]
delay_max = delay_linspace[max_policy_2]
max_w_delay = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people

# format_delay = lambda x: f"{x}"

# to_tex = "\\begin{tabular}{cccccc} \n"
# to_tex += "\\hline \\\\ \n"
# to_tex += "\\multicolumn{2}{c}{carbon tax, no delay} & & \\multicolumn{3}{c}{carbon tax, delay} \\\\ \n"
# to_tex += " \\cline{1-2} \\cline{4-6} \n"
# to_tex += "$\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\text{delay}^{*}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
# to_tex += "\\hline \n"

# for i, scc_val in enumerate(scc_vals):
#     to_tex += format_carbon_tax(carbon_tax_no_delay[i])
#     to_tex += " & " + format_W(max_w_carbon_tax_no_delay[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_delay[i])
#     to_tex += " & " + format_cap_pay(delay_max[i])
#     to_tex += " & " + format_W(max_w_delay[i])
#     to_tex += " \\\\ \n"

# to_tex += "\\hline \n"
# to_tex += "\\end{tabular}"

# print(to_tex)

# create_file(gv.tables_path + "optimal_policy_tax_delay.tex", to_tex)

max_scc_above_0 = np.max(scc_vals[delay_max > 0])
if show_output:
    print(f"Maximum SCC the optimal delay is > 0: {max_scc_above_0}")
if save_output:
    create_file(gv.stats_path + "max_optimal_delay_above_0.tex", f"{int(np.round(max_scc_above_0))}")

# %%
# Renewable production subsidy counterfactuals

# Capacity (paper version)
subsidy_idx = np.array([0, 2, 4, 6])
indices_use = subsidy_idx
labels = [f"$\\varsigma = $A$\\${int(sigma)}$/MWh" for sigma in renewable_subsidies_linspace[subsidy_idx]]
ls = [ls_all[i] for i in range(subsidy_idx.shape[0])]
#colors = ["black" for sigma in renewable_subsidies_linspace[subsidy_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_renewable_production_subsidy.pdf"
plot_capacities(renewable_production_subsidies_results['expected_agg_source_capacity'], indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
ls = [ls_all[0] for i in range(subsidy_idx.shape[0])]
cmap = cm.get_cmap("Greens")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_renewable_production_subsidy_presentation.pdf"
plot_capacities(renewable_production_subsidies_results['expected_agg_source_capacity'], indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
ls = [ls_all[i] for i in range(subsidy_idx.shape[0])]
#colors = ["black" for sigma in renewable_subsidies_linspace[subsidy_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}production_renewable_production_subsidy.pdf"
plot_production(renewable_production_subsidies_results['expected_frac_by_source'], indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
ls = [ls_all[0] for i in range(subsidy_idx.shape[0])]
cmap = cm.get_cmap("Greens")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}production_renewable_production_subsidy_presentation.pdf"
plot_production(renewable_production_subsidies_results['expected_frac_by_source'], indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
ls = [ls_all[i] for i in range(subsidy_idx.shape[0])]
#colors = ["black" for sigma in renewable_subsidies_linspace[subsidy_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_renewable_production_subsidy.pdf"
plot_combined_capacities_production(renewable_production_subsidies_results['expected_agg_source_capacity'], renewable_production_subsidies_results['expected_frac_by_source'], indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
ls = [ls_all[0] for i in range(subsidy_idx.shape[0])]
cmap = cm.get_cmap("Greens")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_renewable_production_subsidy_presentation.pdf"
plot_combined_capacities_production(renewable_production_subsidies_results['expected_agg_source_capacity'], renewable_production_subsidies_results['expected_frac_by_source'], indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
indices_use = np.arange(renewable_subsidies_linspace.shape[0])
cspsg = renewable_production_subsidies_results['expected_product_market_sum'][indices_use] / 1000000000.0
emissions = ton_to_kg(renewable_production_subsidies_results['expected_emissions_sum'][indices_use]) / 1000000000.0
blackouts = renewable_production_subsidies_results['expected_blackouts_sum'][indices_use] / 1000000.0
ls = ["solid" for i in range(3)]
colors = ["black" for i in range(3)]
x_axis_labels = [f"$\\varsigma$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
filename = f"{gv.graphs_path}welfare_renewable_production_subsidy.pdf"
plot_welfare(cspsg[:-1], emissions[:-1], blackouts[:-1], renewable_subsidies_linspace[:-1], x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_renewable_production_subsidy_presentation.pdf"
colors = [cm.get_cmap("Greens")(0.75) for i in range(3)]
plot_welfare(cspsg[:-1], emissions[:-1], blackouts[:-1], renewable_subsidies_linspace[:-1], x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# %%
# Renewable investment subsidy counterfactuals

# Capacity (paper version)
subsidy_idx = np.array([0, 3, 6, 9])
indices_use = subsidy_idx
labels = [f"$s = ${int(sigma * 100.0)}%" for sigma in renewable_investment_subsidy_linspace[subsidy_idx]]
ls = [ls_all[i] for i in range(subsidy_idx.shape[0])]
#colors = ["black" for sigma in renewable_investment_subsidy_linspace[subsidy_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_renewable_investment_subsidy.pdf"
plot_capacities(renewable_investment_subsidies_results['expected_agg_source_capacity'], indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
ls = [ls_all[0] for i in range(subsidy_idx.shape[0])]
cmap = cm.get_cmap("Oranges")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_renewable_investment_subsidy_presentation.pdf"
plot_capacities(renewable_investment_subsidies_results['expected_agg_source_capacity'], indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
ls = [ls_all[i] for i in range(subsidy_idx.shape[0])]
#colors = ["black" for sigma in renewable_investment_subsidy_linspace[subsidy_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}production_renewable_investment_subsidy.pdf"
plot_production(renewable_investment_subsidies_results['expected_frac_by_source'], indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
ls = [ls_all[0] for i in range(subsidy_idx.shape[0])]
cmap = cm.get_cmap("Oranges")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}production_renewable_investment_subsidy_presentation.pdf"
plot_production(renewable_investment_subsidies_results['expected_frac_by_source'], indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
ls = [ls_all[i] for i in range(subsidy_idx.shape[0])]
#colors = ["black" for sigma in renewable_investment_subsidy_linspace[subsidy_idx]]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_renewable_investment_subsidy.pdf"
plot_combined_capacities_production(renewable_investment_subsidies_results['expected_agg_source_capacity'], renewable_investment_subsidies_results['expected_frac_by_source'], indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
ls = [ls_all[0] for i in range(subsidy_idx.shape[0])]
cmap = cm.get_cmap("Oranges")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, subsidy_idx.shape[0])]
filename = f"{gv.graphs_path}capacity_production_renewable_investment_subsidy_presentation.pdf"
plot_combined_capacities_production(renewable_investment_subsidies_results['expected_agg_source_capacity'], renewable_investment_subsidies_results['expected_frac_by_source'], indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
indices_use = np.arange(renewable_investment_subsidy_linspace.shape[0])
cspsg = renewable_investment_subsidies_results['expected_product_market_sum'][indices_use] / 1000000000.0
emissions = ton_to_kg(renewable_investment_subsidies_results['expected_emissions_sum'][indices_use]) / 1000000000.0
blackouts = renewable_investment_subsidies_results['expected_blackouts_sum'][indices_use] / 1000000.0
ls = ["solid" for i in range(3)]
colors = ["black" for i in range(3)]
x_axis_labels = [f"$s$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
filename = f"{gv.graphs_path}welfare_renewable_investment_subsidy.pdf"
plot_welfare(cspsg, emissions, blackouts, renewable_investment_subsidy_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_renewable_investment_subsidy_presentation.pdf"
colors = [cm.get_cmap("Oranges")(0.75) for i in range(3)]
plot_welfare(cspsg, emissions, blackouts, renewable_investment_subsidy_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# %%
# Compare environmental policies

to_tex = "\\setlength\\extrarowheight{0pt} \n"
to_tex += "\\begin{tabular}{@{\\extracolsep{0pt}}cccccccc@{}} \n"
to_tex += "\\hline \n"
to_tex += "$\\Delta \\text{emissions}$ & &  & $\\Delta \\text{blackouts}$ & $\\Delta \\text{CS}$ & $\\Delta \\text{PS}$ & $\\Delta \\text{G}$ & $\\Delta \\left(\\text{CS} + \\text{PS} + \\text{G}\\right)$ \\\\ \n"
to_tex += "(billions kg$\\text{CO}_{2}$-eq) & policy & policy value & (millions MWh) & (billions A\\$) & (billions A\\$) & (billions A\\$) & (billions A\\$) \\\\ \n"
to_tex += "\\hline \n"

emissions_array = np.linspace(0.0, 35.0, 1000)#np.linspace(0.0, 25.0, 6)
emissions_array_coarse = np.linspace(0.0, 35.0, 8)#np.linspace(0.0, 25.0, 6)
policy_predict = {} 
blackouts_predict = {}
CS_predict = {}
PS_predict = {}
G_predict = {}
W_predict = {}
policies = {
    "carbon tax": carbon_taxes_linspace, 
    "renew. prod. subs.": renewable_subsidies_linspace[:-1], 
    "renew. inv. subs.": renewable_investment_subsidy_linspace * 100.0
}
results_dict = {
    "carbon tax": carbon_tax_capacity_payment_results, 
    "renew. prod. subs.": renewable_production_subsidies_results, 
    "renew. inv. subs.": renewable_investment_subsidies_results
}
cap_pay_idx = np.argmin(np.abs(capacity_payments_linspace - 100000.0))
indices = {
    "carbon tax": np.ravel_multi_index(np.vstack((np.arange(carbon_taxes_linspace.shape[0]), np.ones(carbon_taxes_linspace.shape[0], dtype=int) * cap_pay_idx)), (carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0])), 
    "renew. prod. subs.": np.arange(renewable_subsidies_linspace.shape[0])[:-1], 
    "renew. inv. subs.": np.arange(renewable_investment_subsidy_linspace.shape[0])
}
# indices['carbon tax'] = indices['carbon tax'][:-1] # doing this b/c final value results in small increase in emissions, so can't use, this is a workaround since that region not relevant for this table
# policies['carbon tax'] = policies['carbon tax'][:-1]
indices['renew. inv. subs.'] = indices['renew. inv. subs.'][:-3] # doing this b/c final value results in small increase in emissions, so can't use, this is a workaround since that region not relevant for this table
policies['renew. inv. subs.'] = policies['renew. inv. subs.'][:-3]

def predict_welfare(results_dict, indices, policy_vars, emissions_array):
    # Predict policy values based on emissions
    cs_policy = interp.CubicSpline(-(ton_to_kg(results_dict['expected_emissions_sum'].flatten()[indices]) - ton_to_kg(results_dict['expected_emissions_sum'].flatten()[indices][0])) / 1.0e9, policy_vars)
    policy_predict = cs_policy(emissions_array)
    
    # Predict other welfare variables based on predicted policy
    cs_blackouts = interp.CubicSpline(policy_vars, (results_dict['expected_blackouts_sum'].flatten()[indices] - results_dict['expected_blackouts_sum'].flatten()[indices][0]) / 1.0e6)
    cs_CS = interp.CubicSpline(policy_vars, (results_dict['expected_consumer_surplus_sum'].flatten()[indices] - results_dict['expected_consumer_surplus_sum'].flatten()[indices][0]) / 1.0e9)
    cs_PS = interp.CubicSpline(policy_vars, (results_dict['expected_producer_surplus_sum'].flatten()[indices] - results_dict['expected_producer_surplus_sum'].flatten()[indices][0]) / 1.0e9)
    cs_G = interp.CubicSpline(policy_vars, (results_dict['expected_revenue_sum'].flatten()[indices] - results_dict['expected_revenue_sum'].flatten()[indices][0]) / 1.0e9)
    blackouts_predict = cs_blackouts(policy_predict)
    CS_predict = cs_CS(policy_predict)
    PS_predict = cs_PS(policy_predict)
    G_predict = cs_G(policy_predict)
    W_predict = CS_predict + PS_predict + G_predict
    
    return policy_predict, blackouts_predict, CS_predict, PS_predict, G_predict, W_predict

for i, policy in enumerate(policies.keys()):
    res = predict_welfare(results_dict[policy], indices[policy], policies[policy], emissions_array)
    policy_predict[policy] = res[0]
    blackouts_predict[policy] = res[1]
    CS_predict[policy] = res[2]
    PS_predict[policy] = res[3]
    G_predict[policy] = res[4]
    W_predict[policy] = res[5]
    
    unusable = policy_predict[policy] > policies[policy][-1]
    policy_predict[policy][unusable] = np.nan
    blackouts_predict[policy][unusable] = np.nan
    CS_predict[policy][unusable] = np.nan
    PS_predict[policy][unusable] = np.nan
    G_predict[policy][unusable] = np.nan
    W_predict[policy][unusable] = np.nan
    
format_str = lambda x, round_dig: str(np.round(x, round_dig)) if not np.isnan(x) else "-"
    
for delta_emissions in emissions_array_coarse:
    to_tex += f"{int(np.round(delta_emissions))}"
    i = np.argmin((emissions_array - delta_emissions)**2.0)
    for policy in policies.keys():
        to_tex += f" & {policy} & {format_str(policy_predict[policy][i], 1)} & {format_str(blackouts_predict[policy][i], 1)} & {format_str(CS_predict[policy][i], 1)} & {format_str(PS_predict[policy][i], 1)} & {format_str(G_predict[policy][i], 1)} & {format_str(W_predict[policy][i], 1)} \\\\ \n"
    if i < emissions_array.shape[0] - 1:
        to_tex += " & & & & & & & \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

if save_output:
    create_file(gv.tables_path + "compare_env_policies.tex", to_tex)
if show_output:
    print(to_tex)

format_str = lambda x, round_dig: str(np.round(x, round_dig)) if not np.isnan(x) else "-"
fig, axs = plt.subplots(2, source_names.shape[0], figsize=(5.0 * source_names.shape[0], 10), squeeze=False)

# Interpolate policy values
delta_emissions = 10.0
delta_emissions_idx = np.argmin((emissions_array - delta_emissions)**2.0)
cap_pay_idx = 0
cap_production_subsidy = combine_gas(renewable_production_subsidies_results['expected_agg_source_capacity'], 2)
cap_investment_subsidy = combine_gas(renewable_investment_subsidies_results['expected_agg_source_capacity'], 2)
cap_carbon_tax = combine_gas(carbon_tax_capacity_payment_results['expected_agg_source_capacity'][:,cap_pay_idx,:,:], 2) # no capacity payment
prod_production_subsidy = combine_gas(renewable_production_subsidies_results['expected_frac_by_source'], 2)
prod_investment_subsidy = combine_gas(renewable_investment_subsidies_results['expected_frac_by_source'], 2)
prod_carbon_tax = combine_gas(carbon_tax_capacity_payment_results['expected_frac_by_source'][:,cap_pay_idx,:,:], 2) # no capacity payment
cap_production_subsidy_deltaemissions = np.zeros((cap_production_subsidy.shape[1], cap_production_subsidy.shape[2]))
cap_investment_subsidy_deltaemissions = np.zeros((cap_investment_subsidy.shape[1], cap_investment_subsidy.shape[2]))
cap_carbon_tax_deltaemissions = np.zeros((cap_carbon_tax.shape[1], cap_carbon_tax.shape[2]))
prod_production_subsidy_deltaemissions = np.zeros((cap_production_subsidy.shape[1], cap_production_subsidy.shape[2]))
prod_investment_subsidy_deltaemissions = np.zeros((cap_production_subsidy.shape[1], cap_production_subsidy.shape[2]))
prod_carbon_tax_deltaemissions = np.zeros((cap_production_subsidy.shape[1], cap_production_subsidy.shape[2]))

for t in range(cap_production_subsidy.shape[1]):
    for s in range(cap_production_subsidy.shape[2]):
        cap_production_subsidy_deltaemissions[t,s] = interp.CubicSpline(renewable_subsidies_linspace, cap_production_subsidy[:,t,s])(policy_predict['renew. prod. subs.'][delta_emissions_idx])
        cap_investment_subsidy_deltaemissions[t,s] = interp.CubicSpline(renewable_investment_subsidy_linspace, cap_investment_subsidy[:,t,s])(policy_predict['renew. inv. subs.'][delta_emissions_idx] / 100.0)
        cap_carbon_tax_deltaemissions[t,s] = interp.CubicSpline(carbon_taxes_linspace, cap_carbon_tax[:,t,s])(policy_predict['carbon tax'][delta_emissions_idx])
        prod_production_subsidy_deltaemissions[t,s] = interp.CubicSpline(renewable_subsidies_linspace, prod_production_subsidy[:,t,s])(policy_predict['renew. prod. subs.'][delta_emissions_idx])
        prod_investment_subsidy_deltaemissions[t,s] = interp.CubicSpline(renewable_investment_subsidy_linspace, prod_investment_subsidy[:,t,s])(policy_predict['renew. inv. subs.'][delta_emissions_idx] / 100.0)
        prod_carbon_tax_deltaemissions[t,s] = interp.CubicSpline(carbon_taxes_linspace, prod_carbon_tax[:,t,s])(policy_predict['carbon tax'][delta_emissions_idx])
cap_all = np.stack([cap_carbon_tax[0,:,:], cap_carbon_tax_deltaemissions, cap_production_subsidy_deltaemissions, cap_investment_subsidy_deltaemissions], axis=0)
prod_all = np.stack([prod_carbon_tax[0,:,:], prod_carbon_tax_deltaemissions, prod_production_subsidy_deltaemissions, prod_investment_subsidy_deltaemissions], axis=0)

# Colors and line styles
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, 4)]
ls = ["solid", "dotted", "dashed", "dashdot"]

# Labels
labels = [
    "no policy",
    f"carbon tax ($\\tau = $A\\${np.round(policy_predict['carbon tax'][delta_emissions_idx], 1)})",
    f"renew. prod. subs. ($\\varsigma = $A\\${np.round(policy_predict['renew. prod. subs.'][delta_emissions_idx], 1)})",
    f"renew. inv. subs. ($s = ${np.round(policy_predict['renew. inv. subs.'][delta_emissions_idx], 1)}%)"
]

cap_max = np.max(cap_all)
cap_min = 0.0
cap_ylim = (cap_min - 0.1 * (cap_max - cap_min), cap_max + 0.1 * (cap_max - cap_min))

# Emissions and bar chart setup
VOLL = 1000
emissions_levels = [5, 7.5, 10, 12.5]
emissions_indices = [np.argmin(np.abs(emissions_array - level)) for level in emissions_levels]
policy_list = ['carbon tax', 'renew. prod. subs.', 'renew. inv. subs.']
policy_symbols = {'carbon tax': '\\tau', 'renew. prod. subs.': '\\varsigma', 'renew. inv. subs.': 's'}
offsets = [-0.2, 0.0, 0.2]
x = np.arange(5)
components = ['$\\substack{-VOLL \\\\ \\times \\Delta B}$', '$\\Delta CS$', '$\\Delta PS$', '$\\Delta G$', '$\\Delta W$']

# Top Row: Capacity
for s, source in enumerate(source_names):
    for i in range(4):
        axs[0, s].plot(
            years_use,
            select_years(cap_all, 1)[i, :, s],
            color=colors[i],
            label=labels[i] if s == 0 else None,
            lw=lw_paper,
            ls=ls[i]
        )
    axs[0, s].set_title(f"{source_names[s]} capacity", fontsize=title_fontsize_paper)
    axs[0, s].set_ylim(cap_ylim)
    if s == 0:
        axs[0, s].set_ylabel("MW", fontsize=fontsize_paper)

# Bottom Row: Welfare Decomposition
all_wlf = []
for emission_idx in emissions_indices:
    for policy in policy_list:
        blk = blackouts_predict[policy][emission_idx]
        cs = CS_predict[policy][emission_idx]
        ps = PS_predict[policy][emission_idx]
        g = G_predict[policy][emission_idx]
        blk_val = -VOLL * blk * 1_000_000.0 / 1000000000.0
        total = blk_val + cs + ps + g
        all_wlf.extend([blk_val, cs, ps, g, total])

all_wlf = np.array(all_wlf)
wlf_min, wlf_max = np.min(all_wlf), np.max(all_wlf)
pad = 0.05 * (wlf_max - wlf_min)
wlf_lim = (wlf_min - pad, wlf_max + pad)

for idx, (ax, emission_idx) in enumerate(zip(axs[1], emissions_indices)):
    bar_values_main = np.zeros((5, len(policy_list)))
    legend_entries = []

    for p_idx, policy in enumerate(policy_list):
        blk = blackouts_predict[policy][emission_idx]
        cs = CS_predict[policy][emission_idx]
        ps = PS_predict[policy][emission_idx]
        g = G_predict[policy][emission_idx]
        blk_val = -VOLL * blk * 1_000_000.0 / 1000000000.0
        total = blk_val + cs + ps + g
        bar_values_main[:, p_idx] = [blk_val, cs, ps, g, total]

        val = format_str(policy_predict[policy][emission_idx], 1)
        if policy == "carbon tax":
            label_val = f"A\\${val}/ton"
        elif policy == "renew. prod. subs.":
            label_val = f"A\\${val}/MWh"
        elif policy == "renew. inv. subs.":
            label_val = f"{val}%"

        legend_entries.append(f"{policy} (${policy_symbols[policy]} = ${label_val})")

    for p_idx, (label, offset, color) in enumerate(zip(legend_entries, offsets, colors[1:])):
        ax.bar(x + offset, bar_values_main[:, p_idx], width=0.2, color=color, label=label)

    ax.set_ylim(wlf_lim)
    ax.set_xticks(x)
    ax.set_xticklabels(components, ha='center', fontsize=0.8*fontsize_paper)
    ax.axhline(0, color='black', linewidth=0.8, zorder=5)
    ax.set_title(f"$\\Delta E = {emissions_levels[idx]}$", fontsize=title_fontsize_paper)

    if idx == 0:
        ax.set_ylabel('$\\Delta$Welfare (billion A\\$)', fontsize=fontsize_paper)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=fontsize_paper - 4, frameon=False)

# Shared Top Legend
handles, labels_top = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_top, loc='lower center', bbox_to_anchor=(0.5, 0.515), ncol=4, fontsize=fontsize_paper)

# Layout
fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.3, hspace=0.4)
if save_output:
    plt.savefig(f"{gv.graphs_path}compare_co2tax_renewsubsidies.pdf", transparent=True)
    create_file(f"{gv.stats_path}avg_share_co2tax_coal.tex", f"{100.0 * np.mean(select_years(prod_all, 1), axis=1)[1,0]:.1f}")
    create_file(f"{gv.stats_path}avg_share_co2tax_gas.tex", f"{100.0 * np.mean(select_years(prod_all, 1), axis=1)[1,1]:.1f}")
    create_file(f"{gv.stats_path}avg_share_prodsub_coal.tex", f"{100.0 * np.mean(select_years(prod_all, 1), axis=1)[2,0]:.1f}")
    create_file(f"{gv.stats_path}avg_share_prodsub_gas.tex", f"{100.0 * np.mean(select_years(prod_all, 1), axis=1)[2,1]:.1f}")
    create_file(f"{gv.stats_path}avg_share_invsub_coal.tex", f"{100.0 * np.mean(select_years(prod_all, 1), axis=1)[3,0]:.1f}")
    create_file(f"{gv.stats_path}avg_share_invsub_gas.tex", f"{100.0 * np.mean(select_years(prod_all, 1), axis=1)[3,1]:.1f}")
    create_file(f"{gv.stats_path}avg_share_co2tax_renewables.tex", f"{100.0 * np.sum(np.mean(select_years(prod_all, 1), axis=1)[:,-2:], axis=1)[1]:.1f}")
    create_file(f"{gv.stats_path}avg_share_prodsub_renewables.tex", f"{100.0 * np.sum(np.mean(select_years(prod_all, 1), axis=1)[:,-2:], axis=1)[2]:.1f}")
    create_file(f"{gv.stats_path}avg_share_invsub_renewables.tex", f"{100.0 * np.sum(np.mean(select_years(prod_all, 1), axis=1)[:,-2:], axis=1)[3]:.1f}")

# %%
# Compare environmental policies

to_tex = "\\begin{tabular}{@{\\extracolsep{0pt}}cccccccc@{}} \n"
to_tex += "\\hline \n"
to_tex += "$\\Delta \\text{emissions}$ & &  & $\\Delta \\text{blackouts}$ & $\\Delta \\text{CS}$ & $\\Delta \\text{PS}$ & $\\Delta \\text{G}$ & $\\Delta \\left(\\text{CS} + \\text{PS} + \\text{G}\\right)$ \\\\ \n"
to_tex += "(billions kg$\\text{CO}_{2}$-eq) & policy & policy value & (millions MWh) & (billions A\\$) & (billions A\\$) & (billions A\\$) & (billions A\\$) \\\\ \n"
to_tex += "\\hline \n"

emissions_array = np.linspace(0.0, 40.0, 5)
policy_predict = {} 
blackouts_predict = {}
CS_predict = {}
PS_predict = {}
G_predict = {}
W_predict = {}
policies = {
    "carbon tax": carbon_taxes_linspace, 
    "renew. subsidy": renewable_subsidies_linspace[:-1]
}
results_dict = {
    "carbon tax": carbon_tax_capacity_payment_results, 
    "renew. subsidy": renewable_production_subsidies_results
}
cap_pay_idx = np.argmin(np.abs(capacity_payments_linspace - 100000.0))
indices = {
    "carbon tax": np.ravel_multi_index(np.vstack((np.arange(carbon_taxes_linspace.shape[0]), np.ones(carbon_taxes_linspace.shape[0], dtype=int) * cap_pay_idx)), (carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0])), 
    "renew. subsidy": np.arange(renewable_subsidies_linspace.shape[0])[:-1]
}
indices['carbon tax'] = indices['carbon tax'][:-1] # doing this b/c final value results in small increase in emissions, so can't use, this is a workaround since that region not relevant for this table
policies['carbon tax'] = policies['carbon tax'][:-1]

def predict_welfare(results_dict, indices, policy_vars, emissions_array):
    # Predict policy values based on emissions
    cs_policy = interp.CubicSpline(-(ton_to_kg(results_dict['expected_emissions_sum'].flatten()[indices]) - ton_to_kg(results_dict['expected_emissions_sum'].flatten()[indices][0])) / 1.0e9, policy_vars)
    policy_predict = cs_policy(emissions_array)
    
    # Predict other welfare variables based on predicted policy
    cs_blackouts = interp.CubicSpline(policy_vars, (results_dict['expected_blackouts_sum'].flatten()[indices] - results_dict['expected_blackouts_sum'].flatten()[indices][0]) / 1.0e6)
    cs_CS = interp.CubicSpline(policy_vars, (results_dict['expected_consumer_surplus_sum'].flatten()[indices] - results_dict['expected_consumer_surplus_sum'].flatten()[indices][0]) / 1.0e9)
    cs_PS = interp.CubicSpline(policy_vars, (results_dict['expected_producer_surplus_sum'].flatten()[indices] - results_dict['expected_producer_surplus_sum'].flatten()[indices][0]) / 1.0e9)
    cs_G = interp.CubicSpline(policy_vars, (results_dict['expected_revenue_sum'].flatten()[indices] - results_dict['expected_revenue_sum'].flatten()[indices][0]) / 1.0e9)
    blackouts_predict = cs_blackouts(policy_predict)
    CS_predict = cs_CS(policy_predict)
    PS_predict = cs_PS(policy_predict)
    G_predict = cs_G(policy_predict)
    W_predict = CS_predict + PS_predict + G_predict
    
    return policy_predict, blackouts_predict, CS_predict, PS_predict, G_predict, W_predict

for i, policy in enumerate(policies.keys()):
    res = predict_welfare(results_dict[policy], indices[policy], policies[policy], emissions_array)
    policy_predict[policy] = res[0]
    blackouts_predict[policy] = res[1]
    CS_predict[policy] = res[2]
    PS_predict[policy] = res[3]
    G_predict[policy] = res[4]
    W_predict[policy] = res[5]
    
    unusable = policy_predict[policy] > policies[policy][-1]
    policy_predict[policy][unusable] = np.nan
    blackouts_predict[policy][unusable] = np.nan
    CS_predict[policy][unusable] = np.nan
    PS_predict[policy][unusable] = np.nan
    G_predict[policy][unusable] = np.nan
    W_predict[policy][unusable] = np.nan
    
format_str = lambda x, round_dig: str(np.round(x, round_dig)) if not np.isnan(x) else "-"
    
for i, delta_emissions in enumerate(emissions_array):
    to_tex += f"{int(np.round(delta_emissions))}"
    for policy in policies.keys():
        to_tex += f" & {policy} & {format_str(policy_predict[policy][i], 1)} & {format_str(blackouts_predict[policy][i], 1)} & {format_str(CS_predict[policy][i], 1)} & {format_str(PS_predict[policy][i], 1)} & {format_str(G_predict[policy][i], 1)} & {format_str(W_predict[policy][i], 1)} \\\\ \n"
    if i < emissions_array.shape[0] - 1:
        to_tex += " & & & & & & & \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

if save_output:
    create_file(gv.tables_path + "compare_env_policies_presentation.tex", to_tex)
if show_output:
    print(to_tex)

# %%
# Battery counterfactuals

# Capacity (paper version)
tax_idx_low = 0
tax_idx_high = 6
cap_pay_idx = 0
labels = [f"$\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_low]))}$ / ton", f"$\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_low]))}$ w/ battery", f"$\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_high]))}$", f"$\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_high]))}$ w/ battery"]
ls = [f"solid", f"dashed", f"solid", f"dashed"]
colors = [cm.get_cmap("Greys")(0.45), cm.get_cmap("Greys")(0.45), "black", "black"]
filename = f"{gv.graphs_path}capacity_battery.pdf"
capacities = np.concatenate((carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], battery_results['expected_agg_source_capacity'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:], battery_results['expected_agg_source_capacity'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:]), axis=0)
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
filename = f"{gv.graphs_path}capacity_battery_presentation.pdf"
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
filename = f"{gv.graphs_path}production_battery.pdf"
production = np.concatenate((carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], battery_results['expected_frac_by_source'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:], battery_results['expected_frac_by_source'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:]), axis=0)
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
filename = f"{gv.graphs_path}production_battery_presentation.pdf"
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
filename = f"{gv.graphs_path}capacity_production_battery.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
filename = f"{gv.graphs_path}capacity_production_battery_presentation.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
cspsg = np.concatenate((carbon_tax_capacity_payment_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis], battery_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1) / 1000000000.0
emissions = ton_to_kg(np.concatenate((carbon_tax_capacity_payment_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis], battery_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1)) / 1000000000.0
blackouts = np.concatenate((carbon_tax_capacity_payment_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis], battery_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1) / 1000000.0
x_axis_labels = [f"$\\tau$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"no battery", f"w/ battery"]
colors = ["black", "black"]
filename = f"{gv.graphs_path}welfare_battery_carbon_tax.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename, labels=labels)

# Welfare (presentation version)
colors = [cm.get_cmap("Blues")(0.75) for i in range(2)]
filename = f"{gv.graphs_path}welfare_battery_carbon_tax_presentation.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_battery_carbon_tax_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Capacity (paper version)
tax_idx = 0
cap_pay_idx_low = 0
cap_pay_idx_high = 4
labels = [f"$\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_low]))}$", f"$\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_low]))}$, w/ battery", f"$\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_high]))}$", f"$\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_high]))}$, w/ battery"]
ls = [f"solid", f"dashed", f"solid", f"dashed"]
colors = [cm.get_cmap("Greys")(0.45), cm.get_cmap("Greys")(0.45), "black", "black"]
filename = f"{gv.graphs_path}capacity_battery_capacity_payment.pdf"
capacities = np.concatenate((carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], battery_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:], battery_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:]), axis=0)
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
filename = f"{gv.graphs_path}capacity_battery_capacity_payment_presentation.pdf"
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
filename = f"{gv.graphs_path}production_battery_capacity_payment.pdf"
production = np.concatenate((carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], battery_results['expected_frac_by_source'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:], battery_results['expected_frac_by_source'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:]), axis=0)
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
filename = f"{gv.graphs_path}production_battery_capacity_payment_presentation.pdf"
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
filename = f"{gv.graphs_path}capacity_production_battery_capacity_payment.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
filename = f"{gv.graphs_path}capacity_production_battery_capacity_payment_presentation.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
tax_idx = 0
cspsg = np.concatenate((carbon_tax_capacity_payment_results['expected_product_market_sum'][tax_idx,:][:,np.newaxis], battery_results['expected_product_market_sum'][tax_idx,:][:,np.newaxis]), axis=1) / 1000000000.0
emissions = ton_to_kg(np.concatenate((carbon_tax_capacity_payment_results['expected_emissions_sum'][tax_idx,:][:,np.newaxis], battery_results['expected_emissions_sum'][tax_idx,:][:,np.newaxis]), axis=1)) / 1000000000.0
blackouts = np.concatenate((carbon_tax_capacity_payment_results['expected_blackouts_sum'][tax_idx,:][:,np.newaxis], battery_results['expected_blackouts_sum'][tax_idx,:][:,np.newaxis]), axis=1) / 1000000.0
x_axis_labels = [f"$\\kappa$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"no battery", f"w/ battery"]
colors = ["black", "black"]
filename = f"{gv.graphs_path}welfare_battery_capacity_payment.pdf"
plot_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename, labels=labels)

# Welfare (presentation version)
colors = [cm.get_cmap("Reds")(0.75) for i in range(2)]
filename = f"{gv.graphs_path}welfare_battery_capacity_payment_presentation.pdf"
plot_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_battery_capacity_payment_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# === Helper Function ===
def zero_aligned_limits(reference_vals):
    pad = 0.05 * (np.max(reference_vals) - np.min(reference_vals))
    return (np.min(reference_vals) - pad, np.max(reference_vals) + pad)

# === Parameters ===
SCC = 230
VOLL = 1000
cap_pay_idx = 0
tax_indices = [0, 6]
offsets = [-0.1, 0.1]
components = ['$\\Delta CS$', '$\\Delta PS$', '$\\Delta G$', '$\\substack{-SCC \\\\ \\times \\Delta E}$', '$\\substack{-VOLL \\\\ \\times \\Delta B}$', '$\\Delta W$']
x = np.arange(len(components))
colors_bar = [cm.get_cmap("Greys")(0.45), "black"]
policy_labels = ['no battery', 'battery']
policy_dict = [carbon_tax_capacity_payment_results, battery_results]

# === Capacity Plot Setup ===
labels = [
    label
    for i in tax_indices
    for label in (
        f"$\\tau = {int(carbon_taxes_linspace[i])}$",
        f"$\\tau = {int(carbon_taxes_linspace[i])}$ w/ battery"
    )
]
ls = ["solid", "dashed", "solid", "dashed"]
colors = [cm.get_cmap("Greys")(0.45), cm.get_cmap("Greys")(0.45), "black", "black"]

capacities = []
for i in tax_indices:
    capacities.append(carbon_tax_capacity_payment_results['expected_agg_source_capacity'][i, cap_pay_idx])
    capacities.append(battery_results['expected_agg_source_capacity'][i, cap_pay_idx])
capacities = np.stack(capacities)

tax_indices = [0, 2, 4, 6]

# === Collect All Values for Axis Alignment ===
all_welfare_components = []
for tax_idx in tax_indices:
    for p_idx in range(2):
        cs = policy_dict[p_idx]['expected_consumer_surplus_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][0, 0]
        ps = policy_dict[p_idx]['expected_producer_surplus_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_producer_surplus_sum'][0, 0]
        g = policy_dict[p_idx]['expected_revenue_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_revenue_sum'][0, 0]
        e = policy_dict[p_idx]['expected_emissions_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_emissions_sum'][0, 0]
        blk = policy_dict[p_idx]['expected_blackouts_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_blackouts_sum'][0, 0]
        neg_scc_e = -SCC * e
        neg_voll_b = -VOLL * blk
        total = cs + ps + g + neg_scc_e + neg_voll_b
        all_welfare_components.extend([cs, ps, g, neg_scc_e, neg_voll_b, total])
all_welfare_components = np.array(all_welfare_components) / 1e9
y_lim = zero_aligned_limits(all_welfare_components)

# === Figure Setup ===
fig, axs = plt.subplots(2, 4, figsize=(20, 10), squeeze=False)

# === Top Row: Capacity ===
cap_all = combine_gas(capacities, 2)
cap_max = np.max(cap_all)
cap_min = 0.0
cap_ylim = (cap_min - 0.1 * (cap_max - cap_min), cap_max + 0.1 * (cap_max - cap_min))

for s, source in enumerate(source_names):
    for i in range(4):
        axs[0, s].plot(
            years_use,
            select_years(combine_gas(capacities, 2), 1)[i, :, s],
            color=colors[i],
            label=labels[i] if s == 0 else None,
            lw=lw_paper,
            ls=ls[i]
        )
    axs[0, s].set_title(f"{source_names[s]} capacity", fontsize=title_fontsize_paper)
    axs[0, s].set_ylim(cap_ylim)
    if s == 0:
        axs[0, s].set_ylabel("MW", fontsize=fontsize_paper)

# === Bottom Row: Welfare (Unified Axis) ===
for idx, (ax, tax_idx) in enumerate(zip(axs[1], tax_indices)):
    bars = np.zeros((6, 2))
    for p_idx in range(2):
        cs = policy_dict[p_idx]['expected_consumer_surplus_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'][0, 0]
        ps = policy_dict[p_idx]['expected_producer_surplus_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_producer_surplus_sum'][0, 0]
        g = policy_dict[p_idx]['expected_revenue_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_revenue_sum'][0, 0]
        e = policy_dict[p_idx]['expected_emissions_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_emissions_sum'][0, 0]
        blk = policy_dict[p_idx]['expected_blackouts_sum'][tax_idx, cap_pay_idx] - carbon_tax_capacity_payment_results['expected_blackouts_sum'][0, 0]
        neg_scc_e = -SCC * e
        neg_voll_b = -VOLL * blk
        total = cs + ps + g + neg_scc_e + neg_voll_b
        bars[:, p_idx] = [cs, ps, g, neg_scc_e, neg_voll_b, total]

    bars /= 1e9  # Convert all to billions

    for p_idx, offset in enumerate(offsets):
        ax.bar(x + offset, bars[:, p_idx], width=0.2, color=colors_bar[p_idx], label=policy_labels[p_idx])

    ax.set_ylim(y_lim)
    ax.set_xticks(x)
    ax.set_xticklabels(components, ha='center', fontsize=0.8*fontsize_paper)
    ax.axhline(0, color='black', linewidth=0.8, zorder=5)
    ax.set_title(f"$\\tau = {int(carbon_taxes_linspace[tax_idx])}$", fontsize=title_fontsize_paper)

    if idx == 0:
        ax.set_ylabel("$\\Delta$Welfare (billion A\\$)", fontsize=fontsize_paper)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=fontsize_paper - 4, frameon=False)

# === Shared Legend ===
handles, labels_top = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_top, loc='lower center', bbox_to_anchor=(0.5, 0.515), ncol=4, fontsize=fontsize_paper)

# === Layout ===
fig.subplots_adjust(bottom=0.20, top=0.88, wspace=0.3, hspace=0.4)

if save_output:
    plt.savefig(f"{gv.graphs_path}compare_battery.pdf", transparent=True)

# %%
# Optimal carbon taxes / capacity payments

# Set up the space would like to consider
min_carbon_tax = 0.0
max_carbon_tax = 300.0
num_points = 1000 # need large number
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points)
min_capacity_payment = 0.0
max_capacity_payment = 200000.0
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points)

interp_CS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_G = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_revenue_sum'], kx=3, ky=3, s=0.0)
interp_emissions = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
_interp_blackouts = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_blackouts = lambda x, y: np.maximum(0.0, _interp_blackouts(x, y))
interp_CS_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_G_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_revenue_sum'], kx=3, ky=3, s=0.0)
interp_CSPS_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
_interp_blackouts_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_battery = lambda x, y: np.maximum(0.0, _interp_blackouts_battery(x, y))

scc_vals = np.array([230.0])
num_points_scc = scc_vals.shape[0]
voll_vals = np.array([1000.0, 10000.0])
num_points_voll = voll_vals.shape[0]
num_people = 1100000.0 * 1000.0 # interpretation: thousand A$ per person

# Carbon tax, no battery
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
E_W_compare = E_W[:,:,0,0]
CS_compare = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
PS_compare = interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
G_compare = interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
emissions_compare = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
blackouts_compare = interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
max_policy = np.argmax(E_W[:,:,:,0], axis=2) # capacity payments = 0
carbon_tax_alone = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone = (np.max(E_W[:,:,:,0], axis=2) - E_W_compare) / num_people
cs_carbon_tax_alone = (np.take_along_axis(interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_carbon_tax_alone = (np.take_along_axis(interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_carbon_tax_alone = (np.take_along_axis(interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_carbon_tax_alone = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Carbon tax, battery
E_W = interp_CS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,:,0], axis=2) # capacity payments = 0
carbon_tax_alone_battery = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_battery = (np.max(E_W[:,:,:,0], axis=2) - E_W_compare) / num_people
cs_carbon_tax_alone_battery = (np.take_along_axis(interp_CS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_carbon_tax_alone_battery = (np.take_along_axis(interp_PS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_carbon_tax_alone_battery = (np.take_along_axis(interp_G_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_carbon_tax_alone_battery = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone_battery = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Joint, no battery
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], E_W.shape[1], -1)), axis=2)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[2:]))
carbon_tax_joint = carbon_taxes_fine[max_policy_1]
capacity_payment_joint = capacity_payments_fine[max_policy_2]
max_w_joint = (np.max(np.reshape(E_W, (E_W.shape[0],E_W.shape[1],-1)), axis=2) - E_W_compare) / num_people
cs_joint = (np.take_along_axis(np.reshape(interp_CS(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_joint = (np.take_along_axis(np.reshape(interp_PS(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_joint = (np.take_along_axis(np.reshape(interp_G(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_joint = -scc_vals[:,np.newaxis] * (np.take_along_axis(np.reshape(interp_emissions(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_joint = -voll_vals[np.newaxis,:] * (np.take_along_axis(np.reshape(interp_blackouts(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Joint, battery
E_W = interp_CS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], E_W.shape[1], -1)), axis=2)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[2:]))
carbon_tax_battery = carbon_taxes_fine[max_policy_1]
capacity_payment_battery = capacity_payments_fine[max_policy_2]
max_w_battery = (np.max(np.reshape(E_W, (E_W.shape[0],E_W.shape[1],-1)), axis=2) - E_W_compare) / num_people
cs_battery = (np.take_along_axis(np.reshape(interp_CS_battery(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_battery = (np.take_along_axis(np.reshape(interp_PS_battery(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_battery = (np.take_along_axis(np.reshape(interp_G_battery(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_battery = -scc_vals[:,np.newaxis] * (np.take_along_axis(np.reshape(interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_battery = -voll_vals[np.newaxis,:] * (np.take_along_axis(np.reshape(interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,np.newaxis,:], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

format_carbon_tax = lambda x: f"{np.round(x, 1)}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x / 100.0) * 100.0)).replace(",","\\,")
format_W = lambda x: "{:,}".format(np.round(x, 1)).replace(",","\\,") if not np.isclose(x, 0.0) else "0.0"
format_scc = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")
format_voll = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")

to_tex = "\\begin{tabular}{rrccccccccccccc} \n"
to_tex += "\\hline \\\\ \n"
to_tex += " & & \\multicolumn{2}{c}{carbon tax alone, no battery} & & \\multicolumn{2}{c}{carbon tax alone, battery} & & \\multicolumn{3}{c}{joint policies, no battery} & & \\multicolumn{3}{c}{joint policies, battery} \\\\ \n"
to_tex += " \\cline{3-4} \\cline{6-7} \\cline{9-11} \\cline{13-15} \n"
to_tex += "$VOLL$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
to_tex += "\\hline \n"

for i, scc_val in enumerate(scc_vals):
    for j, voll_val in enumerate(voll_vals):
        to_tex += "\\textbf{" + format_voll(voll_val) + "}"
        to_tex += " & & "
        to_tex += "\\textbf{" + format_carbon_tax(carbon_tax_alone[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_carbon_tax_alone[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_alone_battery[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_carbon_tax_alone_battery[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_joint[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_cap_pay(capacity_payment_joint[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_joint[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_battery[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_cap_pay(capacity_payment_battery[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_battery[i,j]) + "}"
        to_tex += " \\\\ \n"

        # Decomposed welfare result
        to_tex += " & $\\Delta \\text{CS}$"
        to_tex += " & & " + format_W(cs_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(cs_carbon_tax_alone_battery[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(cs_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(cs_battery[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $\\Delta \\text{PS}$"
        to_tex += " & & " + format_W(ps_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(ps_carbon_tax_alone_battery[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(ps_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(ps_battery[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $\\Delta \\text{G}$"
        to_tex += " & & " + format_W(g_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(g_carbon_tax_alone_battery[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(g_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(g_battery[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $-SCC \\times \\Delta \\text{E}$"
        to_tex += " & & " + format_W(emissions_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(emissions_carbon_tax_alone_battery[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(emissions_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(emissions_battery[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $-VOLL \\times \\Delta \\text{B}$"
        to_tex += " & & " + format_W(blackouts_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(blackouts_carbon_tax_alone_battery[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(blackouts_joint[i,j])
        to_tex += " & "
        to_tex += " & & & " + format_W(blackouts_battery[i,j])
        to_tex += " \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

create_file(gv.tables_path + "optimal_policy_battery.tex", to_tex)

if show_output:
    print(to_tex)

# %%
# Competitive counterfactuals

# Capacity (paper version)
tax_idx_low = 0
tax_idx_high = 3
cap_pay_idx = 0
labels = [f"$\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_low]))}$ / ton", f"$\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_low]))}$ competitive", f"$\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_high]))}$", f"$\\tau = $A\\$${int(np.round(carbon_taxes_linspace[tax_idx_high]))}$ competitive"]
ls = [f"solid", f"dashed", f"solid", f"dashed"]
colors = [cm.get_cmap("Greys")(0.45), cm.get_cmap("Greys")(0.45), "black", "black"]
filename = f"{gv.graphs_path}capacity_competitive.pdf"
capacities = np.concatenate((carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], competitive_results['expected_agg_source_capacity'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:], competitive_results['expected_agg_source_capacity'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:]), axis=0)
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
filename = f"{gv.graphs_path}capacity_competitive_presentation.pdf"
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
filename = f"{gv.graphs_path}production_competitive.pdf"
production = np.concatenate((carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], competitive_results['expected_frac_by_source'][tax_idx_low, cap_pay_idx,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:], competitive_results['expected_frac_by_source'][tax_idx_high, cap_pay_idx,:,:][np.newaxis,:,:]), axis=0)
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
filename = f"{gv.graphs_path}production_competitive_presentation.pdf"
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
filename = f"{gv.graphs_path}capacity_production_competitive.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
filename = f"{gv.graphs_path}capacity_production_competitive_presentation.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
cspsg = np.concatenate((carbon_tax_capacity_payment_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis], competitive_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1) / 1000000000.0
emissions = ton_to_kg(np.concatenate((carbon_tax_capacity_payment_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis], competitive_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1)) / 1000000000.0
blackouts = np.concatenate((carbon_tax_capacity_payment_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis], competitive_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis]), axis=1) / 1000000.0
x_axis_labels = [f"$\\tau$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"market power", f"competitive"]
colors = ["black", "black"]
filename = f"{gv.graphs_path}welfare_competitive_carbon_tax.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename, labels=labels)

# Welfare (presentation version)
colors = [cm.get_cmap("Blues")(0.75) for i in range(2)]
filename = f"{gv.graphs_path}welfare_competitive_carbon_tax_presentation.pdf"
plot_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_competitive_carbon_tax_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Capacity (paper version)
tax_idx = 0
cap_pay_idx_low = 0
cap_pay_idx_high = 4
labels = [f"$\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_low]))}$", f"$\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_low]))}$, competitive", f"$\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_high]))}$", f"$\\kappa = $A\\$${int(np.round(capacity_payments_linspace[cap_pay_idx_high]))}$, competitive"]
ls = [f"solid", f"dashed", f"solid", f"dashed"]
colors = [cm.get_cmap("Greys")(0.45), cm.get_cmap("Greys")(0.45), "black", "black"]
filename = f"{gv.graphs_path}capacity_competitive_capacity_payment.pdf"
capacities = np.concatenate((carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], competitive_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:], competitive_results['expected_agg_source_capacity'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:]), axis=0)
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity (presentation version)
filename = f"{gv.graphs_path}capacity_competitive_capacity_payment_presentation.pdf"
plot_capacities(capacities, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Production (paper version)
filename = f"{gv.graphs_path}production_competitive_capacity_payment.pdf"
production = np.concatenate((carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], competitive_results['expected_frac_by_source'][tax_idx, cap_pay_idx_low,:,:][np.newaxis,:,:], carbon_tax_capacity_payment_results['expected_frac_by_source'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:], competitive_results['expected_frac_by_source'][tax_idx, cap_pay_idx_high,:,:][np.newaxis,:,:]), axis=0)
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Production (presentation version)
filename = f"{gv.graphs_path}production_competitive_capacity_payment_presentation.pdf"
plot_production(production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Capacity + production (paper version)
filename = f"{gv.graphs_path}capacity_production_competitive_capacity_payment.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename)

# Capacity + production (presentation version)
filename = f"{gv.graphs_path}capacity_production_competitive_capacity_payment_presentation.pdf"
plot_combined_capacities_production(capacities, production, np.array([0, 1, 2, 3]), labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename)

# Welfare (paper version)
tax_idx = 0
cspsg = np.concatenate((carbon_tax_capacity_payment_results['expected_product_market_sum'][tax_idx,:][:,np.newaxis], competitive_results['expected_product_market_sum'][tax_idx,:][:,np.newaxis]), axis=1) / 1000000000.0
emissions = ton_to_kg(np.concatenate((carbon_tax_capacity_payment_results['expected_emissions_sum'][tax_idx,:][:,np.newaxis], competitive_results['expected_emissions_sum'][tax_idx,:][:,np.newaxis]), axis=1)) / 1000000000.0
blackouts = np.concatenate((carbon_tax_capacity_payment_results['expected_blackouts_sum'][tax_idx,:][:,np.newaxis], competitive_results['expected_blackouts_sum'][tax_idx,:][:,np.newaxis]), axis=1) / 1000000.0
x_axis_labels = [f"$\\kappa$" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"market power", f"competitive"]
colors = ["black", "black"]
filename = f"{gv.graphs_path}welfare_competitive_capacity_payment.pdf"
plot_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, filename, labels=labels)

# Welfare (presentation version)
colors = [cm.get_cmap("Reds")(0.75) for i in range(2)]
filename = f"{gv.graphs_path}welfare_competitive_capacity_payment_presentation.pdf"
plot_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# Welfare (presentation version)
filename = f"{gv.graphs_path}welfare_competitive_capacity_payment_presentation_limited.pdf"
plot_limited_welfare(cspsg, emissions, blackouts, capacity_payments_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# %%
# Optimal carbon taxes / capacity payments

# Set up the space would like to consider
min_carbon_tax = 0.0
max_carbon_tax = 300.0
num_points = 500 # need large number
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points)
min_capacity_payment = 0.0
max_capacity_payment = 200000.0
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points)

interp_CS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_G = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_revenue_sum'], kx=3, ky=3, s=0.0)
interp_emissions = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
_interp_blackouts = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_blackouts = lambda x, y: np.maximum(0.0, _interp_blackouts(x, y))
interp_CS_competitive = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, competitive_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS_competitive = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, competitive_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_G_competitive = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, competitive_results['expected_revenue_sum'], kx=3, ky=3, s=0.0)
interp_CSPS_competitive = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, competitive_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions_competitive = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, competitive_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
_interp_blackouts_competitive = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, competitive_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_competitive = lambda x, y: np.maximum(0.0, _interp_blackouts_competitive(x, y))
carbon_taxes_fine = np.linspace(np.min(carbon_taxes_linspace), np.max(carbon_taxes_linspace), 1000)
capacity_payments_fine = np.linspace(np.min(capacity_payments_linspace), np.max(capacity_payments_linspace), 1000)

scc_vals = np.array([230.0])
num_points_scc = scc_vals.shape[0]
voll_vals = np.array([1000.0, 10000.0])
num_points_voll = voll_vals.shape[0]
num_people = 1100000.0 * 1000.0 # interpretation: thousand A$ per person

# Market power
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
E_W_compare = E_W[:,:,0,0]
CS_compare = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
PS_compare = interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
G_compare = interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
emissions_compare = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
blackouts_compare = interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,0,0]
max_policy = np.argmax(E_W[:,:,:1,0], axis=2) # capacity payments = 0, only 0 carbon tax
max_w_marketpower = (np.max(E_W[:,:,:1,0], axis=2) - E_W_compare) / num_people
cs_marketpower = (np.take_along_axis(interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_marketpower = (np.take_along_axis(interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_marketpower = (np.take_along_axis(interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_marketpower = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_marketpower = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Competitive
E_W = interp_CS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,:1,0], axis=2) # capacity payments = 0, only 0 carbon tax
max_w_competitive = (np.max(E_W[:,:,:1,0], axis=2) - E_W_compare) / num_people
cs_competitive = (np.take_along_axis(interp_CS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_competitive = (np.take_along_axis(interp_PS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_competitive = (np.take_along_axis(interp_G_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_competitive = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_competitive = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Carbon tax, market power
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,:,0], axis=2) # capacity payments = 0
carbon_tax_alone = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone = (np.max(E_W[:,:,:,0], axis=2) - E_W_compare) / num_people
cs_carbon_tax_alone = (np.take_along_axis(interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_carbon_tax_alone = (np.take_along_axis(interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_carbon_tax_alone = (np.take_along_axis(interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_carbon_tax_alone = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Carbon tax chosen for market power, but competitive
E_W = interp_CS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
carbon_tax_alone_marketpowercompetitive = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_marketpowercompetitive = (np.take_along_axis(E_W[:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - E_W_compare) / num_people
cs_carbon_tax_alone_marketpowercompetitive = (np.take_along_axis(interp_CS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_carbon_tax_alone_marketpowercompetitive = (np.take_along_axis(interp_PS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_carbon_tax_alone_marketpowercompetitive = (np.take_along_axis(interp_G_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_carbon_tax_alone_marketpowercompetitive = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone_marketpowercompetitive = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

# Carbon tax, competitive
E_W = interp_CS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_PS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] + interp_G_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis,np.newaxis] * interp_emissions_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:] - voll_vals[np.newaxis,:,np.newaxis,np.newaxis] * interp_blackouts_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, num_points_voll, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,:,0], axis=2) # capacity payments = 0
carbon_tax_alone_competitive = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_competitive = (np.max(E_W[:,:,:,0], axis=2) - E_W_compare) / num_people
cs_carbon_tax_alone_competitive = (np.take_along_axis(interp_CS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - CS_compare) / num_people
ps_carbon_tax_alone_competitive = (np.take_along_axis(interp_PS_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - PS_compare) / num_people
g_carbon_tax_alone_competitive = (np.take_along_axis(interp_G_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - G_compare) / num_people
emissions_carbon_tax_alone_competitive = -scc_vals[:,np.newaxis] * (np.take_along_axis(interp_emissions_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone_competitive = -voll_vals[np.newaxis,:] * (np.take_along_axis(interp_blackouts_competitive(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,np.newaxis,:,:][:,:,:,0], max_policy[:,:,np.newaxis], axis=2)[:,:,0] - blackouts_compare) / num_people

format_carbon_tax = lambda x: f"{np.round(x, 1)}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x / 100.0) * 100.0)).replace(",","\\,")
format_W = lambda x: "{:,}".format(np.round(x, 2)).replace(",","\\,") if not np.isclose(x, 0.0) else "0.0"
format_scc = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")
format_voll = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")

to_tex = "\\begin{tabular}{rrcccccccccccc} \n"
to_tex += "\\hline \\\\ \n"
to_tex += " & & market power & & competitive & & \\multicolumn{2}{c}{market power + opt. tax} & & \\multicolumn{2}{c}{competitive + tax} & & \\multicolumn{2}{c}{competitive + opt. tax} \\\\ \n"
to_tex += " \\cline{3-3} \\cline{5-5} \\cline{7-8} \\cline{10-11} \\cline{13-14} \n"
to_tex += "$VOLL$ & & $\\Delta \\mathcal{W}$ & & $\\Delta \\mathcal{W}$ & & $\\tau^{*}_{\\text{mkt pwr}}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}_{\\text{mkt pwr}}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}_{\\text{comp}}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
to_tex += "\\hline \n"

for i, scc_val in enumerate(scc_vals):
    for j, voll_val in enumerate(voll_vals):
        to_tex += "\\textbf{" + format_voll(voll_val) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_W(max_w_marketpower[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_W(max_w_competitive[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_alone[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_carbon_tax_alone[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_alone_marketpowercompetitive[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_carbon_tax_alone_marketpowercompetitive[i,j]) + "}"
        to_tex += " & "
        to_tex += " & " + "\\textbf{" + format_carbon_tax(carbon_tax_alone_competitive[i,j]) + "}"
        to_tex += " & " + "\\textbf{" + format_W(max_w_carbon_tax_alone_competitive[i,j]) + "}"
        to_tex += " \\\\ \n"

        # Decomposed welfare result
        to_tex += " & $\\Delta \\text{CS}$"
        to_tex += " & " + format_W(cs_marketpower[i,j])
        to_tex += " & "
        to_tex += " & " + format_W(cs_competitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(cs_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(cs_carbon_tax_alone_marketpowercompetitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(cs_carbon_tax_alone_competitive[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $\\Delta \\text{PS}$"
        to_tex += " & " + format_W(ps_marketpower[i,j])
        to_tex += " & "
        to_tex += " & " + format_W(ps_competitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(ps_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(ps_carbon_tax_alone_marketpowercompetitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(ps_carbon_tax_alone_competitive[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $\\Delta \\text{T}$"
        to_tex += " & " + format_W(g_marketpower[i,j])
        to_tex += " & "
        to_tex += " & " + format_W(g_competitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(g_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(g_carbon_tax_alone_marketpowercompetitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(g_carbon_tax_alone_competitive[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $-SCC \\times \\Delta \\text{E}$"
        to_tex += " & " + format_W(emissions_marketpower[i,j])
        to_tex += " & "
        to_tex += " & " + format_W(emissions_competitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(emissions_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(emissions_carbon_tax_alone_marketpowercompetitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(emissions_carbon_tax_alone_competitive[i,j])
        to_tex += " \\\\ \n"
    
        to_tex += " & $-VOLL \\times \\Delta \\text{B}$"
        to_tex += " & " + format_W(blackouts_marketpower[i,j])
        to_tex += " & "
        to_tex += " & " + format_W(blackouts_competitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(blackouts_carbon_tax_alone[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(blackouts_carbon_tax_alone_marketpowercompetitive[i,j])
        to_tex += " & "
        to_tex += " & & " + format_W(blackouts_carbon_tax_alone_competitive[i,j])
        to_tex += " \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

if save_output:
    create_file(gv.tables_path + "optimal_policy_competitive.tex", to_tex)

if show_output:
    print(to_tex)

# %%
# Compare results as vary carbon tax

def with_alpha(c, alpha=0.25):
    r, g, b, _ = mpl.colors.to_rgba(c)
    return (r, g, b, alpha)

def plot_limited_welfare_compare(cspsg, emissions, blackouts, x_axis_linspace, x_axis_label, y_axis_label,
                                 ls, colors, lw, fontsize, title_fontsize, filename, labels=None,
                                 alpha=0.25):
    
    multiple_lines = cspsg.ndim > 1
    if not multiple_lines:
        raise ValueError("Stepwise reveal only makes sense when multiple lines are present.")

    x_axis_linspace_fine = np.linspace(np.min(x_axis_linspace), np.max(x_axis_linspace), 1000)
    n_lines = cspsg.shape[1]

    # Precompute interpolated values to get fixed y-axis limits
    emissions_interp = [interp.Akima1DInterpolator(x_axis_linspace, emissions[:, j])(x_axis_linspace_fine) for j in range(n_lines)]
    blackouts_interp = [interp.Akima1DInterpolator(x_axis_linspace, blackouts[:, j])(x_axis_linspace_fine) for j in range(n_lines)]

    def get_buffered_ylim(data, margin=0.1):
        dmin = np.min(data)
        dmax = np.max(data)
        if dmin == dmax:
            # Handle flat lines gracefully
            return dmin - 1, dmax + 1
        drange = dmax - dmin
        min_use, max_use = dmin - margin * drange, dmax + margin * drange
        min_use = np.maximum(-1.0, min_use)
        return min_use, max_use
    
    emissions_ymin, emissions_ymax = get_buffered_ylim(emissions)
    blackouts_ymin, blackouts_ymax = get_buffered_ylim(blackouts)

    for i in range(n_lines):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), squeeze=False)

        legend_labels = []
        legend_handles = []

        # Emissions panel
        for j in range(n_lines):
            if j < i:
                a = alpha
            elif j == i:
                a = 1.0
            else:
                a = 0.0
            line, = axs[0, 0].plot(
                x_axis_linspace_fine,
                emissions_interp[j],
                color=colors[j],
                lw=lw,
                ls=ls[j],
                alpha=a,
                label=labels[j] if a > 0 else None
            )
            if a > 0:
                legend_handles.append(line)
                legend_labels.append(labels[j])
        axs[0, 0].set_xlabel(x_axis_label[1], fontsize=fontsize)
        axs[0, 0].set_ylabel(y_axis_label[1], fontsize=fontsize)
        axs[0, 0].set_title("emissions", fontsize=title_fontsize)
        axs[0, 0].set_ylim(emissions_ymin, emissions_ymax)

        # Blackouts panel
        for j in range(n_lines):
            if j < i:
                a = alpha
            elif j == i:
                a = 1.0
            else:
                a = 0.0
            axs[0, 1].plot(
                x_axis_linspace_fine,
                blackouts_interp[j],
                color=colors[j],
                lw=lw,
                ls=ls[j],
                alpha=a
            )
        axs[0, 1].set_xlabel(x_axis_label[2], fontsize=fontsize)
        axs[0, 1].set_ylabel(y_axis_label[2], fontsize=fontsize)
        axs[0, 1].set_title("blackouts", fontsize=title_fontsize)
        axs[0, 1].set_ylim(blackouts_ymin, blackouts_ymax)

        # Adjust space to make room for manual legend (tweak if needed)
        fig.subplots_adjust(bottom=0.25)
        
        # Legend layout
        n_total = len(labels)
        n_visible = i + 1
        x_padding = 0.08   # padding on each side
        usable_width = 1.0 - 2 * x_padding
        x_step = usable_width / n_total
        x_start = x_padding
        legend_y = 0.02  # closer to figure bottom
        line_x_offset = -0.04  # how far left the line is from the label
        
        for j in range(n_visible):
            main, unit = labels[j].split("\n") if "\n" in labels[j] else (labels[j], "")
        
            # Line to the left of label
            line = mpl.lines.Line2D(
                [x_start + j * x_step + line_x_offset, x_start + j * x_step - 0.005],
                [legend_y + 0.02, legend_y + 0.02],
                transform=fig.transFigure,
                color=colors[j], lw=lw, ls=ls[j], solid_capstyle='butt'
            )
            fig.add_artist(line)
        
            # Top label
            fig.text(x_start + j * x_step, legend_y + 0.04, main,
                     fontsize=fontsize, ha="left", va="center")
        
            # Bottom (unit) label
            fig.text(x_start + j * x_step, legend_y, unit,
                     fontsize=int(fontsize * 0.7), ha="left", va="center", color="black")

        if filename is not None and save_output:
            plt.savefig(f"{filename}_{i}.pdf", transparent=True)
        if show_output:
            plt.show()

        plt.close(fig)

# Carbon tax alone
cap_pay_idx = 0
cspsg = np.concatenate((carbon_tax_capacity_payment_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000000.0,), axis=1)
emissions = ton_to_kg(np.concatenate((carbon_tax_capacity_payment_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000000.0,), axis=1))
blackouts = np.concatenate((carbon_tax_capacity_payment_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000.0,), axis=1)
x_axis_labels = [f"$\\tau$ (A\\$/tonCO2-eq)" for i in range(3)]
y_axis_labels = [f"A$ (billions)", f"kgCO2-eq (billions)", f"MWh (millions)"]
labels = [f"only carbon tax"]
unit_labels = [f""]
ls = [f"solid"]
colors = [cm.get_cmap("Blues")(0.75)]

# High price cap
cspsg = np.concatenate((cspsg, high_price_cap_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000000.0), axis=1)
emissions = np.concatenate((emissions, ton_to_kg(high_price_cap_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000000.0)), axis=1)
blackouts = np.concatenate((blackouts, high_price_cap_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000.0), axis=1)
labels += [f"high price cap"]
unit_labels += [f"(A\\$1,000/MWh)"]
ls += [f"solid"]
colors += [cm.get_cmap("Greens")(0.75)]

# Battery
cspsg = np.concatenate((cspsg, battery_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000000.0), axis=1)
emissions = np.concatenate((emissions, ton_to_kg(battery_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000000.0)), axis=1)
blackouts = np.concatenate((blackouts, battery_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000.0), axis=1)
labels += [f"w/ battery"]
unit_labels += [f"($K = 2,000$MW)"]
ls += [f"solid"]
colors += [cm.get_cmap("Purples")(0.75)]

# Capacity payment
cap_pay_idx = 4
cspsg = np.concatenate((cspsg, carbon_tax_capacity_payment_results['expected_product_market_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000000.0), axis=1)
emissions = np.concatenate((emissions, ton_to_kg(carbon_tax_capacity_payment_results['expected_emissions_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000000.0)), axis=1)
blackouts = np.concatenate((blackouts, carbon_tax_capacity_payment_results['expected_blackouts_sum'][:,cap_pay_idx][:,np.newaxis] / 1000000.0), axis=1)
labels += [f"w/ capacity payment"]
unit_labels += [f"($\\kappa = \\text{{A\\$}}$200,000/MW)"]
ls += [f"solid"]
colors += [cm.get_cmap("Reds")(0.75)]

# Plot them
labels = [f"{main}\n{unit}" for main, unit in zip(labels, unit_labels)]
filename = f"{gv.graphs_path}welfare_compare_carbon_tax_presentation_limited"
plot_limited_welfare_compare(cspsg, emissions, blackouts, carbon_taxes_linspace, x_axis_labels, y_axis_labels, ls, colors, lw_presentation, 0.9*fontsize_presentation, title_fontsize_presentation, filename, labels=labels)

# %%
# Optimal carbon taxes / capacity payments

# Set up the space would like to consider
min_carbon_tax = 0.0
max_carbon_tax = 300.0
num_points = 500 # need large number
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points)
min_capacity_payment = 0.0
max_capacity_payment = 200000.0
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points)

interp_CSPS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
interp_blackouts = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_CSPS_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_CSPS_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
carbon_taxes_fine = np.linspace(np.min(carbon_taxes_linspace), np.max(carbon_taxes_linspace), 1000)
capacity_payments_fine = np.linspace(np.min(capacity_payments_linspace), np.max(capacity_payments_linspace), 1000)

scc_vals = np.array([230.0])
num_points_scc = scc_vals.shape[0]
voll = 1000.0
num_people = 1100000.0 * 1000.0 # interpretation: thousand A$ per person

# Carbon tax, low price cap
E_W = interp_CSPS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
E_W_compare = E_W[:,0,0]
max_policy = np.argmax(E_W[:,:,0], axis=1) # capacity payments = 0
carbon_tax_alone = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone = (np.max(E_W[:,:,0], axis=1) - E_W_compare) / num_people

# Carbon tax, high price cap
E_W = interp_CSPS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,0], axis=1) # capacity payments = 0
carbon_tax_alone_price_cap = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_price_cap = (np.max(E_W[:,:,0], axis=1) - E_W_compare) / num_people

# Carbon tax, battery
E_W = interp_CSPS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,0], axis=1) # capacity payments = 0
carbon_tax_alone_battery = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_battery = (np.max(E_W[:,:,0], axis=1) - E_W_compare) / num_people

# Carbon tax + capacity payment, low price cap
E_W = interp_CSPS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_joint = carbon_taxes_fine[max_policy_1]
capacity_payment_joint = capacity_payments_fine[max_policy_2]
max_w_joint = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people

# Carbon tax + capacity payment, high price cap
E_W = interp_CSPS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_pricecap = carbon_taxes_fine[max_policy_1]
capacity_payment_pricecap = capacity_payments_fine[max_policy_2]
max_w_pricecap = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people

# Carbon tax + capacity payment, battery
E_W = interp_CSPS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_battery = carbon_taxes_fine[max_policy_1]
capacity_payment_battery = capacity_payments_fine[max_policy_2]
max_w_battery = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people

format_carbon_tax = lambda x: f"{np.round(x, 1)}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x / 100.0) * 100.0)).replace(",","\\,")
format_W = lambda x: "{:,}".format(np.round(x, 2)).replace(",","\\,")
format_scc = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")

# to_tex = "\\begin{tabular}{cccccccccccccccccccccccc} \n"
# to_tex += "\\hline \\\\ \n"
# to_tex += "\\multicolumn{2}{c}{carbon tax} & & \\multicolumn{2}{c}{carbon tax $+$ high $\\bar{P}$} & & \\multicolumn{2}{c}{carbon tax $+$ battery} & & \\multicolumn{3}{c}{joint policies} & & \\multicolumn{3}{c}{joint policies $+$ high $\\bar{P}$} & & \\multicolumn{3}{c}{joint policies $+$ battery} \\\\ \n"
# to_tex += " \\cline{1-2} \\cline{4-5} \\cline{7-8} \\cline{10-12} \\cline{14-16} \\cline{18-20} \n"
# to_tex += "$\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
# to_tex += "\\hline \n"

# for i, scc_val in enumerate(scc_vals):
#     to_tex += format_carbon_tax(carbon_tax_alone[i])
#     to_tex += " & " + format_W(max_w_carbon_tax_alone[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_alone_price_cap[i])
#     to_tex += " & " + format_W(max_w_carbon_tax_alone_price_cap[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_alone_battery[i])
#     to_tex += " & " + format_W(max_w_carbon_tax_alone_battery[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_joint[i])
#     to_tex += " & " + format_cap_pay(capacity_payment_joint[i])
#     to_tex += " & " + format_W(max_w_joint[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_pricecap[i])
#     to_tex += " & " + format_cap_pay(capacity_payment_pricecap[i])
#     to_tex += " & " + format_W(max_w_pricecap[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_battery[i])
#     to_tex += " & " + format_cap_pay(capacity_payment_battery[i])
#     to_tex += " & " + format_W(max_w_battery[i])
#     to_tex += " \\\\ \n"

# to_tex += "\\hline \n"
# to_tex += "\\end{tabular}"

to_tex = "\\begin{tabular}{cccccccccccccccc} \n"
to_tex += "\\hline \\\\ \n"
to_tex += "\\multicolumn{2}{c}{carbon tax} & & \\multicolumn{2}{c}{carbon tax $+$ high $\\bar{P}$} & & \\multicolumn{2}{c}{carbon tax $+$ battery} & & \\multicolumn{3}{c}{carbon tax $+$ capacity payments} \\\\ \n"
to_tex += " \\cline{1-2} \\cline{4-5} \\cline{7-8} \\cline{10-12} \n"
to_tex += "$\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
to_tex += "\\hline \n"

for i, scc_val in enumerate(scc_vals):
    to_tex += format_carbon_tax(carbon_tax_alone[i])
    to_tex += " & " + format_W(max_w_carbon_tax_alone[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_alone_price_cap[i])
    to_tex += " & " + format_W(max_w_carbon_tax_alone_price_cap[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_alone_battery[i])
    to_tex += " & " + format_W(max_w_carbon_tax_alone_battery[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_joint[i])
    to_tex += " & " + format_cap_pay(capacity_payment_joint[i])
    to_tex += " & " + format_W(max_w_joint[i])
    to_tex += " \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

# create_file(gv.tables_path + "optimal_policy_presentation.tex", to_tex)

if show_output:
    print(to_tex)

# %%
# Optimal carbon taxes / capacity payments

# Set up the space would like to consider
min_carbon_tax = 0.0
max_carbon_tax = 300.0
num_points = 500 # need large number
carbon_taxes_fine = np.linspace(min_carbon_tax, max_carbon_tax, num_points)
min_capacity_payment = 0.0
max_capacity_payment = 200000.0
capacity_payments_fine = np.linspace(min_capacity_payment, max_capacity_payment, num_points)

interp_CS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_G = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_revenue_sum'], kx=3, ky=3, s=0.0)
interp_CSPS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
interp_blackouts = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_CS_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_G_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_revenue_sum'], kx=3, ky=3, s=0.0)
interp_CSPS_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_CS_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_consumer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_PS_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_producer_surplus_sum'], kx=3, ky=3, s=0.0)
interp_G_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_revenue_sum'], kx=3, ky=3, s=0.0)
interp_CSPS_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_battery = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, battery_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
carbon_taxes_fine = np.linspace(np.min(carbon_taxes_linspace), np.max(carbon_taxes_linspace), 1000)
capacity_payments_fine = np.linspace(np.min(capacity_payments_linspace), np.max(capacity_payments_linspace), 1000)

scc_vals = np.array([230.0])
num_points_scc = scc_vals.shape[0]
voll = 1000.0
num_people = 1100000.0 * 1000.0 # interpretation: thousand A$ per person

# Carbon tax, low price cap
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] + interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] + interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
E_W_compare = E_W[:,0,0]
CS_compare = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,0,0]
PS_compare = interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,0,0]
G_compare = interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,0,0]
emissions_compare = interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,0,0]
blackouts_compare = interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,0,0]
max_policy = np.argmax(E_W[:,:,0], axis=1) # capacity payments = 0
carbon_tax_alone = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone = (np.max(E_W[:,:,0], axis=1) - E_W_compare) / num_people
cs_carbon_tax_alone = (np.take_along_axis(interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - CS_compare) / num_people
ps_carbon_tax_alone = (np.take_along_axis(interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - PS_compare) / num_people
g_carbon_tax_alone = (np.take_along_axis(interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - G_compare) / num_people
emissions_carbon_tax_alone = (np.take_along_axis(scc_vals[:,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone = (np.take_along_axis(voll * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - blackouts_compare) / num_people

# Carbon tax, high price cap
E_W = interp_CS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] + interp_PS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] + interp_G_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,0], axis=1) # capacity payments = 0
carbon_tax_alone_price_cap = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_price_cap = (np.max(E_W[:,:,0], axis=1) - E_W_compare) / num_people
cs_carbon_tax_alone_highpricecap = (np.take_along_axis(interp_CS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - CS_compare) / num_people
ps_carbon_tax_alone_highpricecap = (np.take_along_axis(interp_PS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - PS_compare) / num_people
g_carbon_tax_alone_highpricecap = (np.take_along_axis(interp_G_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - G_compare) / num_people
emissions_carbon_tax_alone_highpricecap = (np.take_along_axis(scc_vals[:,np.newaxis] * interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone_highpricecap = (np.take_along_axis(voll * interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - blackouts_compare) / num_people

# Carbon tax, battery
E_W = interp_CS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] + interp_PS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] + interp_G_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(E_W[:,:,0], axis=1) # capacity payments = 0
carbon_tax_alone_battery = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_battery = (np.max(E_W[:,:,0], axis=1) - E_W_compare) / num_people
cs_carbon_tax_alone_battery = (np.take_along_axis(interp_CS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - CS_compare) / num_people
ps_carbon_tax_alone_battery = (np.take_along_axis(interp_PS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - PS_compare) / num_people
g_carbon_tax_alone_battery = (np.take_along_axis(interp_G_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - G_compare) / num_people
emissions_carbon_tax_alone_battery = (np.take_along_axis(scc_vals[:,np.newaxis] * interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - emissions_compare) / num_people
blackouts_carbon_tax_alone_battery = (np.take_along_axis(voll * interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:][:,:,0], max_policy[:,np.newaxis], axis=1)[:,0] - blackouts_compare) / num_people

# Carbon tax + capacity payment, low price cap
E_W = interp_CS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] + interp_PS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] + interp_G(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_joint = carbon_taxes_fine[max_policy_1]
capacity_payment_joint = capacity_payments_fine[max_policy_2]
max_w_joint = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people
cs_joint = (np.take_along_axis(np.reshape(interp_CS(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,:], max_policy[:,np.newaxis], axis=1)[:,0] - CS_compare) / num_people
ps_joint = (np.take_along_axis(np.reshape(interp_PS(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,:], max_policy[:,np.newaxis], axis=1)[:,0] - PS_compare) / num_people
g_joint = (np.take_along_axis(np.reshape(interp_G(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,:], max_policy[:,np.newaxis], axis=1)[:,0] - G_compare) / num_people
emissions_joint = (np.take_along_axis(scc_vals[:,np.newaxis] * np.reshape(interp_emissions(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,:], max_policy[:,np.newaxis], axis=1)[:,0] - emissions_compare) / num_people
blackouts_joint = (np.take_along_axis(voll * np.reshape(interp_blackouts(carbon_taxes_fine, capacity_payments_fine), (-1,))[np.newaxis,:], max_policy[:,np.newaxis], axis=1)[:,0] - blackouts_compare) / num_people

# Carbon tax + capacity payment, high price cap
E_W = interp_CSPS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_pricecap = carbon_taxes_fine[max_policy_1]
capacity_payment_pricecap = capacity_payments_fine[max_policy_2]
max_w_pricecap = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people

# Carbon tax + capacity payment, battery
E_W = interp_CSPS_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_battery(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_battery = carbon_taxes_fine[max_policy_1]
capacity_payment_battery = capacity_payments_fine[max_policy_2]
max_w_battery = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people

format_carbon_tax = lambda x: f"{np.round(x, 1)}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x / 100.0) * 100.0)).replace(",","\\,")
format_W = lambda x: "{:,}".format(np.round(x, 2)).replace(",","\\,")
format_scc = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")

# to_tex = "\\begin{tabular}{cccccccccccccccccccccccc} \n"
# to_tex += "\\hline \\\\ \n"
# to_tex += "\\multicolumn{2}{c}{carbon tax} & & \\multicolumn{2}{c}{carbon tax $+$ high $\\bar{P}$} & & \\multicolumn{2}{c}{carbon tax $+$ battery} & & \\multicolumn{3}{c}{joint policies} & & \\multicolumn{3}{c}{joint policies $+$ high $\\bar{P}$} & & \\multicolumn{3}{c}{joint policies $+$ battery} \\\\ \n"
# to_tex += " \\cline{1-2} \\cline{4-5} \\cline{7-8} \\cline{10-12} \\cline{14-16} \\cline{18-20} \n"
# to_tex += "$\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
# to_tex += "\\hline \n"

# for i, scc_val in enumerate(scc_vals):
#     to_tex += format_carbon_tax(carbon_tax_alone[i])
#     to_tex += " & " + format_W(max_w_carbon_tax_alone[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_alone_price_cap[i])
#     to_tex += " & " + format_W(max_w_carbon_tax_alone_price_cap[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_alone_battery[i])
#     to_tex += " & " + format_W(max_w_carbon_tax_alone_battery[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_joint[i])
#     to_tex += " & " + format_cap_pay(capacity_payment_joint[i])
#     to_tex += " & " + format_W(max_w_joint[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_pricecap[i])
#     to_tex += " & " + format_cap_pay(capacity_payment_pricecap[i])
#     to_tex += " & " + format_W(max_w_pricecap[i])
#     to_tex += " & "
#     to_tex += " & " + format_carbon_tax(carbon_tax_battery[i])
#     to_tex += " & " + format_cap_pay(capacity_payment_battery[i])
#     to_tex += " & " + format_W(max_w_battery[i])
#     to_tex += " \\\\ \n"

# to_tex += "\\hline \n"
# to_tex += "\\end{tabular}"

to_tex = "\\begin{tabular}{rcccccccccccccccc} \n"
to_tex += "\\hline \\\\ \n"
to_tex += " & \\multicolumn{2}{c}{carbon tax} & & \\multicolumn{2}{c}{carbon tax $+$ high $\\bar{P}$} & & \\multicolumn{2}{c}{carbon tax $+$ battery} & & \\multicolumn{3}{c}{carbon tax $+$ capacity payments} \\\\ \n"
to_tex += " \\cline{2-3} \\cline{5-6} \\cline{8-9} \\cline{11-13} \n"
to_tex += " & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
to_tex += "\\hline \n"

for i, scc_val in enumerate(scc_vals):
    # Main result
    to_tex += " & "
    to_tex += format_carbon_tax(carbon_tax_alone[i])
    to_tex += " & " + format_W(max_w_carbon_tax_alone[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_alone_price_cap[i])
    to_tex += " & " + format_W(max_w_carbon_tax_alone_price_cap[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_alone_battery[i])
    to_tex += " & " + format_W(max_w_carbon_tax_alone_battery[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_joint[i])
    to_tex += " & " + format_cap_pay(capacity_payment_joint[i])
    to_tex += " & " + format_W(max_w_joint[i])
    to_tex += " \\\\ \n"
    
    # Decomposed welfare result
    to_tex += "\\footnotesize{$\\Delta \\text{CS}$}"
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(cs_carbon_tax_alone[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(cs_carbon_tax_alone_highpricecap[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(cs_carbon_tax_alone_battery[i]) + "}}"
    to_tex += " & "
    to_tex += " & & & \\onslide<2->{\\footnotesize{" + format_W(cs_joint[i]) + "}}"
    to_tex += " \\\\ \n"

    to_tex += "\\footnotesize{$\\Delta \\text{PS}$}"
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(ps_carbon_tax_alone[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(ps_carbon_tax_alone_highpricecap[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(ps_carbon_tax_alone_battery[i]) + "}}"
    to_tex += " & "
    to_tex += " & & & \\onslide<2->{\\footnotesize{" + format_W(ps_joint[i]) + "}}"
    to_tex += " \\\\ \n"

    to_tex += "\\footnotesize{$\\Delta \\text{G}$}"
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(g_carbon_tax_alone[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(g_carbon_tax_alone_highpricecap[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(g_carbon_tax_alone_battery[i]) + "}}"
    to_tex += " & "
    to_tex += " & & & \\onslide<2->{\\footnotesize{" + format_W(g_joint[i]) + "}}"
    to_tex += " \\\\ \n"

    to_tex += "\\footnotesize{$SCC \\times \\Delta \\text{emissions}$}"
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(emissions_carbon_tax_alone[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(emissions_carbon_tax_alone_highpricecap[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(emissions_carbon_tax_alone_battery[i]) + "}}"
    to_tex += " & "
    to_tex += " & & & \\onslide<2->{\\footnotesize{" + format_W(emissions_joint[i]) + "}}"
    to_tex += " \\\\ \n"

    to_tex += "\\footnotesize{$VOLL \\times \\Delta \\text{blackouts}$}"
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(blackouts_carbon_tax_alone[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(blackouts_carbon_tax_alone_highpricecap[i]) + "}}"
    to_tex += " & "
    to_tex += " & & \\onslide<2->{\\footnotesize{" + format_W(blackouts_carbon_tax_alone_battery[i]) + "}}"
    to_tex += " & "
    to_tex += " & & & \\onslide<2->{\\footnotesize{" + format_W(blackouts_joint[i]) + "}}"
    to_tex += " \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

# create_file(gv.tables_path + "optimal_policy_presentation.tex", to_tex)

if show_output:
    print(to_tex)
