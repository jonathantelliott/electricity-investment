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

# %%
# Variables governing how script is run

show_output = False
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
with np.load(f"{gv.arrays_path}counterfactual_env_renewablesubisidies.npz") as loaded:
    renewable_subsidies_linspace = loaded['renewable_subsidies_linspace']
with np.load(f"{gv.arrays_path}counterfactual_results_renewableinvestmentsubisidies.npz") as loaded:
    renewable_investment_subsidy_linspace = loaded['renewable_investment_subsidy_linspace']
with np.load(f"{gv.arrays_path}counterfactual_results_delay.npz") as loaded:
    delay_linspace = loaded['delay_linspace']

carbon_tax_capacity_payment_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results.npz") as loaded:
    carbon_tax_capacity_payment_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    carbon_tax_capacity_payment_results['expected_emissions'] = loaded['expected_emissions']
    carbon_tax_capacity_payment_results['expected_blackouts'] = loaded['expected_blackouts']
    carbon_tax_capacity_payment_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    carbon_tax_capacity_payment_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    carbon_tax_capacity_payment_results['expected_total_produced'] = loaded['expected_total_produced']
    carbon_tax_capacity_payment_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    carbon_tax_capacity_payment_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus']
    carbon_tax_capacity_payment_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    carbon_tax_capacity_payment_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    carbon_tax_capacity_payment_results['expected_revenue'] = loaded['expected_revenue']
    carbon_tax_capacity_payment_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    carbon_tax_capacity_payment_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum']
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
    renewable_production_subsidies_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus']
    renewable_production_subsidies_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    renewable_production_subsidies_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    renewable_production_subsidies_results['expected_revenue'] = loaded['expected_revenue']
    renewable_production_subsidies_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    renewable_production_subsidies_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum']
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
    renewable_investment_subsidies_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus']
    renewable_investment_subsidies_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    renewable_investment_subsidies_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    renewable_investment_subsidies_results['expected_revenue'] = loaded['expected_revenue']
    renewable_investment_subsidies_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    renewable_investment_subsidies_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum']
    renewable_investment_subsidies_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    renewable_investment_subsidies_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    renewable_investment_subsidies_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    renewable_investment_subsidies_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    
delay_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results_delay_smoothed.npz") as loaded:
    delay_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    delay_results['expected_emissions'] = loaded['expected_emissions']
    delay_results['expected_blackouts'] = loaded['expected_blackouts']
    delay_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    delay_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    delay_results['expected_total_produced'] = loaded['expected_total_produced']
    delay_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    delay_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus']
    delay_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    delay_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    delay_results['expected_revenue'] = loaded['expected_revenue']
    delay_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    delay_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum']
    delay_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    delay_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    delay_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    delay_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    delay_results['expected_total_production_cost'] = loaded['expected_total_production_cost']
    
high_price_cap_results = {}
with np.load(f"{gv.arrays_path}counterfactual_results_highpricecap.npz") as loaded:
    high_price_cap_results['expected_agg_source_capacity'] = loaded['expected_agg_source_capacity']
    high_price_cap_results['expected_emissions'] = loaded['expected_emissions']
    high_price_cap_results['expected_blackouts'] = loaded['expected_blackouts']
    high_price_cap_results['expected_frac_by_source'] = loaded['expected_frac_by_source']
    high_price_cap_results['expected_quantity_weighted_avg_price'] = loaded['expected_quantity_weighted_avg_price']
    high_price_cap_results['expected_total_produced'] = loaded['expected_total_produced']
    high_price_cap_results['expected_misallocated_demand'] = loaded['expected_misallocated_demand']
    high_price_cap_results['expected_consumer_surplus'] = loaded['expected_consumer_surplus']
    high_price_cap_results['expected_carbon_tax_revenue'] = loaded['expected_carbon_tax_revenue']
    high_price_cap_results['expected_capacity_payments'] = loaded['expected_capacity_payments']
    high_price_cap_results['expected_revenue'] = loaded['expected_revenue']
    high_price_cap_results['expected_producer_surplus_sum'] = loaded['expected_producer_surplus_sum']
    high_price_cap_results['expected_consumer_surplus_sum'] = loaded['expected_consumer_surplus_sum']
    high_price_cap_results['expected_revenue_sum'] = loaded['expected_revenue_sum']
    high_price_cap_results['expected_product_market_sum'] = loaded['expected_product_market_sum']
    high_price_cap_results['expected_emissions_sum'] = loaded['expected_emissions_sum']
    high_price_cap_results['expected_blackouts_sum'] = loaded['expected_blackouts_sum']
    high_price_cap_results['expected_total_production_cost'] = loaded['expected_total_production_cost']
    
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
            axs[0,0].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, cspsg[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i], label=labels[i])
    else:
        axs[0,0].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, cspsg)(x_axis_linspace_fine), color=colors[0], lw=lw, ls=ls[0])
    axs[0,0].set_xlabel(f"{x_axis_label[0]}", fontsize=fontsize)
    axs[0,0].set_ylabel(f"{y_axis_label[0]}", fontsize=fontsize)
    axs[0,0].set_title(f"$CS + PS + G$", fontsize=title_fontsize)
        
    # Plot emissions
    if multiple_lines:
        for i in range(emissions.shape[1]):
            axs[0,1].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, emissions[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i])
    else:
        axs[0,1].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, emissions)(x_axis_linspace_fine), color=colors[1], lw=lw, ls=ls[1])
    axs[0,1].set_xlabel(f"{x_axis_label[1]}", fontsize=fontsize)
    axs[0,1].set_ylabel(f"{y_axis_label[1]}", fontsize=fontsize)
    axs[0,1].set_title(f"emissions", fontsize=title_fontsize)
    
    # Plot blackouts
    if multiple_lines:
        for i in range(blackouts.shape[1]):
            axs[0,2].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, blackouts[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i])
    else:
        axs[0,2].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, blackouts)(x_axis_linspace_fine), color=colors[2], lw=lw, ls=ls[2])
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
            axs[0,0].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, emissions[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i], label=labels[i])
    else:
        axs[0,0].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, emissions)(x_axis_linspace_fine), color=colors[1], lw=lw, ls=ls[1])
    axs[0,0].set_xlabel(f"{x_axis_label[1]}", fontsize=fontsize)
    axs[0,0].set_ylabel(f"{y_axis_label[1]}", fontsize=fontsize)
    axs[0,0].set_title(f"emissions", fontsize=title_fontsize)
    
    # Plot blackouts
    if multiple_lines:
        for i in range(blackouts.shape[1]):
            axs[0,1].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, blackouts[:,i])(x_axis_linspace_fine), color=colors[i], lw=lw, ls=ls[i])
    else:
        axs[0,1].plot(x_axis_linspace_fine, interp.CubicSpline(x_axis_linspace, blackouts)(x_axis_linspace_fine), color=colors[2], lw=lw, ls=ls[2])
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
# Intro numbers

# Indices
capacity_payment_intro_idx = np.argmin(np.abs(capacity_payments_linspace - 150000.0))
carbon_tax_intro_idx = np.argmin(np.abs(carbon_taxes_linspace - 200.0))

# Carbon tax alone
carbon_tax_pct_emissions = (carbon_tax_capacity_payment_results['expected_emissions_sum'][carbon_tax_intro_idx,0] - carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]) / carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]
carbon_tax_pct_blackouts = (carbon_tax_capacity_payment_results['expected_blackouts_sum'][carbon_tax_intro_idx,0] - carbon_tax_capacity_payment_results['expected_blackouts_sum'][0,0]) / carbon_tax_capacity_payment_results['expected_blackouts_sum'][0,0]

# Capacity payments alone
capacity_payment_pct_emissions = (carbon_tax_capacity_payment_results['expected_emissions_sum'][0,capacity_payment_intro_idx] - carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]) / carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]
capacity_payment_pct_blackouts = (carbon_tax_capacity_payment_results['expected_blackouts_sum'][0,capacity_payment_intro_idx] - carbon_tax_capacity_payment_results['expected_blackouts_sum'][0,0]) / carbon_tax_capacity_payment_results['expected_blackouts_sum'][0,0]

# Carbon tax + capacity payments
carbon_tax_capacity_payment_pct_emissions = (carbon_tax_capacity_payment_results['expected_emissions_sum'][carbon_tax_intro_idx,capacity_payment_intro_idx] - carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]) / carbon_tax_capacity_payment_results['expected_emissions_sum'][0,0]
carbon_tax_capacity_payment_pct_blackouts = (carbon_tax_capacity_payment_results['expected_blackouts_sum'][carbon_tax_intro_idx,capacity_payment_intro_idx] - carbon_tax_capacity_payment_results['expected_blackouts_sum'][0,0]) / carbon_tax_capacity_payment_results['expected_blackouts_sum'][0,0]

# Save numbers
create_file(f"{gv.stats_path}carbon_tax_pct_emissions.tex", f"{(-carbon_tax_pct_emissions * 100.0):.1f}")
create_file(f"{gv.stats_path}carbon_tax_pct_blackouts.tex", f"{(carbon_tax_pct_blackouts * 100.0):.1f}")
create_file(f"{gv.stats_path}capacity_payment_pct_emissions.tex", f"{(capacity_payment_pct_emissions * 100.0):.1f}")
create_file(f"{gv.stats_path}capacity_payment_pct_blackouts.tex", f"{(-capacity_payment_pct_blackouts * 100.0):.1f}")
create_file(f"{gv.stats_path}carbon_tax_capacity_payment_pct_emissions.tex", f"{(-carbon_tax_capacity_payment_pct_emissions * 100.0):.1f}")
create_file(f"{gv.stats_path}carbon_tax_capacity_payment_pct_blackouts.tex", f"{(-carbon_tax_capacity_payment_pct_blackouts * 100.0):.1f}")

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

tax_idx = np.array([0, 2, 4, 6])#np.array([0, 5, 10])
cap_pay_idx = np.array([0, 1, 2, 3, 4])

format_carbon_tax = lambda x: f"{int(np.round(x))}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")
format_cs = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_ps = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_g = lambda x: "{:,}".format(np.round(x / 1000000000.0, 2)).replace(",","\\,")
format_emissions = lambda x: "{:,}".format(np.round(ton_to_kg(x) / 1000000000.0, 2)).replace(",","\\,")
format_blackouts = lambda x: "{:,}".format(np.round(x / 1000000.0, 2)).replace(",","\\,")

to_tex = "\\begin{tabular}{ccccccccccccccccc} \n"
to_tex += "\\hline \n"
to_tex += " & & & \\multicolumn{2}{c}{$\\Delta \\text{CS}$ (billions A\\$)} & & \\multicolumn{2}{c}{$\\Delta \\text{PS}$ (billions A\\$)} & & \\multicolumn{2}{c}{$\\Delta \\text{G}$ (billons A\\$)} & & \\multicolumn{2}{c}{$\\Delta$ emissions (billions kg $\\text{CO}_{2}$-eq)} & & \\multicolumn{2}{c}{$\\Delta$ blackouts (millions MWh)} \\\\ \n"
to_tex += "$\\tau$ & $\\kappa$ & & low price cap & high price cap & & low price cap & high price cap & & low price cap & high price cap & & low price cap & high price cap & & low price cap & high price cap \\\\ \n"
to_tex += "\\cline{1-1} \\cline{2-2} \\cline{4-5} \\cline{7-8} \\cline{10-11} \\cline{13-14} \\cline{16-17} \n"
to_tex += " & & & & & & & & & & & & & & & & \\\\ \n"

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
            to_tex += " & " + format_g(results_dict['expected_revenue_sum'][i,j] - carbon_tax_capacity_payment_results['expected_revenue_sum'][default_tax_idx,default_cap_pay_idx])
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_emissions(results_dict['expected_emissions_sum'][i,j] - carbon_tax_capacity_payment_results['expected_emissions_sum'][default_tax_idx,default_cap_pay_idx])
        to_tex += " & "
        for results_dict in results_use:
            to_tex += " & " + format_blackouts(results_dict['expected_blackouts_sum'][i,j] - carbon_tax_capacity_payment_results['expected_blackouts_sum'][default_tax_idx,default_cap_pay_idx])
        to_tex += " \\\\ \n"
        if (i != tax_idx[-1]) and (j == cap_pay_idx[-1]):
            to_tex += " & & & & & & & & & & & & & & & & \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

if save_output:
    create_file(gv.tables_path + "policy_welfare.tex", to_tex)
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

interp_CSPS = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
interp_blackouts = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, carbon_tax_capacity_payment_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
interp_CSPS_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_product_market_sum'], kx=3, ky=3, s=0.0)
interp_emissions_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_emissions_sum'], kx=3, ky=3, s=0.0)
interp_blackouts_highpricecap = interp.RectBivariateSpline(carbon_taxes_linspace, capacity_payments_linspace, high_price_cap_results['expected_blackouts_sum'], kx=3, ky=3, s=0.0)
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
E_W_compare = E_W[:,0,0]
max_policy = np.argmax(E_W[:,:,0], axis=1) # capacity payments = 0
carbon_tax_alone_price_cap = carbon_taxes_fine[max_policy]
max_w_carbon_tax_alone_price_cap = (np.max(E_W[:,:,0], axis=1) - E_W_compare) / num_people

# Low price cap
E_W = interp_CSPS(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
E_W_compare = E_W[:,0,0]
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_joint = carbon_taxes_fine[max_policy_1]
capacity_payment_joint = capacity_payments_fine[max_policy_2]
max_w_joint = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people

# High price cap
E_W = interp_CSPS_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - scc_vals[:,np.newaxis,np.newaxis] * interp_emissions_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:] - voll * interp_blackouts_highpricecap(carbon_taxes_fine, capacity_payments_fine)[np.newaxis,:,:]
E_W = np.reshape(E_W, (num_points_scc, carbon_taxes_fine.shape[0], capacity_payments_fine.shape[0]))
max_policy = np.argmax(np.reshape(E_W, (E_W.shape[0], -1)), axis=1)
max_policy_1, max_policy_2 = np.unravel_index(max_policy, tuple(list(E_W.shape)[1:]))
carbon_tax_pricecap = carbon_taxes_fine[max_policy_1]
capacity_payment_pricecap = capacity_payments_fine[max_policy_2]
max_w_pricecap = (np.max(np.reshape(E_W, (E_W.shape[0],-1)), axis=1) - E_W_compare) / num_people

format_carbon_tax = lambda x: f"{np.round(x, 1)}"
format_cap_pay = lambda x: "{:,}".format(int(np.round(x / 100.0) * 100.0)).replace(",","\\,")
format_W = lambda x: "{:,}".format(np.round(x, 2)).replace(",","\\,")
format_scc = lambda x: "{:,}".format(int(np.round(x))).replace(",","\\,")

to_tex = "\\begin{tabular}{ccccccccccccc} \n"
to_tex += "\\hline \\\\ \n"
to_tex += "\\multicolumn{2}{c}{carbon tax alone, low $\\bar{P}$} & & \\multicolumn{2}{c}{carbon tax alone, high $\\bar{P}$} & & \\multicolumn{3}{c}{joint policies, low $\\bar{P}$} & & \\multicolumn{3}{c}{joint policies, high $\\bar{P}$} \\\\ \n"
to_tex += " \\cline{1-2} \\cline{4-5} \\cline{7-9} \\cline{11-13} \n"
to_tex += "$\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ & & $\\tau^{*}$ & $\\kappa^{*}$ & $\\Delta \\mathcal{W}$ \\\\ \n"
to_tex += "\\hline \n"

for i, scc_val in enumerate(scc_vals):
    to_tex += format_carbon_tax(carbon_tax_alone[i])
    to_tex += " & " + format_W(max_w_carbon_tax_alone[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_alone_price_cap[i])
    to_tex += " & " + format_W(max_w_carbon_tax_alone_price_cap[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_joint[i])
    to_tex += " & " + format_cap_pay(capacity_payment_joint[i])
    to_tex += " & " + format_W(max_w_joint[i])
    to_tex += " & "
    to_tex += " & " + format_carbon_tax(carbon_tax_pricecap[i])
    to_tex += " & " + format_cap_pay(capacity_payment_pricecap[i])
    to_tex += " & " + format_W(max_w_pricecap[i])
    to_tex += " \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"

create_file(gv.tables_path + "optimal_policy.tex", to_tex)
create_file(gv.stats_path + "optimal_policy_scc.tex", format_scc(scc_vals[0]))
create_file(gv.stats_path + "optimal_policy_voll.tex", format_scc(voll))

if show_output:
    print(to_tex)

# %%
# Carbon tax delay counterfactuals

# Capacity (paper version)
tax_idx = np.array([3, 3, 3, 3])
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
delay_idx = np.array([1, 5, 9])
tax_idx_use = 3
create_file(gv.stats_path + "counterfactuals_delay_graph_carbon_tax.tex", f"{int(carbon_taxes_linspace[tax_idx_use]):,}".replace(",","\\,"))
ls = [ls_all[i] for i in range(delay_idx.shape[0])]
#colors = ["black" for i in range(delay_idx.shape[0])]
cmap = cm.get_cmap("Greys")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
labels = [f"years delayed = {delay}" for delay in delay_linspace[delay_idx]]
titles = [f"$\\Delta CS_{{t}}$", f"$\\Delta P_{{t}}$"]
x_labels = [f"year", f"year"]
y_labels = [f"A\\$ (billions)", f"A\\$ / MWh"]
delta_p = delay_results['expected_quantity_weighted_avg_price'][tax_idx_use,:,:][delay_idx,:] - delay_results['expected_quantity_weighted_avg_price'][0,0,:][np.newaxis,:]
delta_cs = (delay_results['expected_consumer_surplus'][tax_idx_use,:,:][delay_idx,:] - delay_results['expected_consumer_surplus'][0,0,:][np.newaxis,:]) / 1000000000.0
delta_production_cost = (delay_results['expected_total_production_cost'][tax_idx_use,:,:][delay_idx,:] - delay_results['expected_total_production_cost'][0,0,:][np.newaxis,:])
var_arr = np.concatenate((delta_cs[np.newaxis,:,:], delta_p[np.newaxis,:,:]), axis=0)
filename = f"{gv.graphs_path}cs_delay.pdf"
plot_general_over_time(var_arr, indices_use, labels, ls, colors, lw_paper, fontsize_paper, title_fontsize_paper, titles, x_labels, y_labels, filename)

# \Delta Prices and \Delta CS over time (presentation)
cmap = cm.get_cmap("Purples")
colors = [cmap(i) for i in np.linspace(0.33, 1.0, tax_idx.shape[0])]
filename = f"{gv.graphs_path}cs_delay_presentation.pdf"
plot_general_over_time(var_arr, indices_use, labels, ls, colors, lw_presentation, fontsize_presentation, title_fontsize_presentation, titles, x_labels, y_labels, filename)

# Welfare table
to_tex = "\\begin{tabular}{cccccccccccc} \n"
to_tex += "\\hline \n"
to_tex += " & & & $\\Delta \\text{CS}$ & & $\\Delta \\text{PS}$ & & $\\Delta \\text{G}$ & & $\\Delta$ emissions & & $\\Delta$ blackouts \\\\ \n"
to_tex += "$\\tau$ & delay & &  (billions A\\$) & & (billions A\\$) & & (billons A\\$) & & (billions kg $\\text{CO}_{2}$-eq) & & (millions MWh) \\\\ \n"
to_tex += "\\hline \n"
to_tex += " & & & & & & & & & & & \\\\ \n"
tax_idx = np.array([2, 4, 6])
delay_idx = np.array([1, 5, 9])
for i in tax_idx:
    to_tex += format_carbon_tax(carbon_taxes_linspace[i])
    for j in delay_idx:
        to_tex += " & " + f"{delay_linspace[j]}"
        to_tex += " & "
        to_tex += " & " + format_cs(delay_results['expected_consumer_surplus_sum'][i,j] - delay_results['expected_consumer_surplus_sum'][0,0])
        to_tex += " & "
        to_tex += " & " + format_ps(delay_results['expected_producer_surplus_sum'][i,j] - delay_results['expected_producer_surplus_sum'][0,0])
        to_tex += " & "
        to_tex += " & " + format_g(delay_results['expected_revenue_sum'][i,j] - delay_results['expected_revenue_sum'][0,0])
        to_tex += " & "
        to_tex += " & " + format_emissions(delay_results['expected_emissions_sum'][i,j] - delay_results['expected_emissions_sum'][0,0])
        to_tex += " & "
        to_tex += " & " + format_blackouts(delay_results['expected_blackouts_sum'][i,j] - delay_results['expected_blackouts_sum'][0,0])
        to_tex += " \\\\ \n"
        if (i != tax_idx[-1]) and (j == delay_idx[-1]):
            to_tex += " & & & & & & & & & & & \\\\ \n"

to_tex += "\\hline \n"
to_tex += "\\end{tabular}"
if show_output:
    print(to_tex)
if save_output:
    create_file(gv.tables_path + "delay_welfare_table.tex", to_tex)
    
# %%
# Optimal carbon taxes / capacity payments

# Set up the space would like to consider
min_carbon_tax = 0.0
max_carbon_tax = 300.0
num_points = 500 # need large number
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
to_tex += "(billions kg$\\text{CO}_{2}$-eq) & policy & policy value & (millions MWh) & (billions A\\$) & (billions A\\$) & (billions AUD) & (billions A\\$) \\\\ \n"
to_tex += "\\hline \n"

emissions_array = np.linspace(0.0, 35.0, 8)#np.linspace(0.0, 25.0, 6)
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
    create_file(gv.tables_path + "compare_env_policies.tex", to_tex)
if show_output:
    print(to_tex)
    
