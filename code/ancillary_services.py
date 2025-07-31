# %%
# Import packages
import global_vars as gv

import numpy as np
import pandas as pd

# %%
# Import data

dates = np.load(gv.dates_file)
facilities = np.load(gv.facilities_file)
participants = np.load(gv.participants_file)
capacities = np.load(gv.capacities_file)
energy_sources = np.load(gv.energy_sources_file)
exited_date = np.load(gv.exited_date_file)
entered_date = np.load(gv.entered_date_file)
loaded = np.load(gv.energy_gen_file)
energy_gen = np.copy(loaded['arr_0'])
loaded.close()
capacity_price = np.load(gv.capacity_price_file)
cap_date_from = np.load(gv.cap_date_from_file)
cap_date_until = np.load(gv.cap_date_until_file)
prices = np.load(gv.prices_realtime_file) # np.load(gv.prices_file)
print(f"Finished importing data.", flush=True)

# %%
# Fraction of revenue is ancillary services

# Wholesale revenues
revenues = np.nansum(np.nansum(energy_gen, axis=0) * prices, axis=1)
wholesale_revenues = {}
for i in range(2006, 2022):
    wholesale_revenues[f'{i}-{i+1}'] = np.sum(revenues[np.isin(dates, np.arange(np.datetime64(f"{i}-04-01"), np.datetime64(f"{i+1}-04-01")))])

# Ancillary services, hand collected from annual reports
ancillary_services = { # sometimes later reports provide updated numbers, I use the most recent reported numbers
    '2006-2007': 11_562_153, 
    '2007-2008': 16_015_053.90, 
    '2008-2009': 38_090_841.66, 
    '2009-2010': 23_590_466.73, 
    '2010-2011': 37_378_176.03, 
    '2011-2012': 44_099_475.26, 
    '2012-2013': 85_094_329.46, 
    '2013-2014': 72_313_959.77, 
    '2014-2015': 60_201_772.14, 
    '2015-2016': 56_901_314, 
    '2016-2017': 82_075_608, 
    '2017-2018': 103_544_808, 
    '2018-2019': 107_225_801, 
    '2019-2020': 95_491_168, 
    '2020-2021': 91_758_854,  
    '2021-2022': 88_001_590, 
}

# Capacity payments
dates_capacity_price = []
capacity_price_extended = []
for start, end, price in zip(cap_date_from, cap_date_until, capacity_price):
    dates_subset = pd.date_range(start=start, end=end, freq="D").to_numpy().astype("datetime64[D]")
    dates_capacity_price.extend(dates_subset)
    capacity_price_extended.extend([price] * len(dates_subset))
dates_capacity_price = np.array(dates_capacity_price, dtype="datetime64[D]")
capacity_price_extended = np.array(capacity_price_extended)
daily_capacity_payment = capacity_price_extended[np.newaxis,:] * capacities[:,np.newaxis] * ~np.isin(energy_sources, gv.intermittent)[:,np.newaxis] * (entered_date[:,np.newaxis] <= dates_capacity_price[np.newaxis,:]) * (exited_date[:,np.newaxis] >= dates_capacity_price[np.newaxis,:]) / 365.0
sum_daily_capacity_payment = np.sum(daily_capacity_payment, axis=0)
capacity_payments = {}
for i in range(2006, 2022):
    capacity_payments[f'{i}-{i+1}'] = np.sum(sum_daily_capacity_payment[np.isin(dates_capacity_price, np.arange(np.datetime64(f"{i}-04-01"), np.datetime64(f"{i+1}-04-01")))])

# Print fraction of revenue ancillary services are
for i in range(2006, 2022):
    key = f'{i}-{i+1}'
    print(f"{key}: {np.round(ancillary_services[key] / (wholesale_revenues[key] + ancillary_services[key] + capacity_payments[key]) * 100.0, 2)}%")

# Print min and max
ratios = []
for i in range(2006, 2022):
    key = f'{i}-{i+1}'
    ratio = ancillary_services[key] / (wholesale_revenues[key] + ancillary_services[key] + capacity_payments[key])
    ratios.append(ratio)
    print(f"{key}: {np.round(ratio * 100.0, 2)}%")
ratios = np.array(ratios)
print("")
print(f"Min:  {np.round(np.min(ratios) * 100.0, 1)}%")
print(f"Max:  {np.round(np.max(ratios) * 100.0, 1)}%")

# Print share on average
total_wholesale_revenues = np.sum(list(wholesale_revenues.values()))
total_capacity_payments = np.sum(list(capacity_payments.values()))
total_ancillary_services = np.sum(list(ancillary_services.values()))
print(f"Mean: {np.round(total_ancillary_services / (total_wholesale_revenues + total_capacity_payments + total_ancillary_services) * 100.0, 1)}%")

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
create_file(f"{gv.stats_path}ancillary_services_share.tex", f"{np.round(total_ancillary_services / (total_wholesale_revenues + total_capacity_payments + total_ancillary_services) * 100.0, 1)}")
