# %%
# Import packages
import global_vars as gv

import numpy as np
import pandas as pd

# %%
# Import data
dates = np.load(gv.dates_file)
facilities = np.load(gv.facilities_file)
energy_sources = np.load(gv.energy_sources_file)
loaded = np.load(gv.energy_gen_file)
energy_gen = np.copy(loaded['arr_0'])
loaded.close()
prices = np.load(gv.prices_realtime_file) # np.load(gv.prices_file)
balancing_avail = np.load(gv.balancing_avail_file)
max_prices_notinflationadjusted = np.load(gv.max_prices_file)
min_prices_notinflationadjusted = np.load(gv.min_prices_file)

# Initialize arrays and counters
balancing_facilities = np.copy(facilities)
balancing_energy_sources = np.copy(energy_sources)
T = dates.shape[0]
H = gv.num_intervals_in_day
balancing_bid_min = np.ones((balancing_facilities.shape[0], T, H)) * np.nan
balancing_bid_max = np.ones((balancing_facilities.shape[0], T, H)) * np.nan
balancing_bid_q = np.ones((balancing_facilities.shape[0], T, H)) * np.nan
yr = gv.start_year
month = gv.start_month
ctr = 0 # to keep track of the date

# Go through each month and create a dataframe of quantity of energy produced by each facility
df_trading_col = "Trading Date"
df_interval_col = "Interval Number"
df_participant_col = "Participant Code"
df_facility_col = "Facility Code"
df_bid_offer_col = "Bid or Offer"
price_col = "Submitted Price ($/MWh)"
q_col = "Submitted Quantity (MW)"
bmo_q_col = "BMO Quantity (MW)"
type_col = "Type"
max_val = "MAX"
min_val = "MIN"
unavail_val = "UNAV"
rpqty_val = "RPQTY"
maxcap_val = "MAXCAP_MINUS_RPQTY"
while yr <= gv.end_year and (month <= gv.end_month or yr < gv.end_year):
    yrmonth_str = str(yr).zfill(4) + "-" + str(month).zfill(2)
    print(f"{yrmonth_str}")
    nextmonth_yr = yr if month < 12 else yr + 1
    nextmonth_month = month + 1 if month < 12 else 1
    dates_in_monthyr = np.arange(np.datetime64(str(yr).zfill(4) + "-" + str(month).zfill(2) + "-" + str(1).zfill(2)),
                                 np.datetime64(str(nextmonth_yr).zfill(4) + "-" + str(nextmonth_month).zfill(2) + "-" + str(1).zfill(2)))
    
    if (yr > 2012) or ((yr == 2012) and (month >= 7)): # once balancing market began
        # Import relevant csv file
        df_balancing_myr = (pd.read_csv(gv.data_path + "effective_balancing_submission/effective-balancing-submission-" + yrmonth_str + ".csv")
                            .sort_values(by=[df_trading_col, df_interval_col, df_facility_col]))
        df_balancing_myr = df_balancing_myr[df_balancing_myr[type_col] == "Balancing"] # drop non-balancing (i.e., unavailable) types

        # Replace MAX, MIN, missing with NaN
        df_maxmin_prices = pd.DataFrame({df_trading_col: dates_in_monthyr, 'max_price': max_prices_notinflationadjusted[np.isin(dates, dates_in_monthyr)], 'min_price': min_prices_notinflationadjusted[np.isin(dates, dates_in_monthyr)]})
        df_balancing_myr[df_trading_col] = pd.to_datetime(df_balancing_myr[df_trading_col]) # object -> datettime (needed for merge)
        df_balancing_myr = df_balancing_myr.merge(df_maxmin_prices, on=df_trading_col, how="left")
        max_val_rows = df_balancing_myr[price_col] == max_val
        df_balancing_myr.loc[max_val_rows, price_col] = np.nan#df_balancing_myr['max_price'][max_val_rows]
        min_val_rows = df_balancing_myr[price_col] == min_val
        df_balancing_myr.loc[min_val_rows, price_col] = np.nan#df_balancing_myr['min_price'][min_val_rows]
        df_balancing_myr.loc[df_balancing_myr[price_col] == unavail_val, price_col] = np.nan
        df_balancing_myr[price_col] = pd.to_numeric(df_balancing_myr[price_col])
        rpqty_rows = (df_balancing_myr[q_col] == rpqty_val) | (df_balancing_myr[q_col] == maxcap_val)
        df_balancing_myr.loc[rpqty_rows, q_col] = df_balancing_myr[bmo_q_col][rpqty_rows]
        df_balancing_myr.loc[rpqty_rows, q_col] = df_balancing_myr[bmo_q_col][rpqty_rows]
        df_balancing_myr[q_col] = pd.to_numeric(df_balancing_myr[q_col])
        
        # Need to update balancing_facilities?
        balancing_facilities_myr = np.unique(df_balancing_myr[df_facility_col])
        balancing_facilities_new = balancing_facilities_myr[~np.isin(balancing_facilities_myr, balancing_facilities)]
        balancing_facilities_new_num = balancing_facilities_new.shape[0]
        if balancing_facilities_new_num > 0:
            balancing_bid_min = np.concatenate((balancing_bid_min, np.ones((balancing_facilities_new_num, T, H)) * np.nan), axis=0)
            balancing_bid_max = np.concatenate((balancing_bid_max, np.ones((balancing_facilities_new_num, T, H)) * np.nan), axis=0)
            balancing_bid_q = np.concatenate((balancing_bid_max, np.ones((balancing_facilities_new_num, T, H)) * np.nan), axis=0)
            balancing_facilities = np.concatenate((balancing_facilities, balancing_facilities_new))
            balancing_energy_sources = np.concatenate((energy_sources, np.array([""] * balancing_facilities_new_num)))
            balancing_facilities_argsort = np.argsort(balancing_facilities)
            balancing_facilities = balancing_facilities[balancing_facilities_argsort]
            balancing_energy_sources = balancing_energy_sources[balancing_facilities_argsort]
            balancing_bid_min = balancing_bid_min[balancing_facilities_argsort,:,:] # resort
            balancing_bid_max = balancing_bid_max[balancing_facilities_argsort,:,:] # resort
            balancing_bid_q = balancing_bid_max[balancing_facilities_argsort,:,:] # resort

        # Process bids
        df_balancing_myr = df_balancing_myr[[df_facility_col, df_trading_col, df_interval_col, price_col, q_col]]
        df_balancing_myr.dropna(inplace=True)
        df_balancing_myr = df_balancing_myr.groupby([df_facility_col, df_trading_col, df_interval_col]).agg({price_col: ["min", "max"], q_col: ["sum"]}).reset_index()
        mux = pd.MultiIndex.from_product([balancing_facilities, dates_in_monthyr, range(1, H+1)], names=(df_facility_col,df_trading_col,df_interval_col))
        group = mux.to_frame(index=False).merge(df_balancing_myr, on=[df_facility_col, df_trading_col, df_interval_col], how="left").fillna(np.nan) # add the facility codes that are missing, also puts them in correct order
        
        # Add to balancing bids array
        balancing_bid_min[:,ctr:ctr+dates_in_monthyr.shape[0],:] = group[(price_col, "min")].values.reshape((balancing_facilities.shape[0],dates_in_monthyr.shape[0],H))
        balancing_bid_max[:,ctr:ctr+dates_in_monthyr.shape[0],:] = group[(price_col, "max")].values.reshape((balancing_facilities.shape[0],dates_in_monthyr.shape[0],H))
        balancing_bid_q[:,ctr:ctr+dates_in_monthyr.shape[0],:] = group[(q_col, "sum")].values.reshape((balancing_facilities.shape[0],dates_in_monthyr.shape[0],H))

    # Update month + year
    ctr += dates_in_monthyr.shape[0]
    yr = yr if month < 12 else yr + 1
    month = month + 1 if month < 12 else 1
    
def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()

slope = (balancing_bid_max - balancing_bid_min) / balancing_bid_q
slope[np.isinf(slope)] = np.nan
avg_slope = np.nanmean(slope)
create_file(gv.stats_path + "bid_steepness.tex", f"{avg_slope:.2f}")
