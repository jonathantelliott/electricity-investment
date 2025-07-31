# %%
# Import packages
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

import global_vars as gv


# %%
# Variables for data processing

# Saving array info
save_arrays = True

# Dates to use
start_year = gv.start_year
start_month = gv.start_month
start_day = gv.start_day
end_year = gv.end_year
end_month = gv.end_month
end_day = gv.end_day


# %%
# Process dates

dates = np.arange(np.datetime64(str(start_year).zfill(4) + "-" + str(start_month).zfill(2) + "-" + str(start_day).zfill(2)), np.datetime64(str(end_year).zfill(4) + "-" + str(end_month).zfill(2) + "-" + str(end_day).zfill(2)) + 1)

if save_arrays:
    np.save(gv.dates_file, dates)
    
T = dates.shape[0]
H = gv.num_intervals_in_day

print(f"Dates processed", flush=True)

# %%
# Create temperature data

# Import data
df_temperature = pd.read_csv(gv.temp_data_path + "HM01X_Data_009225_9999999910680996.txt")
df_temperature.sort_values(by=['Year Month Day Hour Minutes in YYYY', 'MM', 'DD', 'HH24', 'MI format in Local time'], inplace=True)

# Save only dates in the array dates
df_temperature['date'] = pd.to_datetime(df_temperature[['Year Month Day Hour Minutes in YYYY', 'MM', 'DD']].rename(columns={'Year Month Day Hour Minutes in YYYY': 'year', 'MM': 'month', 'DD': 'day'}))
df_temperature = df_temperature[df_temperature['date'].isin(dates)]

# Save variables
df_temperature['Air Temperature in degrees C'] = pd.to_numeric(df_temperature['Air Temperature in degrees C'], errors='coerce')
df_temperature['Wet bulb temperature in degrees C'] = pd.to_numeric(df_temperature['Wet bulb temperature in degrees C'], errors='coerce')
air_temps = np.reshape(df_temperature['Air Temperature in degrees C'].values, (T, H))
wet_bulb_temps = np.reshape(df_temperature['Wet bulb temperature in degrees C'].values, (T, H))
if save_arrays:
    np.save(gv.air_temps_file, air_temps)
    np.save(gv.wet_bulb_temps_file, wet_bulb_temps)

print(f"Temperatures processed", flush=True)

# %%
# Create facility characteristics

# Construct dictionary of facility-interval level data (SCADA and STEM bids/offers), with an entry being a month
df_scada = {}
df_stem = {}
yr = start_year
month = start_month
num_months = 0
df_trading_col = "Trading Date"
df_interval_col = "Interval Number"
df_participant_col = "Participant Code"
df_facility_col = "Facility Code"
df_bid_offer_col = "Bid or Offer"
df_stem_price_col = "Price ($/MWh)"
while yr <= end_year and (month <= end_month or yr < end_year):
    
    # Import month-year bids/offers datafile
    yrmonth_str = str(yr).zfill(4) + "-" + str(month).zfill(2)
    df_scada[yrmonth_str] = (pd.read_csv(gv.data_path + "facility_scada/facility-scada-" + yrmonth_str + ".csv", low_memory=False)
                               .sort_values(by=[df_trading_col, df_interval_col, df_participant_col, df_facility_col]))
    df_stem[yrmonth_str] = (pd.read_csv(gv.data_path + "stem_bids_and_offers/stem-bids-and-offers-" + yrmonth_str + ".csv", low_memory=False)
                              .sort_values(by=[df_trading_col, df_interval_col, df_participant_col, df_bid_offer_col, df_stem_price_col]))
    
    # Update month + year
    yr = yr if month < 12 else yr + 1
    month = month + 1 if month < 12 else 1
    num_months += 1

# Create DataFrame with facility data
df_facilities = pd.read_csv(gv.data_path + "facilities/facilities_2022.csv")

# Determine facility characteristics
# Initialize facility arrays
facilities = np.array([], dtype=str)
participants = np.array([], dtype=str)
maximum_generation = np.array([])
df_max_gen_col = "Energy Generated (MWh)"

# Go through the monthly facility SCADA and populate facility arrays
for key in df_scada:
    # Aggregate by facility, saving the maximum amount generated in a half-hour
    df_scada[key] = df_scada[key][[df_facility_col, df_participant_col, df_trading_col, df_interval_col, df_max_gen_col]]
    df_scada[key][df_max_gen_col] = df_scada[key][df_max_gen_col].apply(pd.to_numeric, errors="coerce")
    df_temp = df_scada[key].groupby([df_facility_col, df_participant_col]) \
                           .agg(["max"]) \
                           .reset_index()

    # Populate temporary arrays with unique information from the month
    thisdf_facilities, thisdf_facilities_idx = np.unique(df_temp[df_facility_col], return_index=True)
    facilities_concat = np.concatenate((facilities, thisdf_facilities))
    participants_concat = np.concatenate((participants, df_temp[df_participant_col].values[thisdf_facilities_idx]))
    maximum_generation_concat = np.concatenate((maximum_generation, df_temp[df_max_gen_col].values[thisdf_facilities_idx,0]))

    # Determine the maximum generation for each facility so far
    df_max_gen = pd.DataFrame({df_facility_col: facilities_concat, 
                               df_participant_col: participants_concat, 
                               df_max_gen_col: maximum_generation_concat}) \
                   .groupby(df_facility_col) \
                   .agg({df_participant_col: "last", # doesn't matter which chosen
                         df_max_gen_col: "max"}) \
                   .reset_index()
    facilities, facilities_idx = np.unique(df_max_gen[df_facility_col], return_index=True)
    participants = df_max_gen[df_participant_col].values[facilities_idx]
    maximum_generation = df_max_gen[df_max_gen_col].values[facilities_idx]

# Create facility characteristics based on imputed characteristics (above) and provided characteristics (df_facilities), with priority going to provided if available
df_all_facilities = pd.DataFrame({df_facility_col: facilities, 
                                  df_participant_col: participants, 
                                  df_max_gen_col: maximum_generation * 2.0}) # the x2 gets from a half-hour measurement to an hourly measurement
df_facilities = df_facilities.merge(right=df_all_facilities, how="right", on=df_facility_col, indicator=True)

# Drop facilities that show up only a few times
drop_these = np.array(["ALINTA_WGP_AGG", "MHPS", "ALCOA_KW_IL", "ALCOA_PNJ_IL"]) # facilities that show up only a few times - I think it's a case of improper naming
df_facilities.drop(df_facilities.loc[np.isin(df_facilities[df_facility_col], drop_these)].index, inplace=True)

# Save each facility's characteristics based on 
facilities, facilities_idx = np.unique(df_facilities[df_facility_col], return_index=True)
participants = df_facilities[df_participant_col + "_y"].values[facilities_idx] # _y b/c there were two columns w/ same name, use df_all_facilities version
capacities = df_facilities[df_max_gen_col].values[facilities_idx]
exited = (df_facilities['_merge'] == "right_only").values[facilities_idx]

# Create DataFrame of facility types
df_energy_source_col = "Energy Source"
wind = gv.wind
solar = gv.solar
gas_ocgt = gv.gas_ocgt
gas_ccgt = gv.gas_ccgt
gas_cogen = gv.gas_cogen
coal = gv.coal
landfill = gv.landfill
distillate = gv.distillate
df_facility_type = pd.DataFrame({df_facility_col: facilities, df_energy_source_col: ""})

# Identify non-wind & PV from AEMO website https://aemo.com.au/en/energy-systems/electricity/wholesale-electricity-market-wem/data-wem/data-dashboard#generation-facilities
energy_source_replace_dict = {
    "ALBANY_WF1": wind, 
    "ALCOA_WGP": gas_cogen, 
    "AMBRISOLAR_PV1": solar, 
    "ATLAS": landfill, 
    "BADGINGARRA_WF1": wind, 
    "BLAIRFOX_BEROSRD_WF1": wind, 
    "BW2_BLUEWATERS_G1": coal, 
    "BW1_BLUEWATERS_G2": coal, 
    "BREMER_BAY_WF1": wind, 
    "BIOGAS01": landfill, 
    "COCKBURN_CCG1": gas_ccgt, 
    "INVESTEC_COLLGAR_WF1": wind, 
    "COLLIE_G1": coal, 
    "DCWL_DENMARK_WF1": wind, 
    "EDWFMAN_WF1": wind, 
    "PRK_AG": gas_ocgt, 
    "GOSNELLS": landfill, 
    "GRASMERE_WF1": wind, 
    "GREENOUGH_RIVER_PV1": solar, 
    "HENDERSON_RENEWABLE_IG1": landfill, 
    "KALAMUNDA_SG": distillate, 
    "KALBARRI_WF1": wind, 
    "BLAIRFOX_KARAKIN_WF1": wind, 
    "KEMERTON_GT11": gas_ocgt, 
    "KEMERTON_GT12": gas_ocgt, 
    "NEWGEN_KWINANA_CCG1": gas_ccgt, 
    "PERTHENERGY_KWINANA_GT1": gas_ocgt, 
    "KWINANA_GT2": gas_ocgt, 
    "KWINANA_GT3": gas_ocgt, 
    "NAMKKN_MERR_SG1": distillate, 
    "MERSOLAR_PV1": solar, 
    "SKYFRM_MTBARKER_WF1": wind, 
    "MUJA_G6": coal, 
    "MUJA_G7": coal, 
    "MUJA_G8": coal, 
    "MWF_MUMBIDA_WF1": wind, 
    "MUNGARRA_GT1": gas_ocgt,
    "MUNGARRA_GT3": gas_ocgt, 
    "NEWGEN_NEERABUP_GT1": gas_ocgt, 
    "NORTHAM_SF_PV1": solar, 
    "PINJAR_GT1": gas_ocgt, 
    "PINJAR_GT10": gas_ocgt, 
    "PINJAR_GT11": gas_ocgt, 
    "PINJAR_GT2": gas_ocgt, 
    "PINJAR_GT3": gas_ocgt, 
    "PINJAR_GT4": gas_ocgt, 
    "PINJAR_GT5": gas_ocgt, 
    "PINJAR_GT7": gas_ocgt, 
    "PINJAR_GT9": gas_ocgt, 
    "ALINTA_PNJ_U1": gas_cogen, 
    "ALINTA_PNJ_U2": gas_cogen, 
    "RED_HILL": landfill, 
    "ROCKINGHAM": landfill, 
    "SOUTH_CARDUP": landfill, 
    "STHRNCRS_EG": gas_ocgt, 
    "TAMALA_PARK": landfill, 
    "TESLA_GERALDTON_G1": distillate, 
    "TESLA_KEMERTON_G1": distillate, 
    "TESLA_NORTHAM_G1": distillate, 
    "TESLA_PICTON_G1": distillate, 
    "TIWEST_COG1": gas_cogen, 
    "ALINTA_WGP_GT": gas_ocgt, 
    "ALINTA_WGP_U2": gas_ocgt, 
    "ALINTA_WWF": wind, 
    "WARRADARGE_WF1": wind, 
    "BLAIRFOX_WESTHILLS_WF3": wind, 
    "WEST_KALGOORLIE_GT2": distillate, 
    "WEST_KALGOORLIE_GT3": distillate, 
    "YANDIN_WF1": wind
}
for key in energy_source_replace_dict:
    df_facility_type.loc[df_facility_type[df_facility_col].str.contains(key), df_energy_source_col] = energy_source_replace_dict[key]

# Identify facilities that have retired through alternative sources, using https://en.wikipedia.org/wiki/List_of_power_stations_in_Western_Australia unless o/w noted
energy_source_replace_dict = {
    "CANNING_MELVILLE": landfill, 
    "MUJA": coal, 
    "MUNGARRA": gas_ocgt, 
    "SWCJV_WORSLEY_COGEN_COG1": gas_cogen, 
    "KALAMUNDA": landfill, 
    "GERALDTON_GT1": distillate, 
    "KWINANA_GT1": gas_ocgt, 
    "ALCOA": gas_cogen, 
    "PPP_KCP_EG1": gas_cogen, #
    "KWINANA_G1": coal, 
    "KWINANA_G2": coal, 
    "KWINANA_G3": coal, # https://www.gem.wiki/Kwinana_Power_Station, used both coal and gas (and oil) but appears to have mostly used coal, need to note this in text
    "KWINANA_G4": coal, 
    "KWINANA_G5": coal, 
    "KWINANA_G6": coal
}
for key in energy_source_replace_dict:
    df_facility_type.loc[df_facility_type[df_facility_col].str.contains(key) & (df_facility_type[df_energy_source_col] == ""), df_energy_source_col] = energy_source_replace_dict[key]

# Relabel natural gas so that only two categories: CCGT and OCGT
# df_facility_type.loc[df_facility_type[df_energy_source_col] == gas_cogen, df_energy_source_col] = gas_ocgt # works b/c all cogeneration generators are OCGT, want to keep separate, production is a bit different, want to estimate separately

# Merge generator energy sources with facility data
df_facilities = df_facilities.merge(df_facility_type, on=df_facility_col, how="left")

# Save energy sources
energy_sources = df_facilities[df_energy_source_col].values[facilities_idx]

# Create DataFrame of facility heat rates
df_heat_rate_col = "Heat Rate (GJ/MWh)"
df_facility_type = pd.DataFrame({df_facility_col: facilities,df_heat_rate_col: np.nan})

# Identify heat rates from various sources
heat_rate_replace_dict = {
    "ALBANY_WF1": np.nan, 
    "ALCOA_WGP": 12.0, 
    "AMBRISOLAR_PV1": np.nan, 
    "ATLAS": 11.3, # landfill gas imputation
    "BADGINGARRA_WF1": np.nan, 
    "BLAIRFOX_BEROSRD_WF1": np.nan, 
    "BW2_BLUEWATERS_G1": 9.75, 
    "BW1_BLUEWATERS_G2": 9.75, 
    "BREMER_BAY_WF1": np.nan, 
    "BIOGAS01": 11.3, # landfill gas imputation
    "COCKBURN_CCG1": 9.0, 
    "INVESTEC_COLLGAR_WF1": np.nan, 
    "COLLIE_G1": 9.5, 
    "DCWL_DENMARK_WF1": np.nan, 
    "EDWFMAN_WF1": np.nan, 
    "PRK_AG": np.nan, # confidential
    "GOSNELLS": 11.3, # landfill gas imputation
    "GRASMERE_WF1": np.nan, 
    "GREENOUGH_RIVER_PV1": np.nan, 
    "HENDERSON_RENEWABLE_IG1": 11.3, # landfill gas imputation
    "KALAMUNDA_SG": 15.27, 
    "KALBARRI_WF1": np.nan, 
    "BLAIRFOX_KARAKIN_WF1": np.nan, 
    "KEMERTON_GT11": 12.2, 
    "KEMERTON_GT12": 12.2, 
    "NEWGEN_KWINANA_CCG1": 7.9, # Jha & Leslie
    "PERTHENERGY_KWINANA_GT1": 10.7, 
    "KWINANA_GT2": 9.35, 
    "KWINANA_GT3": 9.35, 
    "NAMKKN_MERR_SG1": 12.58, 
    "MERSOLAR_PV1": np.nan, 
    "SKYFRM_MTBARKER_WF1": np.nan, 
    "MWF_MUMBIDA_WF1": np.nan, 
    "MUNGARRA_GT1": 13.5,
    "MUNGARRA_GT2": 13.5,
    "MUNGARRA_GT3": 13.5, 
    "NEWGEN_NEERABUP_GT1": 11.1, # Jha & Leslie
    "NORTHAM_SF_PV1": np.nan, 
    "PINJAR_GT1": 13.5, 
    "PINJAR_GT10": 12.08, 
    "PINJAR_GT11": 12.01, 
    "PINJAR_GT2": 13.5, 
    "PINJAR_GT3": 13.2, 
    "PINJAR_GT4": 13.2, 
    "PINJAR_GT5": 13.2, 
    "PINJAR_GT7": 13.2, 
    "PINJAR_GT9": 12.08, 
    "ALINTA_PNJ_U1": 12.0, 
    "ALINTA_PNJ_U2": 12.0, 
    "RED_HILL": 11.3, # landfill gas imputation
    "ROCKINGHAM": 11.3, # landfill gas imputation
    "SOUTH_CARDUP": 11.3, # landfill gas imputation
    "STHRNCRS_EG": np.nan, 
    "TAMALA_PARK": 11.3, # landfill gas imputation
    "TESLA_GERALDTON_G1": 14.44, 
    "TESLA_KEMERTON_G1": 14.44, 
    "TESLA_NORTHAM_G1": 14.44, 
    "TESLA_PICTON_G1": 14.44, 
    "TIWEST_COG1": 13.0, 
    "ALINTA_WGP_GT": 11.5, 
    "ALINTA_WGP_U2": 11.5, 
    "ALINTA_WWF": np.nan, 
    "WARRADARGE_WF1": np.nan, 
    "BLAIRFOX_WESTHILLS_WF3": np.nan, 
    "WEST_KALGOORLIE_GT2": 13.5, 
    "WEST_KALGOORLIE_GT3": 14.75, 
    "YANDIN_WF1": np.nan, 
    "CANNING_MELVILLE": 11.3, # landfill gas imputation
    "MUJA_G1": np.nan, # confidential
    "MUJA_G2": np.nan, # confidential
    "MUJA_G3": np.nan, # confidential
    "MUJA_G4": np.nan, # confidential
    "MUJA_G5": 11.04, 
    "MUJA_G6": 11.04, 
    "MUJA_G7": 9.85, 
    "MUJA_G8": 9.85, 
    "SWCJV_WORSLEY_COGEN_COG1": 12.0, 
    "KALAMUNDA": 11.3, # landfill gas imputation
    "GERALDTON_GT1": 15.25, 
    "KWINANA_GT1": 14.1, 
    "PPP_KCP_EG1": 9.0, 
    "KWINANA_G1": 11.7, # imported from numbers for KWINANA_G5 and KWINANA_G6
    "KWINANA_G2": 11.7, # imported from numbers for KWINANA_G5 and KWINANA_G6
    "KWINANA_G3": 11.7, # imported from numbers for KWINANA_G5 and KWINANA_G6 
    "KWINANA_G4": 11.7, # imported from numbers for KWINANA_G5 and KWINANA_G6 
    "KWINANA_G5": 11.7, 
    "KWINANA_G6": 11.7
}
for key in heat_rate_replace_dict:
    df_facility_type.loc[df_facility_type[df_facility_col].str.contains(key), df_heat_rate_col] = heat_rate_replace_dict[key]

# Merge generator energy sources with facility data
df_facilities = df_facilities.merge(df_facility_type, on=df_facility_col, how="left")

# Save heat_rates
heat_rates = df_facilities[df_heat_rate_col].values[facilities_idx]

# save Kwinanas to be average of other Kwinanas
# heat_rates[(np.char.find(facilities.astype(str), "KWINANA") >= 0) & np.isnan(heat_rates)] = np.mean(heat_rates[(np.char.find(facilities.astype(str), "KWINANA") >= 0) & ~np.isnan(heat_rates)])

# save Mujas to be average of other Mujas
heat_rates[(np.char.find(facilities.astype(str), "MUJA") >= 0) & np.isnan(heat_rates)] = np.mean(heat_rates[(np.char.find(facilities.astype(str), "MUJA") >= 0) & ~np.isnan(heat_rates)])

# For others, use average of others in category
for source in np.unique(energy_sources):
    heat_rates[(energy_sources == source) & np.isnan(heat_rates)] = np.mean(heat_rates[(energy_sources == source) & ~np.isnan(heat_rates)])
    
# Create DataFrame of facility CO2 rates
df_co2_col = "CO2 rate (kg/MWh)"
df_facility_type = pd.DataFrame({df_facility_col: facilities,df_co2_col: np.nan})

# Identify CO2 rates from various sources
co2_replace_dict = {
    "ALBANY_WF1": 0.0, 
    "ALCOA_WGP": 627.0, 
    "AMBRISOLAR_PV1": 0.0, 
    "ATLAS": np.nan, # landfill gas imputation
    "BADGINGARRA_WF1": 0.0, 
    "BLAIRFOX_BEROSRD_WF1": 0.0, 
    "BW2_BLUEWATERS_G1": 908.0, 
    "BW1_BLUEWATERS_G2": 908.0, 
    "BREMER_BAY_WF1": 0.0, 
    "BIOGAS01": np.nan, # landfill gas imputation
    "COCKBURN_CCG1": 470.0, 
    "INVESTEC_COLLGAR_WF1": 0.0, 
    "COLLIE_G1": 884.0, 
    "DCWL_DENMARK_WF1": 0.0, 
    "EDWFMAN_WF1": 0.0, 
    "PRK_AG": np.nan, # confidential
    "GOSNELLS": np.nan, # landfill gas imputation
    "GRASMERE_WF1": 0.0, 
    "GREENOUGH_RIVER_PV1": 0.0, 
    "HENDERSON_RENEWABLE_IG1": np.nan, # landfill gas imputation
    "KALAMUNDA_SG": 1142.0, 
    "KALBARRI_WF1": 0.0, 
    "BLAIRFOX_KARAKIN_WF1": 0.0, 
    "KEMERTON_GT11": 638.0, 
    "KEMERTON_GT12": 638.0, 
    "NEWGEN_KWINANA_CCG1": np.nan, # Jha & Leslie
    "PERTHENERGY_KWINANA_GT1": 559.0, 
    "KWINANA_GT2": 486.0, 
    "KWINANA_GT3": 486.0, 
    "NAMKKN_MERR_SG1": 941.0, 
    "MERSOLAR_PV1": 0.0, 
    "SKYFRM_MTBARKER_WF1": 0.0, 
    "MWF_MUMBIDA_WF1": 0.0, 
    "MUNGARRA_GT1": 706.0,
    "MUNGARRA_GT2": 706.0,
    "MUNGARRA_GT3": 690.0, 
    "NEWGEN_NEERABUP_GT1": np.nan, # Jha & Leslie
    "NORTHAM_SF_PV1": 0.0, 
    "PINJAR_GT1": 706.0, 
    "PINJAR_GT10": 653.0, 
    "PINJAR_GT11": 638.0, 
    "PINJAR_GT2": 706.0, 
    "PINJAR_GT3": 690.0, 
    "PINJAR_GT4": 690.0, 
    "PINJAR_GT5": 690.0, 
    "PINJAR_GT7": 690.0, 
    "PINJAR_GT9": 653.0, 
    "ALINTA_PNJ_U1": 627.0, 
    "ALINTA_PNJ_U2": 627.0, 
    "RED_HILL": np.nan, # landfill gas imputation
    "ROCKINGHAM": np.nan, # landfill gas imputation
    "SOUTH_CARDUP": np.nan, # landfill gas imputation
    "STHRNCRS_EG": np.nan, 
    "TAMALA_PARK": np.nan, # landfill gas imputation
    "TESLA_GERALDTON_G1": 1080.0, 
    "TESLA_KEMERTON_G1": 1080.0, 
    "TESLA_NORTHAM_G1": 1080.0, 
    "TESLA_PICTON_G1": 1080.0, 
    "TIWEST_COG1": 679.0, 
    "ALINTA_WGP_GT": 601.0, 
    "ALINTA_WGP_U2": 601.0, 
    "ALINTA_WWF": 0.0, 
    "WARRADARGE_WF1": 0.0, 
    "BLAIRFOX_WESTHILLS_WF3": 0.0, 
    "WEST_KALGOORLIE_GT2": 1010.0, 
    "WEST_KALGOORLIE_GT3": 1103.0, 
    "YANDIN_WF1": 0.0, 
    "CANNING_MELVILLE": np.nan, # landfill gas imputation
    "MUJA_G1": np.nan, # confidential
    "MUJA_G2": np.nan, # confidential
    "MUJA_G3": np.nan, # confidential
    "MUJA_G4": np.nan, # confidential
    "MUJA_G5": 1028.0, 
    "MUJA_G6": 1028.0, 
    "MUJA_G7": 917.0, 
    "MUJA_G8": 917.0, 
    "SWCJV_WORSLEY_COGEN_COG1": 627.0, 
    "KALAMUNDA": np.nan, # landfill gas imputation
    "GERALDTON_GT1": 1141.0, 
    "KWINANA_GT1": 763.0, 
    "PPP_KCP_EG1": 470.0, 
    "KWINANA_G1": 850.0, 
    "KWINANA_G2": 850.0, 
    "KWINANA_G3": 850.0, 
    "KWINANA_G4": 850.0, 
    "KWINANA_G5": 850.0, 
    "KWINANA_G6": 850.0
}
for key in heat_rate_replace_dict:
    df_facility_type.loc[df_facility_type[df_facility_col].str.contains(key), df_co2_col] = co2_replace_dict[key]

# Merge generator energy sources with facility data
df_facilities = df_facilities.merge(df_facility_type, on=df_facility_col, how="left")

# Save co2 rates
co2_rates = df_facilities[df_co2_col].values[facilities_idx]

# save Kwinanas to be average of other Kwinanas
co2_rates[(np.char.find(facilities.astype(str), "KWINANA") >= 0) & np.isnan(co2_rates)] = np.mean(co2_rates[(np.char.find(facilities.astype(str), "KWINANA") >= 0) & ~np.isnan(co2_rates)])

# save Mujas to be average of other Mujas
co2_rates[(np.char.find(facilities.astype(str), "MUJA") >= 0) & np.isnan(co2_rates)] = np.mean(co2_rates[(np.char.find(facilities.astype(str), "MUJA") >= 0) & ~np.isnan(co2_rates)])

# For others, use average of others in category
for source in np.unique(energy_sources):
    co2_rates[(energy_sources == source) & np.isnan(co2_rates)] = np.mean(co2_rates[(energy_sources == source) & ~np.isnan(co2_rates)])
    
co2_rates[energy_sources == landfill] = 0.0 # won't really end up mattering b/c going to abstract away from landfill gas

# Create DataFrame of transport charges
df_transport_col = "transport charge ($/GJ)"
df_facility_type = pd.DataFrame({df_facility_col: facilities,df_transport_col: np.nan})

# Identify transport charges
transport_charge_replace_dict = {
    "ALBANY_WF1": 0.0, 
    "ALCOA_WGP": 1.09, 
    "AMBRISOLAR_PV1": 0.0, 
    "ATLAS": 0.0, 
    "BADGINGARRA_WF1": 0.0, 
    "BLAIRFOX_BEROSRD_WF1": 0.0, 
    "BW2_BLUEWATERS_G1": 0.0, 
    "BW1_BLUEWATERS_G2": 0.0, 
    "BREMER_BAY_WF1": 0.0, 
    "BIOGAS01": 0.0,
    "COCKBURN_CCG1": 1.09, 
    "INVESTEC_COLLGAR_WF1": 0.0, 
    "COLLIE_G1": 0.0, 
    "DCWL_DENMARK_WF1": 0.0, 
    "EDWFMAN_WF1": 0.0, 
    "PRK_AG": 3.94, 
    "GOSNELLS": 0.0, 
    "GRASMERE_WF1": 0.0, 
    "GREENOUGH_RIVER_PV1": 0.0, 
    "HENDERSON_RENEWABLE_IG1": 0.0, 
    "KALAMUNDA_SG": 0.0, 
    "KALBARRI_WF1": 0.0, 
    "BLAIRFOX_KARAKIN_WF1": 0.0, 
    "KEMERTON_GT11": 1.09, 
    "KEMERTON_GT12": 1.09, 
    "NEWGEN_KWINANA_CCG1": 1.09, 
    "PERTHENERGY_KWINANA_GT1": 1.09, 
    "KWINANA_GT2": 1.09, 
    "KWINANA_GT3": 1.09, 
    "NAMKKN_MERR_SG1": 0.0, 
    "MERSOLAR_PV1": 0.0, 
    "SKYFRM_MTBARKER_WF1": 0.0, 
    "MWF_MUMBIDA_WF1": 0.0, 
    "MUNGARRA_GT1": 0.80,
    "MUNGARRA_GT2": 0.80,
    "MUNGARRA_GT3": 0.80, 
    "NEWGEN_NEERABUP_GT1": 1.09, 
    "NORTHAM_SF_PV1": 0.0, 
    "PINJAR_GT1": 1.09, 
    "PINJAR_GT10": 1.09, 
    "PINJAR_GT11": 1.09, 
    "PINJAR_GT2": 1.09, 
    "PINJAR_GT3": 1.09, 
    "PINJAR_GT4": 1.09, 
    "PINJAR_GT5": 1.09, 
    "PINJAR_GT7": 1.09, 
    "PINJAR_GT9": 1.09, 
    "ALINTA_PNJ_U1": 1.09, 
    "ALINTA_PNJ_U2": 1.09, 
    "RED_HILL": 0.0, 
    "ROCKINGHAM": 0.0, 
    "SOUTH_CARDUP": 0.0, 
    "STHRNCRS_EG": np.nan, 
    "TAMALA_PARK": 0.0, 
    "TESLA_GERALDTON_G1": 0.0, 
    "TESLA_KEMERTON_G1": 0.0, 
    "TESLA_NORTHAM_G1": 0.0, 
    "TESLA_PICTON_G1": 0.0, 
    "TIWEST_COG1": 1.09, 
    "ALINTA_WGP_GT": 0.0, 
    "ALINTA_WGP_U2": 0.0, 
    "ALINTA_WWF": 0.0, 
    "WARRADARGE_WF1": 0.0, 
    "BLAIRFOX_WESTHILLS_WF3": 0.0, 
    "WEST_KALGOORLIE_GT2": 0.96, 
    "WEST_KALGOORLIE_GT3": 0.96, 
    "YANDIN_WF1": 0.0, 
    "CANNING_MELVILLE": 0.0, 
    "MUJA_G1": 0.0, 
    "MUJA_G2": 0.0, 
    "MUJA_G3": 0.0, 
    "MUJA_G4": 0.0, 
    "MUJA_G5": 0.0, 
    "MUJA_G6": 0.0, 
    "MUJA_G7": 0.0, 
    "MUJA_G8": 0.0, 
    "SWCJV_WORSLEY_COGEN_COG1": 1.09, 
    "KALAMUNDA": 0.0, 
    "GERALDTON_GT1": 0.0, 
    "KWINANA_GT1": 1.09, 
    "PPP_KCP_EG1": 1.09, 
    "KWINANA_G1": 1.09, # imputed from KWINANA_G5 and KWINANA_G6
    "KWINANA_G2": 1.09, # imputed from KWINANA_G5 and KWINANA_G6
    "KWINANA_G3": 1.09, # imputed from KWINANA_G5 and KWINANA_G6
    "KWINANA_G4": 1.09, # imputed from KWINANA_G5 and KWINANA_G6
    "KWINANA_G5": 1.09, 
    "KWINANA_G6": 1.09
}
for key in transport_charge_replace_dict:
    df_facility_type.loc[df_facility_type[df_facility_col].str.contains(key), df_transport_col] = transport_charge_replace_dict[key]

# Merge generator energy sources with facility data
df_facilities = df_facilities.merge(df_facility_type, on=df_facility_col, how="left")

# Create transport charges array
transport_charges = df_facilities[df_transport_col].values[facilities_idx]

# save Kwinanas to be average of other Kwinanas
transport_charges[(np.char.find(facilities.astype(str), "KWINANA") >= 0) & np.isnan(transport_charges)] = np.mean(transport_charges[(np.char.find(facilities.astype(str), "KWINANA") >= 0) & ~np.isnan(transport_charges)])

# save Mujas to be average of other Kwinanas
transport_charges[(np.char.find(facilities.astype(str), "MUJA") >= 0) & np.isnan(transport_charges)] = np.mean(transport_charges[(np.char.find(facilities.astype(str), "MUJA") >= 0) & ~np.isnan(transport_charges)])

# For others, use average of others in category
for source in np.unique(energy_sources):
    transport_charges[(energy_sources == source) & np.isnan(transport_charges)] = np.mean(transport_charges[(energy_sources == source) & ~np.isnan(transport_charges)])
    
# Create DataFrame of information sources
df_data_source_col = "data source"
df_facility_type = pd.DataFrame({df_facility_col: facilities,df_data_source_col: ""})

none = ""
skm = "skm"
skm_impute = "skm_impute"
skm_confidential = "skm_confidential"
jl = "jl"
info_source_replace_dict = {
    "ALBANY_WF1": none, 
    "ALCOA_WGP": skm, 
    "AMBRISOLAR_PV1": none, 
    "ATLAS": skm,
    "BADGINGARRA_WF1": none, 
    "BLAIRFOX_BEROSRD_WF1": none, 
    "BW2_BLUEWATERS_G1": skm, 
    "BW1_BLUEWATERS_G2": skm, 
    "BREMER_BAY_WF1": none, 
    "BIOGAS01": skm,
    "COCKBURN_CCG1": skm, 
    "INVESTEC_COLLGAR_WF1": none, 
    "COLLIE_G1": skm, 
    "DCWL_DENMARK_WF1": none, 
    "EDWFMAN_WF1": none, 
    "PRK_AG": skm_confidential, 
    "GOSNELLS": skm, 
    "GRASMERE_WF1": none, 
    "GREENOUGH_RIVER_PV1": none, 
    "HENDERSON_RENEWABLE_IG1": skm, 
    "KALAMUNDA_SG": skm, 
    "KALBARRI_WF1": none, 
    "BLAIRFOX_KARAKIN_WF1": none, 
    "KEMERTON_GT11": skm, 
    "KEMERTON_GT12": skm, 
    "NEWGEN_KWINANA_CCG1": jl, 
    "PERTHENERGY_KWINANA_GT1": skm, 
    "KWINANA_GT2": skm, 
    "KWINANA_GT3": skm, 
    "NAMKKN_MERR_SG1": skm, 
    "MERSOLAR_PV1": none, 
    "SKYFRM_MTBARKER_WF1": none, 
    "MWF_MUMBIDA_WF1": none, 
    "MUNGARRA_GT1": skm,
    "MUNGARRA_GT2": skm,
    "MUNGARRA_GT3": skm, 
    "NEWGEN_NEERABUP_GT1": jl, 
    "NORTHAM_SF_PV1": none, 
    "PINJAR_GT1": skm, 
    "PINJAR_GT10": skm, 
    "PINJAR_GT11": skm, 
    "PINJAR_GT2": skm, 
    "PINJAR_GT3": skm, 
    "PINJAR_GT4": skm, 
    "PINJAR_GT5": skm, 
    "PINJAR_GT7": skm, 
    "PINJAR_GT9": skm, 
    "ALINTA_PNJ_U1": skm, 
    "ALINTA_PNJ_U2": skm, 
    "RED_HILL": skm, 
    "ROCKINGHAM": skm, 
    "SOUTH_CARDUP": skm, 
    "STHRNCRS_EG": none, 
    "TAMALA_PARK": skm, 
    "TESLA_GERALDTON_G1": skm, 
    "TESLA_KEMERTON_G1": skm, 
    "TESLA_NORTHAM_G1": skm, 
    "TESLA_PICTON_G1": skm, 
    "TIWEST_COG1": skm, 
    "ALINTA_WGP_GT": skm, 
    "ALINTA_WGP_U2": skm, 
    "ALINTA_WWF": none, 
    "WARRADARGE_WF1": none, 
    "BLAIRFOX_WESTHILLS_WF3": none, 
    "WEST_KALGOORLIE_GT2": skm, 
    "WEST_KALGOORLIE_GT3": skm, 
    "YANDIN_WF1": none, 
    "CANNING_MELVILLE": skm, 
    "MUJA_G1": skm_confidential, 
    "MUJA_G2": skm_confidential, 
    "MUJA_G3": skm_confidential, 
    "MUJA_G4": skm_confidential, 
    "MUJA_G5": skm, 
    "MUJA_G6": skm, 
    "MUJA_G7": skm, 
    "MUJA_G8": skm, 
    "SWCJV_WORSLEY_COGEN_COG1": skm, 
    "KALAMUNDA": skm, 
    "GERALDTON_GT1": skm, 
    "KWINANA_GT1": skm, 
    "PPP_KCP_EG1": skm, 
    "KWINANA_G1": skm_impute, 
    "KWINANA_G2": skm_impute, 
    "KWINANA_G3": skm_impute, 
    "KWINANA_G4": skm_impute, 
    "KWINANA_G5": skm, 
    "KWINANA_G6": skm
}
for key in transport_charge_replace_dict:
    df_facility_type.loc[df_facility_type[df_facility_col].str.contains(key), df_data_source_col] = info_source_replace_dict[key]

# Merge generator energy sources with facility data
df_facilities = df_facilities.merge(df_facility_type, on=df_facility_col, how="left")

# Save heat_rates
info_sources = df_facilities[df_data_source_col].values[facilities_idx]

# Fix participant names
participant_name_replace_dict = {
    "GRIFFIN2": "GRIFFINP"
}
for key, value in participant_name_replace_dict.items():
    participants[participants == key] = value
    
# Convert object arrays into correct type
facilities = facilities.astype(str)
F = facilities.shape[0]
participants = participants.astype(str)
energy_sources = energy_sources.astype(str)

# Determine possible generator types
energy_types = np.unique(energy_sources)
intermittent = np.unique(np.array([wind, solar])) # np.unique sorts them

if save_arrays:
    np.save(gv.facilities_file, facilities)
    np.save(gv.participants_file, participants)
    np.save(gv.capacities_file, capacities)
    np.save(gv.energy_sources_file, energy_sources)
    np.save(gv.exited_file, exited)
    np.save(gv.energy_types_file, energy_types)
    np.save(gv.heat_rates, heat_rates)
    np.save(gv.co2_rates, co2_rates)
    #np.save(gv.transport_charges, transport_charges) - don't save b/c need to convert to 2015 $
    np.save(gv.info_sources, info_sources)
    np.save(gv.intermittent_file, intermittent)

print(f"Facility characteristics processed.", flush=True)

# %%
# Process load curtailment

# Generate price array
# Initialize arrays and counters
load_curtailment = np.ones((T, H)) * np.nan
yr = start_year
ctr = 0
df_curtailment_col = "Estimated Load Curtailment (MW)"

# Go through each year's STEM summary, select correct dates, insert into prices array
while yr <= end_year:
    # Import month-year bids/offers datafile and clean dates
    yr_str = str(yr).zfill(4)
    df_load_summary_yr = (pd.read_csv(gv.data_path + "load_summary/load-summary-" + yr_str + ".csv")
                          .sort_values(by=[df_trading_col, df_interval_col]))
    df_load_summary_yr['date'] = pd.to_datetime(df_load_summary_yr[df_trading_col])
    df_load_summary_yr.drop(df_load_summary_yr.loc[df_load_summary_yr['date'] < np.datetime64(str(start_year) + "-" + str(start_month) + "-01")].index, inplace=True)

    # Create array of dates
    month_begin = start_month if yr == start_year else 1
    nextmonthyr_yr = yr + 1
    nextmonthyr_month = 1
    if yr == end_year:
        nextmonthyr_yr = yr if end_month < 12 else yr + 1
        nextmonthyr_month = end_month + 1 if end_month < 12 else 1
    dates_in_yr = np.arange(np.datetime64(str(yr).zfill(4) + "-" + str(month_begin).zfill(2) + "-" + str(1).zfill(2)), 
                            np.datetime64(str(nextmonthyr_yr).zfill(4) + "-" + str(nextmonthyr_month).zfill(2) + "-" + str(1).zfill(2)))

    # Ensure that the number of observations is the same as dates x intervals
    group = df_load_summary_yr.groupby(['date', df_interval_col]).agg("sum").reset_index() # shouldn't matter, but just in case there are multiple observations
    mux = pd.MultiIndex.from_product([dates_in_yr, range(1, H+1)], names=('date',df_interval_col))
    group = mux.to_frame(index=False).merge(group, on=['date',df_interval_col], how="left") \
                                     .fillna(np.nan) # add the facility codes that are missing, also puts them in correct order

    # Add to prices array
    load_curtailment[ctr:ctr+dates_in_yr.shape[0],:] = np.reshape(np.nan_to_num(group[df_curtailment_col].values), 
                                                                  (dates_in_yr.shape[0],H))

    # Update counters
    ctr += dates_in_yr.shape[0]
    yr += 1
    
if save_arrays:
    np.save(gv.load_curtailment_file, load_curtailment)
        
print(f"Load curtailment processed.", flush=True)

# %%
# Process energy generated

# Create array of electricity generated
    # NOTE: Need to deal with "ALINTA_WGP_AGG" and "MHPS"
# Initialize arrays and counters
energy_gen = np.ones((F, T, H)) * np.nan
yr = start_year
month = start_month
ctr = 0 # to keep track of the date

# Go through each month and create a dataframe of quantity of energy produced by each facility
while yr <= end_year and (month <= end_month or yr < end_year):
    yrmonth_str = str(yr).zfill(4) + "-" + str(month).zfill(2)
    nextmonth_yr = yr if month < 12 else yr + 1
    nextmonth_month = month + 1 if month < 12 else 1
    dates_in_monthyr = np.arange(np.datetime64(str(yr).zfill(4) + "-" + str(month).zfill(2) + "-" + str(1).zfill(2)),
                                 np.datetime64(str(nextmonth_yr).zfill(4) + "-" + str(nextmonth_month).zfill(2) + "-" + str(1).zfill(2)))
    group = df_scada[yrmonth_str].groupby([df_facility_col, df_trading_col, df_interval_col]).agg("sum").reset_index()
    mux = pd.MultiIndex.from_product([facilities, np.unique(group[df_trading_col]), range(1, H+1)], names=(df_facility_col,df_trading_col,df_interval_col))
    group = mux.to_frame(index=False).merge(group, on=[df_facility_col, df_trading_col, df_interval_col], how="left").fillna(np.nan) # add the facility codes that are missing, also puts them in correct order

    # Add to energy generated array
    energy_gen[:,ctr:ctr+dates_in_monthyr.shape[0],:] = group[df_max_gen_col].values.reshape((F,dates_in_monthyr.shape[0],H))
    ctr += dates_in_monthyr.shape[0]

    # Update month + year
    yr = yr if month < 12 else yr + 1
    month = month + 1 if month < 12 else 1

if save_arrays:
    np.savez_compressed(gv.energy_gen_file, energy_gen) # compress it b/c very large array
        
print(f"Energy generated processed.", flush=True)


# %%
# Process prices for each interval

# Create inflation adjusters
# Process inflation raw data and make it yearly
df_inflation = pd.read_excel(gv.data_loc + "raw_data/abs_data/640101_2022.xlsx", sheet_name="Data1", usecols="A,J", skiprows=lambda x: x < 9)
df_inflation.rename(columns={df_inflation.columns[0]: "date", df_inflation.columns[1]: "index"}, inplace=True)
df_inflation['year'] = pd.DatetimeIndex(df_inflation['date']).year
df_inflation['month'] = pd.DatetimeIndex(df_inflation['date']).month
df_inflation.drop(df_inflation.index[df_inflation['month'] != 6], inplace=True)
df_inflation.drop(['date','month'], axis=1, inplace=True)

# Use price indices to create conversion factor for reference year
index_refyear = df_inflation['index'][df_inflation['year'] == gv.refyear].values[0]
df_inflation['convert_factor'] = index_refyear / df_inflation['index']

# Extend year variables to every day of year
min_year = np.min(df_inflation['year'])
max_year = np.max(df_inflation['year'])
df_inflation_extend = pd.DataFrame({'date': pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")})
df_inflation_extend['year'] = pd.DatetimeIndex(df_inflation_extend['date']).year
df_inflation_extend = df_inflation_extend.merge(df_inflation, how="left", on=['year'])

# Create date array and conversion factor array
dates_inflation = df_inflation_extend['date'].values
convert_factor_inflation = df_inflation_extend['convert_factor'].values

# Generate price array
# Initialize arrays and counters
prices = np.ones((T, H)) * np.nan
yr = start_year
ctr = 0
df_price_col = "Clearing Price ($/MWh)"

# Go through each year's STEM summary, select correct dates, insert into prices array
while yr <= end_year:
    # Import month-year bids/offers datafile and clean dates
    yr_str = str(yr).zfill(4)
    df_stem_yr = (pd.read_csv(gv.data_path + "stem_summary/stem-summary-" + yr_str + ".csv")
                    .sort_values(by=[df_trading_col, df_interval_col]))
    df_stem_yr['date'] = pd.to_datetime(df_stem_yr[df_trading_col])
    df_stem_yr.drop(df_stem_yr.loc[df_stem_yr['date'] < np.datetime64(str(start_year) + "-" + str(start_month) + "-01")].index, inplace=True)

    # Create array of dates
    month_begin = start_month if yr == start_year else 1
    nextmonthyr_yr = yr + 1
    nextmonthyr_month = 1
    if yr == end_year:
        nextmonthyr_yr = yr if end_month < 12 else yr + 1
        nextmonthyr_month = end_month + 1 if end_month < 12 else 1
    dates_in_yr = np.arange(np.datetime64(str(yr).zfill(4) + "-" + str(month_begin).zfill(2) + "-" + str(1).zfill(2)), 
                            np.datetime64(str(nextmonthyr_yr).zfill(4) + "-" + str(nextmonthyr_month).zfill(2) + "-" + str(1).zfill(2)))

    # Ensure that the number of observations is the same as dates x intervals
    group = df_stem_yr[['date', df_interval_col, df_price_col]].groupby(['date', df_interval_col]).agg("mean").reset_index() # shouldn't matter, but just in case there are multiple observations
    mux = pd.MultiIndex.from_product([dates_in_yr, range(1, H+1)], names=('date',df_interval_col))
    group = mux.to_frame(index=False).merge(group, on=['date',df_interval_col], how="left") \
                                     .fillna(np.nan) # add the facility codes that are missing, also puts them in correct order

    # Add to prices array
    prices[ctr:ctr+dates_in_yr.shape[0],:] = np.reshape(group[df_price_col].values, 
                                                        (dates_in_yr.shape[0],H))

    # Update counters
    ctr += dates_in_yr.shape[0]
    yr += 1

# Adjust prices for inflation
prices = prices * convert_factor_inflation[np.isin(dates_inflation, dates),np.newaxis]

if save_arrays:
    np.save(gv.prices_file, prices)
        
print(f"Prices processed.", flush=True)

# %%
# Process DSPs

# Import raw data
df_cap_credits = pd.read_excel(gv.data_loc + "raw_data/wem_data/capacity_credit_assignments/Capacity Credits since market start up to 2023-2024.xlsx", skiprows=lambda x: x < 11)
df_cap_credits = df_cap_credits.iloc[:,2:] # drop columns A & B
df_cap_credits = df_cap_credits.loc[:,~df_cap_credits.columns.str.startswith("Unnamed")] # drop columns shouldn't have picked up
df_cap_credits = df_cap_credits.dropna(how='all') # drop row if all NaN
df_cap_credits = df_cap_credits[df_cap_credits['Facility name'].notna()]

# Force all values to be floats
df_cap_credits_columns = df_cap_credits.columns
df_cap_credits_dates = df_cap_credits_columns[df_cap_credits_columns != "Facility name"] # just the year columns
for col in df_cap_credits_dates:
    df_cap_credits[col] = pd.to_numeric(df_cap_credits[col], errors="coerce")
    
# Identify just DSPs
df_cap_credits_dsp = df_cap_credits[df_cap_credits['Facility name'].str.contains("_DSP_")]
df_cap_credits_dsp = df_cap_credits_dsp.fillna(0.0)

# Create DSP values
dates_year_capacity = pd.to_datetime(dates).year
dates_year_capacity = dates_year_capacity - 1 * (pd.to_datetime(dates).month <= 9)
start_capacity_years, start_capacity_years_counts = np.unique(dates_year_capacity, return_counts=True)
start_capacity_years_col = np.array([f"{yr}-{str(yr+1)[2:]}" for yr in start_capacity_years])
df_cap_credits_dsp = df_cap_credits_dsp[df_cap_credits_dsp.columns[np.isin(df_cap_credits_dsp.columns, np.concatenate((np.array(["Facility name"]), start_capacity_years_col)))]]
dsp_names = np.array([f"{name}" for name in df_cap_credits_dsp['Facility name'].values])
dsp_quantities = np.repeat(df_cap_credits_dsp[df_cap_credits_dsp.columns[1:]].values, start_capacity_years_counts, axis=1)

# How much DSP dispatched
dsp_price_start_year = 2012
dsp_price_end_year = 2022
df_dsp_prices = pd.concat([pd.read_csv(f"{gv.data_path}demand_side_programme_prices/dsp-decrease-price-{yr}.csv") for yr in np.arange(dsp_price_start_year, dsp_price_end_year + 1)], ignore_index=True)
df_dsp_prices['Trading Date'] = pd.to_datetime(df_dsp_prices['Trading Date'])
mux = pd.MultiIndex.from_product([dates, dsp_names], names=('Trading Date', 'Facility Code'))
df_dsp_prices = mux.to_frame(index=False).merge(df_dsp_prices, on=['Trading Date', 'Facility Code'], how="left").fillna(np.nan)
dsp_quantity_dispacthed = df_dsp_prices[df_dsp_prices.columns[5:-3]].fillna(0.0).values * np.reshape(dsp_quantities.T, (-1,))[:,np.newaxis]
dsp_quantity_dispacthed = np.reshape(dsp_quantity_dispacthed, (dsp_quantities.shape[1], dsp_quantities.shape[0], dsp_quantity_dispacthed.shape[1]))
dsp_quantity_dispacthed = np.sum(dsp_quantity_dispacthed, axis=1)

# Only interested in aggregate quantity
dsp_quantities = np.sum(dsp_quantities, axis=0) # sum them all up, we're only interested in the aggregate

# Save arrays
if save_arrays:
    np.save(gv.dsp_quantities_file, dsp_quantities)
    np.save(gv.dsp_dispatch_file, dsp_quantity_dispacthed)
print(f"Demand-side programme data processed.", flush=True)

# %%
# Process residential tariffs

df_tariffs = pd.read_csv(gv.western_power_data_path + "electricity_tariffs/cleaned_tariffs/residential_tariffs.csv")
residential_tariffs = np.ones(dates.shape) * np.nan
year_from_arr = df_tariffs['year_from'].values
year_to_arr = df_tariffs['year_to'].values
tariff_arr = df_tariffs['tariff (AUD/mwh)'].values
for i in range(tariff_arr.shape[0]):
    dates_arr_i = np.arange(np.datetime64(str(year_from_arr[i]).zfill(4) + "-07-01"), np.datetime64(str(year_to_arr[i]).zfill(4) + "-06-30") + 1)
    select_dates = np.isin(dates, dates_arr_i)
    residential_tariffs[select_dates] = tariff_arr[i]
    
# Adjust for inflation
residential_tariffs = residential_tariffs * convert_factor_inflation[np.isin(dates_inflation, dates)]

if save_arrays:
    np.save(gv.residential_tariff_file, residential_tariffs)
    
print(f"Residential tariffs processed.", flush=True)

# %%
# Process real-time prices for each interval

# Generate price array
# Initialize arrays and counters
# Generate price array
# Initialize arrays and counters
prices_realtime = np.ones((T, H)) * np.nan
load_realtime = np.ones((T, H)) * np.nan
yr = start_year
month = start_month
ctr = 0
df_price_col = "Final Price ($/MWh)"
df_load_col = "Total Generation (MW)"

# Go through each year's STEM summary, select correct dates, insert into prices array
while yr <= end_year and (month <= end_month or yr < end_year):
    # Create array of dates
    yrmonth_str = str(yr).zfill(4) + "-" + str(month).zfill(2)
    nextmonth_yr = yr if month < 12 else yr + 1
    nextmonth_month = month + 1 if month < 12 else 1
    dates_in_monthyr = np.arange(np.datetime64(str(yr).zfill(4) + "-" + str(month).zfill(2) + "-" + str(1).zfill(2)),
                                 np.datetime64(str(nextmonth_yr).zfill(4) + "-" + str(nextmonth_month).zfill(2) + "-" + str(1).zfill(2)))

    if (yr > 2012) or ((yr == 2012) and (month >= 7)): # first year of balancing market
        # Import yearly balancing market datafile and clean dates
        yr_str = str(yr).zfill(4)
        df_balancing_yr = (pd.read_csv(gv.data_path + "balancing_market_summary/balancing-summary-" + yr_str + ".csv")
                           .sort_values(by=[df_trading_col, df_interval_col]))
        df_balancing_yr['date'] = pd.to_datetime(df_balancing_yr[df_trading_col])
        df_balancing_yr.drop(df_balancing_yr.loc[df_balancing_yr['date'] < np.datetime64(str(start_year) + "-" + str(start_month) + "-01")].index, inplace=True)

        # Ensure that the number of observations is the same as dates x intervals
        group = df_balancing_yr[['date', df_interval_col, df_price_col, df_load_col]].groupby(['date', df_interval_col]).agg("mean").reset_index() # shouldn't matter, but just in case there are multiple observations
        mux = pd.MultiIndex.from_product([dates_in_monthyr, range(1, H+1)], names=('date',df_interval_col))
        group = mux.to_frame(index=False).merge(group, on=['date',df_interval_col], how="left") \
                                         .fillna(np.nan) # add the facility codes that are missing, also puts them in correct order

        # Add to prices array
        prices_realtime[ctr:ctr+dates_in_monthyr.shape[0],:] = np.reshape(group[df_price_col].values, (dates_in_monthyr.shape[0],H))
        
        # Add to load array
        load_realtime[ctr:ctr+dates_in_monthyr.shape[0],:] = np.reshape(group[df_load_col].values, (dates_in_monthyr.shape[0],H))
        
    else:
        df_balancing_yr = (pd.read_csv(gv.data_path + "historical_balancing_market_data/pre-balancing-market-data.csv")
                           .sort_values(by=['Trade Date', 'Delivery Hour', 'Delivery Interval']))
        df_balancing_yr['date'] = pd.to_datetime(df_balancing_yr['Trade Date'])
        df_balancing_yr['Delivery Hour'] = df_balancing_yr['Delivery Hour'] * 2 + (df_balancing_yr['Delivery Interval'] - 1) + 1 # make same indexing as in other datasets
        df_balancing_yr.drop(df_balancing_yr.loc[df_balancing_yr['date'] < np.datetime64(str(start_year) + "-" + str(start_month) + "-01")].index, inplace=True)
        
        # Ensure that the number of observations is the same as dates x intervals
        group = df_balancing_yr[['date','Delivery Hour','MCAP Price Per MWh']].groupby(['date', 'Delivery Hour']).agg("mean").reset_index() # shouldn't matter, but just in case there are multiple observations
        mux = pd.MultiIndex.from_product([dates_in_monthyr, range(1, H+1)], names=('date','Delivery Hour'))
        group = mux.to_frame(index=False).merge(group, on=['date','Delivery Hour'], how="left") \
                                         .fillna(np.nan) # add the facility codes that are missing, also puts them in correct order

        # Add to prices array
        prices_realtime[ctr:ctr+dates_in_monthyr.shape[0],:] = np.reshape(group['MCAP Price Per MWh'].values, (dates_in_monthyr.shape[0],H))

    # Update counters
    ctr += dates_in_monthyr.shape[0]
    yr = yr if month < 12 else yr + 1
    month = month + 1 if month < 12 else 1

# Adjust prices for inflation
prices_realtime = prices_realtime * convert_factor_inflation[np.isin(dates_inflation, dates),np.newaxis]

if save_arrays:
    np.save(gv.prices_realtime_file, prices_realtime)
    np.save(gv.load_realtime_file, load_realtime)
        
print(f"Real-time prices and loads processed.", flush=True)

# %%
# Adjust transport charges for inflation
transport_charges = transport_charges * np.unique(convert_factor_inflation[pd.DatetimeIndex(dates_inflation).year == 2014])[0]
np.save(gv.transport_charges, transport_charges)
print(f"Transport charges adjusted for inflation.", flush=True)

# %%
# Create carbon taxes array
carbon_tax_1 = 23.0 / 1000.0 # convert to $ / kg CO2
carbon_tax_2 = 24.15 / 1000.0 # convert to $ / kg CO2
carbon_taxes = ((dates >= np.datetime64("2012-07-01")) & (dates < np.datetime64("2013-07-01"))) * carbon_tax_1 + ((dates >= np.datetime64("2013-07-01")) & (dates < np.datetime64("2014-07-01"))) * carbon_tax_2
carbon_taxes = carbon_taxes * convert_factor_inflation[np.isin(dates_inflation, dates)]
np.save(gv.carbon_taxes_file, carbon_taxes)
print(f"Carbon taxes processed.", flush=True)

# %%
# Process whether generator available in balancing market

# Initialize arrays and counters
balancing_avail = np.ones((F, T, H)) * np.nan
yr = start_year
month = start_month
ctr = 0 # to keep track of the date

# Go through each month and create a dataframe of quantity of energy produced by each facility
while yr <= end_year and (month <= end_month or yr < end_year):
    yrmonth_str = str(yr).zfill(4) + "-" + str(month).zfill(2)
    nextmonth_yr = yr if month < 12 else yr + 1
    nextmonth_month = month + 1 if month < 12 else 1
    dates_in_monthyr = np.arange(np.datetime64(str(yr).zfill(4) + "-" + str(month).zfill(2) + "-" + str(1).zfill(2)),
                                 np.datetime64(str(nextmonth_yr).zfill(4) + "-" + str(nextmonth_month).zfill(2) + "-" + str(1).zfill(2)))
    
    if (yr > 2012) or ((yr == 2012) and (month >= 7)): # once balancing market began
        df_balancing_myr = (pd.read_csv(gv.data_path + "effective_balancing_submission/effective-balancing-submission-" + yrmonth_str + ".csv")
                            .sort_values(by=[df_trading_col, df_interval_col, df_facility_col]))
        df_balancing_myr['available'] = 1.0 * (df_balancing_myr['Type'] != "Unavailable")
        group = df_balancing_myr[[df_facility_col, df_trading_col, df_interval_col, 'available']].groupby([df_facility_col, df_trading_col, df_interval_col]).agg("min").reset_index()
        mux = pd.MultiIndex.from_product([facilities, np.unique(group[df_trading_col]), range(1, H+1)], names=(df_facility_col,df_trading_col,df_interval_col))
        group = mux.to_frame(index=False).merge(group, on=[df_facility_col, df_trading_col, df_interval_col], how="left").fillna(np.nan) # add the facility codes that are missing, also puts them in correct order

        # Add to balancing availability array
        balancing_avail[:,ctr:ctr+dates_in_monthyr.shape[0],:] = group['available'].values.reshape((F,dates_in_monthyr.shape[0],H))

    # Update month + year
    ctr += dates_in_monthyr.shape[0]
    yr = yr if month < 12 else yr + 1
    month = month + 1 if month < 12 else 1

if save_arrays:
    np.save(gv.balancing_avail_file, balancing_avail) # compress it b/c very large array
        
print(f"Availability processed.", flush=True)


# %%
# Process entry and exit dates

# Create array of exited dates
dates_idx = np.arange(T)
produced_morethan0_in_day = np.sum(np.nan_to_num(energy_gen), axis=2) > 0.0
last_day_produced_morethan0 = np.max(produced_morethan0_in_day * dates_idx[np.newaxis,:], axis=1)
#exited_date = np.array(["NaT" for i in range(facilities.shape[0])], dtype="datetime64[D]")
exited_date = np.repeat(dates[-1], F)
exited_date[exited] = dates[last_day_produced_morethan0[exited]] # last date operating

# Create array of entered dates
in_data = np.any(~np.isnan(energy_gen), axis=2)
first_day_in_data = np.argmin(1.0 - in_data, axis=1) # will return the index of first time generator produced
entered_date = dates[first_day_in_data] # doing it for all participants since we have a problem with the Registered From dates

if save_arrays:
    np.save(gv.exited_date_file, exited_date)
    np.save(gv.entered_date_file, entered_date)
        
print(f"Entry / exit dates processed.", flush=True)


# %%
# Process capacity prices

# Import raw data
df_year_from_col = "Year From"
df_year_until_col = "Year Until"
df_benchmark_price_col = "BRCP ($ per MW per year)"
df_cap_price_col = "Reserve Capacity Price ($ per MW per year)"
df_cap_price = (pd.read_csv(gv.data_path + "reserve_capacity/benchmark_reserve_capacity_price.csv")
                  .sort_values(by=[df_year_from_col, df_year_until_col]))

# Convert prices into floats
for col in [df_benchmark_price_col, df_cap_price_col]:
    df_cap_price[col] = df_cap_price[col].astype("str")
    for i in ["(", ")", "$", ","]:
        df_cap_price[col] = df_cap_price[col].str.replace(i, "")
    df_cap_price[col] = pd.to_numeric(df_cap_price[col], errors="coerce")
    df_cap_price.drop(df_cap_price.index[pd.isnull(df_cap_price[col])], inplace=True) # get rid of "NaN"s

# Sort
df_cap_price = df_cap_price.sort_values(by=[df_year_from_col, df_year_until_col])

# Convert years to dates
for col in [df_year_from_col, df_year_until_col]:
    df_cap_price[col] = df_cap_price[col].astype("str")
df_cap_price[df_year_from_col] = df_cap_price[df_year_from_col].map(lambda x: "10/01/" + x)
df_cap_price[df_year_until_col] = df_cap_price[df_year_until_col].map(lambda x: "09/30/" + x)
for col in [df_year_from_col, df_year_until_col]:
    df_cap_price[col] = pd.to_datetime(df_cap_price[col])

# Create arrays
capacity_price = df_cap_price[df_cap_price_col].values
cap_date_from = df_cap_price[df_year_from_col].values
cap_date_until = df_cap_price[df_year_until_col].values

# Add post-2020-2021 data
additional_years = np.arange(2021, 2024)
for yr in additional_years:
    df_cap_price = pd.read_csv(gv.data_path + f"reserve_capacity_prices/reserve-capacity-prices-{yr}.csv")[['Price Type', 'Price ($)']]
    cap_price_add = df_cap_price[df_cap_price['Price Type'] == "Reserve Capacity"]['Price ($)'].values[0]
    capacity_price = np.concatenate((capacity_price, np.array([cap_price_add])))
    cap_date_from = np.concatenate((cap_date_from, np.array([np.datetime64(f"{yr}-10-01", "D")])))
    cap_date_until = np.concatenate((cap_date_until, np.array([np.datetime64(f"{yr + 1}-09-30", "D")])))

# Adjust for inflation
convert_factor_inflation_capacity = convert_factor_inflation[np.isin(dates_inflation, cap_date_until)] # use cap_date_until instead of cap_date_fromm b/c most of the capacity year is in the year of cap_date_until
convert_factor_inflation_capacity = np.concatenate((convert_factor_inflation_capacity, np.repeat(np.array([convert_factor_inflation[-1]]), np.sum(cap_date_until > dates_inflation[-1])))) # for years past the last year of inflation data, just use the last conversion factor in the data
capacity_price = capacity_price * convert_factor_inflation_capacity

# Remove those that are too early
too_early = cap_date_from < np.datetime64("2006-10-01")
capacity_price, cap_date_from, cap_date_until = capacity_price[~too_early], cap_date_from[~too_early], cap_date_until[~too_early]

if save_arrays:
    np.save(gv.capacity_price_file, capacity_price)
    np.save(gv.cap_date_from_file, cap_date_from)
    np.save(gv.cap_date_until_file, cap_date_until)
        
print(f"Capacity payments processed.", flush=True)


# %%
# Process capacity credit assignments

df_facilities = pd.DataFrame({'Facility name': facilities})
df_cap_credits = df_cap_credits.merge(df_facilities, how="right", on="Facility name") # use all of the facilities
df_cap_credits = df_cap_credits.groupby("Facility name").agg("sum").reset_index() # if facility listed twice, just sum, also turns NaN into 0

# Select just the dates we need
df_cap_credits_dates = pd.to_datetime(df_cap_credits_dates, errors='coerce').year
df_cap_credits_use = np.isin(df_cap_credits_dates, pd.to_datetime(cap_date_from).year) # which years to include
capacity_commitments = df_cap_credits.values[:,1:][:,df_cap_credits_use]

# Update array type
capacity_commitments = capacity_commitments.astype(float)

if save_arrays:
    np.save(gv.capacity_commitments_file, capacity_commitments)
        
print(f"Capacity credit assignments processed.", flush=True)


# %%
# Process price limits

# Import raw data
df_price_from_col = "From Date"
df_price_until_col = "Until Date"
df_price_lim_col = "Value"
df_price_type_col = "Name"
df_price_lims = pd.read_csv(gv.data_path + "price_limits/price-limits.csv")

# Convert prices into floats
df_price_lims[df_price_lim_col] = df_price_lims[df_price_lim_col].astype("str")
for i in [")", "$", ","]:
    df_price_lims[df_price_lim_col] = df_price_lims[df_price_lim_col].str.replace(i, "")
df_price_lims[df_price_lim_col] = df_price_lims[df_price_lim_col].str.replace("(", "-") # make number negative
df_price_lims[df_price_lim_col] = pd.to_numeric(df_price_lims[df_price_lim_col], errors="coerce")

# The month of 10/1/11 - 11/1/11 (American dates) is missing, we're going to assume the values before it carry over for one more month until new values begin
df_price_lims.loc[df_price_lims['Until Date'] == "1/10/11", 'Until Date'] = "1/11/11"

# Convert from and until dates to dates
max_date = pd.to_datetime(np.max(cap_date_until))
max_date_str = f"{max_date.day}/{max_date.month}/{str(max_date.year)[-2:]}"
df_price_lims[df_price_from_col] = df_price_lims[df_price_from_col].str.replace("TBA", max_date_str)
df_price_lims[df_price_until_col] = df_price_lims[df_price_until_col].str.replace("TBA", max_date_str)
df_price_lims[df_price_from_col] = pd.to_datetime(df_price_lims[df_price_from_col], dayfirst=True)
df_price_lims[df_price_until_col] = pd.to_datetime(df_price_lims[df_price_until_col], dayfirst=True)

# Determine max/min prices
df_price_lims_maxstem = df_price_lims[df_price_lims[df_price_type_col] == "Maximum STEM Price"]
df_price_lims_minstem = df_price_lims[df_price_lims[df_price_type_col] == "Minimum STEM Price"]
max_prices = np.ones(dates.shape)
max_prices[:] = np.nan
min_prices = np.ones(dates.shape)
min_prices[:] = np.nan
for date_idx, date in enumerate(dates):
    price_lims_temp = df_price_lims_maxstem[(df_price_lims_maxstem[df_price_from_col] <= date) & (df_price_lims_maxstem[df_price_until_col] > date)][df_price_lim_col].values
    if price_lims_temp.shape[0] > 0:
        max_prices[date_idx] = price_lims_temp[0] # shouldn't ever be more than one entry in array
    price_lims_temp = df_price_lims_minstem[(df_price_lims_minstem[df_price_from_col] <= date) & (df_price_lims_minstem[df_price_until_col] > date)][df_price_lim_col].values
    if price_lims_temp.shape[0] > 0:
        min_prices[date_idx] = price_lims_temp[0] # shouldn't ever be more than one entry in array

# Adjust for inflation
max_prices_notinflationadjusted = np.copy(max_prices) # want to save this b/c need it later
convert_factor_inflation_maxprice = convert_factor_inflation[np.isin(dates_inflation, dates)]
convert_factor_inflation_maxprice = np.concatenate((convert_factor_inflation_maxprice, np.repeat(np.array([convert_factor_inflation[-1]]), np.sum(dates > dates_inflation[-1])))) # for years past the last year of inflation data, just use the last conversion factor in the data
max_prices = max_prices * convert_factor_inflation_maxprice
min_prices_notinflationadjusted = np.copy(min_prices) # want to save this b/c need it later
convert_factor_inflation_minprice = convert_factor_inflation[np.isin(dates_inflation, dates)]
convert_factor_inflation_minprice = np.concatenate((convert_factor_inflation_minprice, np.repeat(np.array([convert_factor_inflation[-1]]), np.sum(dates > dates_inflation[-1])))) # for years past the last year of inflation data, just use the last conversion factor in the data
min_prices = min_prices * convert_factor_inflation_minprice

if save_arrays:
    np.save(gv.max_prices_file, max_prices)
    np.save(gv.min_prices_file, min_prices)
        
print(f"Maximum/minimum STEM prices processed.", flush=True)


# %%
# Process outage information

# Initialize arrays and counters
outages_exante = np.ones((F, T, H)) * np.nan
outages_expost = np.ones((F, T, H)) * np.nan
outages_type = np.tile(np.array(["                                           "])[:,np.newaxis,np.newaxis], (F, T, H)) # need it to be long enough to fit anything that might occur
yr = start_year
ctr = 0
df_outage_exante_col = "Ex-Ante Outage (MW)"
df_outage_expost_col = "Ex-Post Outage (MW)"
df_outage_type_col = "Outage Type"

# Go through each year's outage summary, insert it into outage arrays
while yr <= end_year:
    
    # Import month-year bids/offers datafile and clean dates
    yr_str = str(yr).zfill(4)
    df_outage_yr = (pd.read_csv(gv.data_path + "outages/outages-" + yr_str + ".csv")
                      .sort_values(by=[df_facility_col, df_trading_col, df_interval_col]))
    df_outage_yr['date'] = pd.to_datetime(df_outage_yr[df_trading_col])
    df_outage_yr.drop(df_outage_yr.loc[df_outage_yr['date'] < np.datetime64(str(start_year) + "-" + str(start_month) + "-01")].index, inplace=True)

    # Create array of dates
    month_begin = start_month if yr == start_year else 1
    nextmonthyr_yr = yr + 1
    nextmonthyr_month = 1
    if yr == end_year:
        nextmonthyr_yr = yr if end_month < 12 else yr + 1
        nextmonthyr_month = end_month + 1 if end_month < 12 else 1
    dates_in_yr = np.arange(np.datetime64(str(yr).zfill(4) + "-" + str(month_begin).zfill(2) + "-" + str(1).zfill(2)), 
                            np.datetime64(str(nextmonthyr_yr).zfill(4) + "-" + str(nextmonthyr_month).zfill(2) + "-" + str(1).zfill(2)))

    # Ensure that the number of observations is the same as dates x intervals
    group = df_outage_yr.groupby([df_facility_col, 'date', df_interval_col]).agg("last").reset_index() # shouldn't matter, but just in case there are multiple observations
    mux = pd.MultiIndex.from_product([facilities, dates_in_yr, range(1, H+1)], names=(df_facility_col,'date',df_interval_col))
    group = mux.to_frame(index=False).merge(group, on=[df_facility_col,'date',df_interval_col], how="left") # add the facility codes that are missing, also puts them in correct order

    # Reshape arrays
    outages_exante_yr = np.reshape(group[df_outage_exante_col].fillna(0.).values, (F,dates_in_yr.shape[0],H))
    outages_expost_yr = np.reshape(group[df_outage_expost_col].fillna(0.).values, (F,dates_in_yr.shape[0],H))
    outages_type_yr = np.reshape(group[df_outage_type_col].fillna("").values, (F,dates_in_yr.shape[0],H))

    # Create NaN for facilities that were no in the data yet
    facility_in_market = np.repeat(((dates_in_yr[np.newaxis,:] >= entered_date[:,np.newaxis]) & (dates_in_yr[np.newaxis,:] <= exited_date[:,np.newaxis]))[:,:,np.newaxis], H, axis=2)
    outages_exante_yr[~facility_in_market] = np.nan
    outages_expost_yr[~facility_in_market] = np.nan

    # Add to outages array
    outages_exante[:,ctr:ctr+dates_in_yr.shape[0],:] = outages_exante_yr
    outages_expost[:,ctr:ctr+dates_in_yr.shape[0],:] = outages_expost_yr
    outages_type[:,ctr:ctr+dates_in_yr.shape[0],:] = outages_type_yr

    # Update counters
    ctr += dates_in_yr.shape[0]
    yr += 1

if save_arrays:
    np.savez_compressed(gv.outages_file, outages_exante, outages_expost, outages_type) # compress it b/c very large arrays

outage_types = np.unique(outages_type[outages_type != ""])

print(f"Outages processed.", flush=True)


# %%
# Process capacity costs

# Import EIA and AETA data
df_eia = pd.read_csv(gv.data_loc + "raw_data/capacity_cost_data/eia_capacity_costs.csv")
df_aeta = pd.read_csv(gv.data_loc + "raw_data/capacity_cost_data/aeta_capacity_costs.csv")

# Replace source names with the source names consistently using
energy_source_replace_dict = {
    "coal": gv.coal, 
    "gas": gv.gas_ccgt, 
    "wind": gv.wind, 
    "solar": gv.solar, 
    "gas ocgt": gv.gas_ocgt
}
for key in energy_source_replace_dict:
    df_eia.loc[df_eia['source'] == key, 'source'] = energy_source_replace_dict[key]
    df_aeta.loc[df_aeta['source'] == key, 'source'] = energy_source_replace_dict[key]

# Expand energy sources to all
mux = pd.MultiIndex.from_product([np.unique(df_eia['year']), gv.use_sources], names=('year', 'source'))
df_eia = mux.to_frame(index=False).merge(df_eia, on=['year', 'source'], how="left").fillna(np.nan) # add the sources that are missing, also puts them in correct order
mux = pd.MultiIndex.from_product([np.unique(df_aeta['year']), gv.use_sources], names=('year', 'source'))
df_aeta = mux.to_frame(index=False).merge(df_aeta, on=['year', 'source'], how="left").fillna(np.nan) # add the sources that are missing, also puts them in correct order

# Construct arrays from data
capital_costs_us = np.reshape(df_eia['capital_cost_per_kw'].values, (-1,gv.use_sources.shape[0])) # years x energy sources
capital_costs_wa = np.reshape(df_aeta['capital_cost_per_kw'].values, (-1,gv.use_sources.shape[0]))
years = np.arange(start_year, end_year + 1)
years = years[np.isin(years, pd.DatetimeIndex(cap_date_from).year)]

# Expand to every year
capital_costs_full_us = np.ones((years.shape[0], capital_costs_us.shape[1])) * np.nan
capital_costs_full_wa = np.ones((years.shape[0], capital_costs_us.shape[1])) * np.nan
ctr_us = 0
ctr_wa = 0
for y, year in enumerate(years):
    if year in np.reshape(df_eia['year'].values, (-1,gv.use_sources.shape[0]))[:,0]:
        capital_costs_full_us[y,:] = capital_costs_us[ctr_us,:]
        ctr_us += 1
    if year in np.reshape(df_aeta['year'].values, (-1,gv.use_sources.shape[0]))[:,0]:
        capital_costs_full_wa[y,:] = capital_costs_wa[ctr_wa,:]
        ctr_wa += 1

# Interpolate EIA costs
for s, source in enumerate(np.reshape(df_eia['source'].values, (-1,gv.use_sources.shape[0]))[0,:]):
    interpolate_costs = interp1d(np.reshape(df_eia['year'].values, (-1,gv.use_sources.shape[0]))[:,0], capital_costs_us[:,s], bounds_error=False, fill_value=0.0) # SciPy interpolation to get linear extrapolation
    capital_costs_full_us[:,s] = interpolate_costs(years)
    capital_costs_full_us[years < np.reshape(df_eia['year'].values, (-1,gv.use_sources.shape[0]))[0,0],s] = capital_costs_us[0,s]
    capital_costs_full_us[years > np.reshape(df_eia['year'].values, (-1,gv.use_sources.shape[0]))[-1,0],s] = capital_costs_us[-1,s]

# Convert to Western Australia
factors = capital_costs_full_wa / capital_costs_full_us
for s, source in enumerate(np.reshape(df_aeta['source'].values, (-1,gv.use_sources.shape[0]))[0,:]):
    if not np.all(np.isnan(factors[:,s])):
        factors[:,s] = np.interp(years, np.reshape(df_aeta['year'].values, (-1,gv.use_sources.shape[0]))[:,0], factors[~np.isnan(factors[:,s]),s]) # numpy interpolation to work with only one factor
capital_costs_full_wa = factors * capital_costs_full_us

# Adjust prices for inflation
dates_inflation_yr = pd.DatetimeIndex(dates_inflation).year
dates_inflation_yr, dates_inflation_yr_idx = np.unique(dates_inflation_yr, return_index=True)
convert_factor_inflation_yr = convert_factor_inflation[dates_inflation_yr_idx]
capacity_costs = capital_costs_full_wa * convert_factor_inflation_yr[np.isin(dates_inflation_yr, years),np.newaxis]

# Record sources
capacity_costs_sources = np.reshape(df_eia['source'].values, (-1,gv.use_sources.shape[0]))[0,:]

# Import NREL data
nrel_data_sheetname = {
    gv.solar: "Solar - Utility PV", 
    gv.wind: "Land-Based Wind", 
    gv.gas_ocgt: "Natural Gas_FE", 
    gv.gas_ccgt: "Natural Gas_FE", 
    gv.gas_cogen: "Natural Gas_FE", 
    gv.coal: "Coal_FE"
}
nrel_data_cols = {
    gv.solar: "M:AP", 
    gv.wind: "M:AP", 
    gv.gas_ocgt: "M:AP", 
    gv.gas_ccgt: "M:AP", 
    gv.gas_cogen: "M:AP", 
    gv.coal: "M:AP"
}
nrel_data_rows = {
    gv.solar: [149, 163], 
    gv.wind: [138, 149], 
    gv.gas_ocgt: [107, 115], 
    gv.gas_ccgt: [107, 115], 
    gv.gas_cogen: [107, 115], 
    gv.coal: [97, 99]
}
df_nrel = {}
for key in nrel_data_sheetname.keys():
    df_nrel[key] = pd.read_excel(gv.data_loc + "raw_data/capacity_cost_data/nrel_annual_technology_baseline/2023-ATB-Data_Master_v9.0.xlsx", sheet_name=nrel_data_sheetname[key], usecols=nrel_data_cols[key], skiprows=lambda x: x not in nrel_data_rows[key])

# Add on NREL projections
nrel_years = np.array(df_nrel[gv.solar].columns)
nrel_years = nrel_years[nrel_years >= np.max(years)] # only want the last year in the data and the years after
nrel_vals = np.concatenate(tuple([df_nrel[source][nrel_years].values[0,:][:,np.newaxis] for source in capacity_costs_sources]), axis=1)
nrel_factors = nrel_vals[1:,:] / nrel_vals[0,:][np.newaxis,:]
capacity_costs = np.concatenate((capacity_costs, nrel_factors * capacity_costs[-1,:][np.newaxis,:]), axis=0)
years = np.concatenate((years, nrel_years[1:]))

if save_arrays:
    np.save(gv.capacity_costs_file, capacity_costs)
    np.save(gv.capacity_costs_sources_file, capacity_costs_sources)
    np.save(gv.capacity_costs_years_file, years)
        
print(f"Capacity costs processed.", flush=True)


# # %%
# # Process emissions rates

# df_aeta = pd.read_csv(gv.data_loc + "raw_data/capacity_cost_data/aeta_capacity_costs.csv")
# for key in energy_source_replace_dict:
#     df_aeta.loc[df_aeta['source'].str.contains(key), 'source'] = energy_source_replace_dict[key]
    
# # Expand energy sources to all
# mux = pd.MultiIndex.from_product([np.unique(df_aeta['year']), gv.use_sources], names=('year', 'source'))
# df_aeta = mux.to_frame(index=False).merge(df_aeta, on=['year', 'source'], how="left").fillna(np.nan) # add the sources that are missing, also puts them in correct order

# emissions = np.reshape(df_aeta['emissions_rate_per_mwh'].values, (-1,gv.use_sources.shape[0]))[0,:] # in alphabetical order

# if save_arrays:
#     np.save(gv.emissions_file, emissions)
        
# print(f"Emissions rates processed.", flush=True)


# %%
# Process coal and gas prices

# Gas prices
df_gas = pd.read_excel(gv.commodity_data_path + "2022_Major_Commodities_Resource_Data_File.xlsx", 
                       sheet_name="Petroleum - Dom. Gas - Prices", 
                       usecols="C:E", 
                       skiprows=range(7))
df_gas.drop(index=20, inplace=True) # this is wrong in the original data, need to just remove
df_gas.reset_index(drop=True, inplace=True)
df_gas['month'] = df_gas['Quarter'] * 3
df_gas.rename(columns={'Year': "year"}, inplace=True)
df_gas['day'] = 1
df_gas['date'] = pd.to_datetime(df_gas[['year','month','day']])
df_gas_annual = pd.read_excel(gv.commodity_data_path + "2022_Major_Commodities_Resource_Data_File.xlsx", 
                              sheet_name="Petroleum - Dom. Gas - Prices", 
                              usecols="I:J", 
                              skiprows=lambda x: x in range(7) or x > 30)
df_gas_annual['month'] = 6
df_gas_annual.rename(columns={'Year.1': "year", 'A$ per GJ.1': "A$ per GJ"}, inplace=True)
df_gas_annual['day'] = 1
df_gas_annual['date'] = pd.to_datetime(df_gas_annual[['year','month','day']])
df_gas = pd.concat([df_gas_annual, df_gas], ignore_index=True)
gas_prices = np.interp(pd.to_datetime(dates).astype("int64"), pd.to_datetime(df_gas['date'], unit="s").astype("int64") // 10**9, df_gas['A$ per GJ'].values) # interpolate (have the relevant range)
gas_prices = gas_prices * convert_factor_inflation[np.isin(dates_inflation, dates)]

# Coal prices
df_coal = pd.read_excel(gv.commodity_data_path + "2022_Major_Commodities_Resource_Data_File.xlsx", 
                        sheet_name="Other Minerals - Q&V", 
                        usecols="A,D:E", 
                        skiprows=range(7), 
                        nrows=96)
df_coal.rename(columns={'Quarter': "date", 'Quantity (Mt)': "quantity (Mt)", 'Value ($ Million).1': "value ($ million)"}, inplace=True)
df_coal['price (A$/t)'] = (df_coal['value ($ million)'] * 1000000.0) / (df_coal['quantity (Mt)'] * 1000000.0)
gj_to_tonne_coal = 27.0 # gj / tonne # https://content.ces.ncsu.edu/conversion-factors-for-bioenergy
df_coal['price (A$/GJ)'] = df_coal['price (A$/t)'] / gj_to_tonne_coal
coal_prices = np.interp(pd.to_datetime(dates).astype("int64"), pd.to_datetime(df_coal['date'], unit="s").astype("int64") // 10**9, df_coal['price (A$/GJ)'].values) # interpolate (have the relevant range)
coal_prices = coal_prices * convert_factor_inflation[np.isin(dates_inflation, dates)]

if save_arrays:
    np.save(gv.gas_prices_file, gas_prices)
    np.save(gv.coal_prices_file, coal_prices)
        
print(f"Commodity prices processed.", flush=True)

# %%
# Process balancing market submissions

# Initialize arrays and counters
num_bids = 0
balancing_facilities = np.copy(facilities)
balancing_bid = np.ones((F, T, H, num_bids)) * np.nan
balancing_q = np.ones((F, T, H, num_bids)) * np.nan
yr = start_year
month = start_month
ctr = 0 # to keep track of the date

# Go through each month and create a dataframe of quantity of energy produced by each facility
price_col = "Submitted Price ($/MWh)"
q_col = "Submitted Quantity (MW)"
bmo_q_col = "BMO Quantity (MW)"
type_col = "Type"
max_val = "MAX"
min_val = "MIN"
unavail_val = "UNAV"
rpqty_val = "RPQTY"
maxcap_val = "MAXCAP_MINUS_RPQTY"
while yr <= end_year and (month <= end_month or yr < end_year):
    yrmonth_str = str(yr).zfill(4) + "-" + str(month).zfill(2)
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
        df_balancing_myr.loc[max_val_rows, price_col] = df_balancing_myr['max_price'][max_val_rows]
        min_val_rows = df_balancing_myr[price_col] == min_val
        df_balancing_myr.loc[min_val_rows, price_col] = df_balancing_myr['min_price'][min_val_rows]
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
            balancing_bid = np.concatenate((balancing_bid, np.ones((balancing_facilities_new_num, T, H, num_bids)) * np.nan), axis=0)
            balancing_q = np.concatenate((balancing_q, np.ones((balancing_facilities_new_num, T, H, num_bids)) * np.nan), axis=0)
            balancing_facilities = np.concatenate((balancing_facilities, balancing_facilities_new))
            balancing_facilities_argsort = np.argsort(balancing_facilities)
            balancing_facilities = balancing_facilities[balancing_facilities_argsort]
            balancing_bid = balancing_bid[balancing_facilities_argsort,:,:,:] # resort
            balancing_q = balancing_q[balancing_facilities_argsort,:,:,:] # resort
            
        # Need to update num_bids?
        df_balancing_myr['bid_n'] = df_balancing_myr.groupby([df_facility_col, df_trading_col, df_interval_col], sort=False).cumcount() + 1
        num_bids_myr = np.max(df_balancing_myr['bid_n'])
        if num_bids_myr > num_bids:
            balancing_bid = np.concatenate((balancing_bid, np.ones((balancing_facilities.shape[0], T, H, num_bids_myr - num_bids)) * np.nan), axis=3)
            balancing_q = np.concatenate((balancing_q, np.ones((balancing_facilities.shape[0], T, H, num_bids_myr - num_bids)) * np.nan), axis=3)
            num_bids = num_bids_myr

        # Process bids
        df_balancing_myr = df_balancing_myr[[df_facility_col, df_trading_col, df_interval_col, price_col, q_col, 'bid_n']]
        df_balancing_myr.dropna(inplace=True)
        mux = pd.MultiIndex.from_product([balancing_facilities, dates_in_monthyr, range(1, H+1), range(1, num_bids+1)], names=(df_facility_col,df_trading_col,df_interval_col,'bid_n'))
        group = mux.to_frame(index=False).merge(df_balancing_myr, on=[df_facility_col, df_trading_col, df_interval_col, 'bid_n'], how="left").fillna(np.nan) # add the facility codes that are missing, also puts them in correct order
        
        # Add to balancing bids array
        balancing_bid[:,ctr:ctr+dates_in_monthyr.shape[0],:,:] = group[price_col].values.reshape((balancing_facilities.shape[0],dates_in_monthyr.shape[0],H,num_bids))
        balancing_q[:,ctr:ctr+dates_in_monthyr.shape[0],:,:] = group[q_col].values.reshape((balancing_facilities.shape[0],dates_in_monthyr.shape[0],H,num_bids))

    # Update month + year
    ctr += dates_in_monthyr.shape[0]
    yr = yr if month < 12 else yr + 1
    month = month + 1 if month < 12 else 1
    
# Adjust prices for inflation
balancing_bid = balancing_bid * convert_factor_inflation[np.newaxis,np.isin(dates_inflation, dates),np.newaxis,np.newaxis]

# Reshape to combine facilities and bid intervals
balancing_bid = np.reshape(np.moveaxis(balancing_bid, 0, -1), (balancing_bid.shape[1], balancing_bid.shape[2], balancing_bid.shape[0] * balancing_bid.shape[3]))
balancing_q = np.reshape(np.moveaxis(balancing_q, 0, -1), (balancing_q.shape[1], balancing_q.shape[2], balancing_q.shape[0] * balancing_q.shape[3]))
    
# Sort bids
bid_sort_idx = np.argsort(balancing_bid, axis=2)
balancing_bid = np.take_along_axis(balancing_bid, bid_sort_idx, axis=2)
balancing_q = np.take_along_axis(balancing_q, bid_sort_idx, axis=2)
del bid_sort_idx # don't need it any more and takes up a lot of memory

# Cumulative sum quantities
balancing_q = np.nancumsum(balancing_q, axis=2)
    
if save_arrays:
    np.savez_compressed(gv.balancing_arrays_file, balancing_bid=balancing_bid, balancing_q=balancing_q)
        
print(f"Balancing bids processed.", flush=True)
