#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 08:42:40 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of embedding function outputs into prompts with f-strings
and sending them to the OpenAI ChatCompletion endpoint.
"""



"""
Important Note:

- The climate .nc file is quite large and can not be uploaded on the 
    Github!

- The .nc file that is used in this code is saved here: https://zenodo.org/records/14979073

- You can also downlaod your favorite climate dataset from publicaly available website, and 
    apply the approapriate preprocessing steps.
    You can find more about the application of this dataset in the following article:

    " Climate Aware Deep Neural Networks (CADNN) for Wind Power Simulation": https://arxiv.org/abs/2412.12160
    Github repository: https://github.com/Ali-Forootani/neural_wind_model/tree/main
"""





import numpy as np
import sys
import os
def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir
root_dir = setting_directory(0)



from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init
from siren_modules import Siren

import warnings
import time

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#####################################
#####################################


from wind_dataset_preparation_psr import (
    extract_pressure_for_germany,
    extract_wind_speed_for_germany,
    load_real_wind_csv,
    interpolate_wind_speed,
    loading_wind,
    interpolate_pressure,
    scale_interpolated_data,
    combine_data,
    repeat_target_points,
    scale_target_points
    )


######################################
######################################



warnings.filterwarnings("ignore")


# ----------------------------------------------------------------
# OpenAI initialization (adjust credentials/URLs as needed)
# ----------------------------------------------------------------






import os
from openai import OpenAI


# Set OpenAI API key and base URL
os.environ["OPENAI_API_KEY"] = "Your API Key"
os.environ["OPENAI_API_BASE"] = "Your API Base"

api_key = "Your API Key"
api_base = "Your API Base"

# Initialize the OpenAI client
client = OpenAI(api_key=api_key, base_url=api_base)
# ----------------------------------------------------------------
# Main workflow illustrating how to pass function outputs into prompts
# ----------------------------------------------------------------

def main():
    
    
    # Example usage
    nc_file_path = 'nc_files/dataset-projections-2020/ps_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r3i1p1_GERICS-REMO2015_v1_3hr_202001010100-202012312200.nc'
    csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'
    
    # Extract pressure data
    pressure_data, grid_lats, grid_lons = extract_pressure_for_germany(nc_file_path)
    
    
    
    # Example usage
    nc_file_path = 'nc_files/Klima_Daten_10m_3h_2020_RCP26.nc'
    csv_file_path = 'Results_2020_REMix_ReSTEP_hourly_REF.csv'
    
    wind_speeds, grid_lats, grid_lons = extract_wind_speed_for_germany(nc_file_path)
    
    
    
    print(f"Shape of extracted wind speed: {wind_speeds.shape}")
    print(f"Sample of extracted wind speed (first 5 time steps, first 5 locations):")
    
    
    target_points = load_real_wind_csv(csv_file_path)
    interpolated_wind_speeds = interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points)
    
    scaled_unix_time_array, filtered_x_y, filtered_wind_power = loading_wind()
    
    interpolated_pressure = interpolate_pressure(pressure_data, grid_lats, grid_lons, target_points)
    
    
    scaled_wind_speeds = scale_interpolated_data(interpolated_wind_speeds)
    
    scaled_pressure = scale_interpolated_data(interpolated_pressure)
    
    scaled_wind_power = scale_interpolated_data(filtered_wind_power)
    
    scaled_target_points = scale_target_points(target_points)
    
    
    # Number of time steps (from scaled_wind_speeds)
    num_time_steps = scaled_wind_speeds.shape[0]
    repeated_scaled_target_points = repeat_target_points(scaled_target_points, num_time_steps)
    
    print(f"Shape of repeated_scaled_target_points: {repeated_scaled_target_points.shape}")
    
    
    
    # Combine the data
    combined_array = combine_data(scaled_target_points, scaled_unix_time_array,
                          scaled_wind_speeds,
                          scaled_pressure,
                          scaled_wind_power)
    
    #import numpy as np
    from scipy.stats import skew, kurtosis
    
    def compute_location_stats(data_array):
        """
        data_array shape => (time_steps, locations)
        returns dict of arrays each shaped => (locations,)
        
        Extends the basic statistics with additional
        features such as range, IQR, variance, skewness, etc.
        """
        # Handle edge cases where mean might be zero (for coefficient of variation)
        # by adding a small epsilon to avoid divide-by-zero errors.
        epsilon = 1e-8
        means = np.mean(data_array, axis=0)
        
        stats_dict = {
            "mean":                 means,  # shape => (locations,)
            "median":               np.median(data_array, axis=0),
            "std":                  np.std(data_array, axis=0),
            "variance":             np.var(data_array, axis=0),
            "min":                  np.min(data_array, axis=0),
            "max":                  np.max(data_array, axis=0),
            "range":                np.ptp(data_array, axis=0),  # peak-to-peak, max-min
            "25th_percentile":      np.percentile(data_array, 25, axis=0),
            "75th_percentile":      np.percentile(data_array, 75, axis=0),
            "iqr":                  np.percentile(data_array, 75, axis=0) 
                                    - np.percentile(data_array, 25, axis=0),
            "skewness":             skew(data_array, axis=0, bias=False),
            "kurtosis":             kurtosis(data_array, axis=0, bias=False),
            "coef_var":             np.std(data_array, axis=0) 
                                    / (means + epsilon),  # Coefficient of variation
        }
        
        return stats_dict

    
    wind_speeds_stats   = compute_location_stats(interpolated_wind_speeds)
    pressure_stats      = compute_location_stats(interpolated_pressure)
    
    
    # For example, let’s create a brief textual summary:
    locationwise_summaries = []
    for loc_idx in range(232):
        loc_summary = (
        f"Location {loc_idx}:\n"
        f"  Wind Speed Stats:\n"
        f"    Mean    = {wind_speeds_stats['mean'][loc_idx]:.3f}\n"
        f"    Median  = {wind_speeds_stats['median'][loc_idx]:.3f}\n"
        f"    StdDev  = {wind_speeds_stats['std'][loc_idx]:.3f}\n"
        f"    Variance= {wind_speeds_stats['variance'][loc_idx]:.3f}\n"
        f"    Range   = {wind_speeds_stats['range'][loc_idx]:.3f}\n"
        f"    25%     = {wind_speeds_stats['25th_percentile'][loc_idx]:.3f}\n"
        f"    75%     = {wind_speeds_stats['75th_percentile'][loc_idx]:.3f}\n"
        f"    IQR     = {wind_speeds_stats['iqr'][loc_idx]:.3f}\n"
        f"    Skewness= {wind_speeds_stats['skewness'][loc_idx]:.3f}\n"
        f"    Kurtosis= {wind_speeds_stats['kurtosis'][loc_idx]:.3f}\n"
        f"    CoefVar = {wind_speeds_stats['coef_var'][loc_idx]:.3f}\n"
        f"  Pressure Stats:\n"
        f"    Mean    = {pressure_stats['mean'][loc_idx]:.3f}\n"
        f"    Median  = {pressure_stats['median'][loc_idx]:.3f}\n"
        f"    StdDev  = {pressure_stats['std'][loc_idx]:.3f}\n"
        f"    Variance= {pressure_stats['variance'][loc_idx]:.3f}\n"
        f"    Range   = {pressure_stats['range'][loc_idx]:.3f}\n"
        f"    25%     = {pressure_stats['25th_percentile'][loc_idx]:.3f}\n"
        f"    75%     = {pressure_stats['75th_percentile'][loc_idx]:.3f}\n"
        f"    IQR     = {pressure_stats['iqr'][loc_idx]:.3f}\n"
        f"    Skewness= {pressure_stats['skewness'][loc_idx]:.3f}\n"
        f"    Kurtosis= {pressure_stats['kurtosis'][loc_idx]:.3f}\n"
        f"    CoefVar = {pressure_stats['coef_var'][loc_idx]:.3f}\n"
            )
    
        print(loc_summary)

        locationwise_summaries.append(loc_summary)
        
    num_locations_to_show = 10
    # Join them into a single string (but be mindful of token limits!)
    # If needed, consider summarizing further or only printing a subset of locations.
    stats_string = "\n".join(locationwise_summaries[:num_locations_to_show])  # only first 10 locations for brevity

    # We might also want general stats across all locations to give the big picture:
    overall_mean_ws = np.mean(interpolated_wind_speeds)
    overall_mean_p  = np.mean(interpolated_pressure)
    
    
    
    
    prompt = f"""
Below are location-wise statistical features for wind speeds and pressure data
over a 1-year period (2928 time steps at 3-hour intervals, 232 locations).

General statistics (all locations, all time steps):
- Overall Mean Wind Speed: {overall_mean_ws:.3f}
- Overall Mean Pressure:   {overall_mean_p:.3f}

Sample of per-location stats (first 10 locations):

    
    Given these statistics:
1. Identify any interesting patterns in wind speed and pressure across the locations.
2. Suggest how these patterns could inform the optimal placement of wind turbines
   or the best time periods for energy capture.
3. Highlight if additional data or transformations might improve downstream analysis.
"""
    


    prompt = f"""
    Below are location-wise **extended** statistical features for wind speeds and pressure data
    collected over a 1-year period (2928 time steps at 3-hour intervals, across 232 locations).
    
    The following metrics are provided for each location:
    • Mean                • Median              • Standard Deviation
    • Variance            • Min, Max            • Range
    • 25th Percentile     • 75th Percentile     • Interquartile Range (IQR)
    • Skewness            • Kurtosis            • Coefficient of Variation 
    
    Overall (all locations, all time steps) aggregates include:
    • Overall Mean Wind Speed: {overall_mean_ws:.3f}
    • Overall Mean Pressure:   {overall_mean_p:.3f}
    
    Sample of per-location extended stats (first {num_locations_to_show} locations):
    {stats_string}
    
    Using these extended statistics, please address the following:
    
    1. **Distribution & Patterns**  
       - Comment on any notable patterns in mean, standard deviation, range, or skewness for wind speed and pressure.
       - Identify whether certain locations exhibit heavy tails or extreme values (based on kurtosis) and any implications for site selection or forecasting.
    
    2. **Optimal Placement & Time Periods**  
       - Explain how the observed patterns could inform optimal turbine placement.
       - Highlight which time periods or seasons might yield the highest energy output.
    
    3. **Data Quality & Transformations**  
       - Suggest if any data transformations (e.g., log-transform, outlier removal, normalization) might be warranted based on skewness/kurtosis.
       - Propose additional data (e.g., temperature, humidity, topographic) that could refine site-selection or energy forecasting models.
    
    4. **Further Analysis**  
       - Discuss whether metrics like coefficient of variation or range reveal anomalies/outliers needing investigation.
       - If relevant, suggest how these statistics could guide more advanced modeling approaches.
       """
    

    import os
    from openai import OpenAI


    # Set OpenAI API key and base URL
    os.environ["OPENAI_API_KEY"] = "Your API Key"
    os.environ["OPENAI_API_BASE"] = "Your API Base"

    api_key = "Your API Key"
    api_base = "Your API Base"

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant skilled in climate dataset analysis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # -----------------------------
    # 6) Print the response
    # -----------------------------
    print("OpenAI Response:")
    print(response.choices[0].message.content)




if __name__ == "__main__":
    main()
