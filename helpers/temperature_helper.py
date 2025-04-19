import pandas as pd
import numpy as np
import time
from datetime import datetime

def log_progress(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def create_temperature_lookup_file():
    start_time = time.time()
    log_progress("Starting temperature lookup file creation")
    
    # Load the temperature data
    log_progress("Loading temperature data...")
    temp_df = pd.read_csv("data/temperature.csv")
    
    # Convert datetime column to datetime type and create date and hour columns
    log_progress("Processing datetime columns...")
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    temp_df['date'] = temp_df['datetime'].dt.date.astype(str)  # Convert to string for easier lookup
    temp_df['hour'] = temp_df['datetime'].dt.hour
    
    # Check for Chicago column or use a nearby city
    temp_column = 'Chicago'
    if 'Chicago' not in temp_df.columns:
        nearby_cities = ['Indianapolis', 'Detroit', 'Minneapolis', 'Saint Louis']
        for city in nearby_cities:
            if city in temp_df.columns:
                temp_column = city
                log_progress(f"Using {city} temperature data as proxy.")
                break
        else:
            # If no nearby city, use the first temperature column available
            temp_columns = [col for col in temp_df.columns if col not in ['datetime', 'date', 'hour']]
            if temp_columns:
                temp_column = temp_columns[0]
                log_progress(f"Using {temp_column} temperature data as proxy.")
    
    # Create a simplified lookup dataframe with just the columns we need
    log_progress("Creating simplified lookup table...")
    lookup_df = temp_df[['date', 'hour', temp_column]].copy()
    lookup_df.rename(columns={temp_column: 'temperature'}, inplace=True)
    
    # Calculate mean temperature for missing values
    mean_temp = lookup_df['temperature'].mean()
    log_progress(f"Mean temperature: {mean_temp:.2f}K")
    
    # Fill any missing values with the mean temperature
    missing_temps = lookup_df['temperature'].isna().sum()
    if missing_temps > 0:
        log_progress(f"Filling {missing_temps} missing temperature values with mean")
        lookup_df['temperature'].fillna(mean_temp, inplace=True)
    
    # Save the lookup table
    output_file = "data/temperature_lookup.csv"
    log_progress(f"Saving lookup table to {output_file}...")
    lookup_df.to_csv(output_file, index=False)
    
    elapsed_time = time.time() - start_time
    log_progress(f"Temperature lookup file created with {len(lookup_df)} entries in {elapsed_time:.2f} seconds")
    log_progress(f"Lookup table saved to {output_file}")

if __name__ == "__main__":
    create_temperature_lookup_file()