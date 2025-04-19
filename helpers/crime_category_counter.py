import pandas as pd

def count_crime_categories(csv_file_path):
    """
    Counts the instances of each crime category in the Chicago crime CSV file.
    
    Parameters:
    csv_file_path (str): Path to the Chicago crime CSV file
    
    Returns:
    pandas.Series: Crime categories sorted by count in descending order
    """
    # Load the CSV file
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path, low_memory=False)
    
    # Count instances of each crime category in 'Primary Type' column
    crime_counts = df['Primary Type'].value_counts()
    
    # Print total number of records
    total_records = len(df)
    print(f"\nTotal number of crime records: {total_records}")
    
    # Print crime categories sorted by count
    print("\nCrime categories by number of instances:")
    for crime_type, count in crime_counts.items():
        percentage = (count / total_records) * 100
        print(f"{crime_type}: {count} ({percentage:.2f}%)")
    
    return crime_counts

def save_crime_stats(csv_file_path, output_file='data/crime_statistics.csv'):
    """
    Saves crime category statistics to a CSV file.
    
    Parameters:
    csv_file_path (str): Path to the Chicago crime CSV file
    output_file (str): Path to save the output statistics
    """
    crime_counts = count_crime_categories(csv_file_path)
    
    # Convert to DataFrame for easier saving
    crime_stats = pd.DataFrame({
        'Crime_Type': crime_counts.index,
        'Count': crime_counts.values,
        'Percentage': (crime_counts.values / sum(crime_counts.values)) * 100
    })
    
    # Save to CSV
    crime_stats.to_csv(output_file, index=False)
    print(f"\nCrime statistics saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    csv_file_path = "data/chicago_crime.csv"
    crime_counts = count_crime_categories(csv_file_path)
    save_crime_stats(csv_file_path)