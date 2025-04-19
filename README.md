# Chicago Crime Classification Project

This repository contains a comprehensive implementation of crime classification models for Chicago crime data, exploring various approaches to categorize crimes based on their characteristics.

## Prerequisites

### Python Version
This project was developed using Python 3.13.1. You can check your Python version with:
python --version

### Required Libraries

Install all required libraries using the requirements file:
pip install -r requirements.txt

If you encounter any version conflicts, you can try installing specific versions of problematic packages:
pip install pandas==2.2.3
pip install scikit-learn==1.6.1
pip install matplotlib==3.10.0
pip install seaborn==0.13.2

## Data Setup

1. Download the Chicago crime dataset from [this Google Drive link](https://drive.google.com/file/d/18f47YB0SgvsYPEUG7an4AxoDGzE7ThwA/view?usp=sharing)

2. Unzip the downloaded file

3. Make sure you're in the root directory of the project (`chicago-crime`)

4. Place the unzipped `chicago_crime.csv` file in the `data` folder:
   ```bash
   mv path/to/chicago_crime.csv data/
   ```

## Running the Implementation

Execute the main implementation script: python implementation.py

This will:
1. Load the Chicago crime dataset
2. Run five different implementations of crime classification models
3. Compare the results across all implementations
4. Generate visualizations in the `figures` directory

## Project Structure

- `implementation.py`: Main script containing all implementations and comparison logic
- `helpers/`: Directory containing helper scripts
  - `temperature_helper.py`: Utility for processing temperature data
  - `crime_category_counter.py`: Utility for analyzing crime category distributions
- `data/`: Directory for storing datasets
  - `chicago_crime.csv`: Main crime dataset (you need to download this)
  - `temperature.csv`: Temperature data (generated during execution)
  - `temperature_lookup.csv`: Lookup table for temperature data (generated during execution)
- `figures/`: Directory where visualizations are saved (created during execution)

## Implementations

The project includes five different implementations:

1. **Original (4 categories)**: Uses the 4 umbrella categories as described in the paper
2. **Original Categories (Unbalanced)**: Uses original crime categories without resampling
3. **Balanced (2246 per category)**: Uses a balanced dataset with 2246 samples per category
4. **Balanced (5000 per category)**: Uses a balanced dataset with 5000 samples per category
5. **Balanced (5000/cat) + Temperature**: Adds temperature data to the balanced dataset

Each implementation trains both Decision Tree and Naive Bayes models and evaluates their performance.

## Results

After running the implementation, you'll see a comparison of all implementations in the terminal output. Visualizations will be saved to the `figures` directory, including:

- Model accuracy comparison across implementations
- Performance analysis for specific crime categories
- Temperature effect analysis (for implementation 5)

## Troubleshooting

- If you encounter memory issues, try running the script on a machine with more RAM
- If the temperature data processing fails, ensure you have sufficient disk space
- For any other issues, check the error messages for specific guidance


