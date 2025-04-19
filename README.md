# Chicago Crime Classification Project

This repository contains a comprehensive implementation of crime classification models for Chicago crime data, exploring various approaches to categorize crimes based on their characteristics.

## Prerequisites

### Python Version

- Tested with **Python 3.13.1**. Verify your version:
  ```bash
  python --version
  ```

### Editor

- It is recommended to use a text editor or IDE such as **Visual Studio Code** (VSCode) for editing and running scripts.

### Virtual Environment

- It is best practice to create and activate a virtual environment before installing dependencies:
  ```bash
  python -m venv venv
  source venv/bin/activate   # On Windows use `venv\\Scripts\\activate`
  ```

### Required Libraries

- Install all required libraries using the provided `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

- If you encounter version conflicts, you can install known compatible versions:
  ```bash
  pip install pandas==2.2.3 scikit-learn==1.6.1 matplotlib==3.10.0 seaborn==0.13.2 numpy==2.2.2 imbalanced-learn==0.13.0 scipy==1.15.1 xgboost==3.0.0 anyio==4.9.0 argon2-cffi==23.1.0
  ```

- To capture your exact environment for reproducibility, you can run:
  ```bash
  pip list --format=freeze > requirements-local.txt
  ```

## Data Setup

1. Download the Chicago crime dataset from [this Google Drive link](https://drive.google.com/file/d/18f47YB0SgvsYPEUG7an4AxoDGzE7ThwA/view?usp=sharing).
2. Unzip the downloaded file.
3. Ensure you are in the root directory of the project (`chicago-crime`).
4. Place the unzipped `chicago_crime.csv` file in the `data` folder:
   ```bash
   mv path/to/chicago_crime.csv data/
   ```

## Running the Implementation

Execute the main script:
```bash
python implementation.py
```

This will:
1. Load the Chicago crime dataset
2. Run five different implementations of crime classification models
3. Compare the results across all implementations
4. Generate visualizations in the `figures` directory

> **Note:** A full run takes approximately **4 minutes** on average, depending on your hardware.

## Project Structure

```
chicago-crime/
├── data/                      # Datasets
│   ├── chicago_crime.csv      # Main crime dataset (you must download this)
│   ├── temperature.csv        # Generated temperature data
│   └── temperature_lookup.csv # Precomputed lookup for temperature
├── figures/                   # Output visualizations (generated)
├── helpers/                   # Utility scripts
│   ├── crime_category_counter.py
│   └── temperature_helper.py
├── implementation.py          # Main script with all implementations
├── requirements.txt           # Core dependencies
└── README.md                  # Project documentation
```

## Implementations

The project includes five different implementations:

1. **Original (4 categories):** Uses the 4 umbrella categories as described in the paper
2. **Original Categories (Unbalanced):** Uses original crime categories without resampling
3. **Balanced (2246 per category):** Uses a balanced dataset with 2246 samples per category
4. **Balanced (5000 per category):** Uses a balanced dataset with 5000 samples per category
5. **Balanced (5000/cat) + Temperature:** Adds temperature data to the balanced dataset

Each implementation trains both Decision Tree and Naive Bayes models and evaluates their performance.

## Results

After running the script, you'll see a comparison of all implementations in the terminal output. Visualizations will be saved to the `figures` directory, including:

- Model accuracy comparison across implementations
- Performance analysis for specific crime categories
- Temperature effect analysis (for implementation 5)

## Troubleshooting

- If you encounter memory issues, try running the script on a machine with more RAM.
- If temperature data processing fails, ensure you have sufficient disk space.
- Check error messages for specific guidance or missing dependencies.

