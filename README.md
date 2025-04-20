# Chicago Crime Classification Project

> **Note:** A full run takes approximately **4 minutes** on average.
>
> **Important:** **You must download the dataset (not included in this repository) from [this link](https://drive.google.com/file/d/18f47YB0SgvsYPEUG7an4AxoDGzE7ThwA/view?usp=sharing) for the code to work.**

This repository contains a comprehensive implementation of crime classification models for Chicago crime data, exploring various approaches to categorize crimes based on their characteristics.

## Getting Started

1. **Install Python** (version 3.13.1 or higher recommended) or ensure it is available on your system.
2. **Clone this repository**:
   ```bash
   git clone https://github.com/FKande/chicago-crime-classification.git
   cd chicago-crime-classification
   ```
3. **Optional:** Create & activate a virtual environment (only if you wish to isolate dependencies and avoid impacting your global Python packages):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Prerequisites

### Python Version

- Tested with **Python 3.13.1**. Verify your version:
  ```bash
  python --version
  ```
- If you need to switch, consider using `pyenv` or your system package manager as described below.

### Optional: Virtual Environment

If you already have a working Python environment and are comfortable managing packages globally, you can skip this. Otherwise, to isolate dependencies and avoid conflicts, you can create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Required Libraries

- Install all required libraries using the provided `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
- For most Python users, running `pip install -r requirements.txt` should work without issues.

**Fallback tips (if you run into problems):**
- If you encounter version conflicts, you can install known compatible versions:
  ```bash
  pip install pandas==2.2.3 scikit-learn==1.6.1 matplotlib==3.10.0 seaborn==0.13.2 numpy==2.2.2 imbalanced-learn==0.13.0 scipy==1.15.1
  ```
- To capture your exact environment for reproducibility:
  ```bash
  pip list --format=freeze > requirements-local.txt
  ```
  ```bash
  pip list --format=freeze > requirements-local.txt
  ```

## Data Setup

1. Download the Chicago crime dataset from [this link](https://drive.google.com/file/d/18f47YB0SgvsYPEUG7an4AxoDGzE7ThwA/view?usp=sharing).
2. Unzip the downloaded file.
3. Ensure you are in the root directory of the project (`chicago-crime-classification`).
4. **IMPORTANT:** When you unzip, you may get a folder (e.g., `chicago_crime/`) containing the CSV. Move **only** the bare `chicago_crime.csv` file—**not** its parent folder—into the `data/` directory:
   ```bash
   mv path/to/chicago_crime_classification/chicago_crime.csv data/
   ```

## Running the Implementation

Execute the main script:
```bash
python implementation.py
```
If that fails on your system, you can also try:
```bash
py implementation.py
```

This will:
1. Load the Chicago crime dataset
2. Run five different implementations of crime classification models
3. Compare the results across all implementations
4. Generate visualizations in the `figures` directory

> **Note:** A full run takes approximately **4 minutes** on average, depending on your hardware.

## Project Structure

```
chicago-crime-classification/
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

After running the script, you'll see a detailed comparison of all implementations in your terminal. Visualizations will be saved to the `figures` directory, including:

- Model accuracy comparison across implementations
- Performance analysis for specific crime categories
- Temperature effect analysis (for implementation 5)

## Troubleshooting

- If you encounter memory issues, try running the script on a machine with more RAM.
- If temperature data processing fails, ensure you have sufficient disk space.
- Check error messages for specific guidance or missing dependencies.

