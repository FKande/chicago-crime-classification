import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import os
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def log_progress(message):
    """Helper function to log progress with timestamps"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

###############################################################################
# ORIGINAL IMPLEMENTATION - 4 UMBRELLA CATEGORIES
###############################################################################
def original_implementation(crime_df):
    """
    Original implementation using the 4 umbrella categories as described in the paper
    
    Parameters:
    crime_df (DataFrame): The loaded Chicago crime dataset
    """
    start_implementation_time = time.time()
    log_progress("Starting original implementation...")
    
    print("\n\n" + "="*80)
    print("ORIGINAL IMPLEMENTATION - 4 UMBRELLA CATEGORIES")
    print("="*80)
    
    # 1. LOAD AND FILTER THE DATA
    log_progress("Loading and filtering data...")
    start_time = time.time()
    
    # Create a deep copy to avoid modifying the original dataframe
    df = crime_df.copy(deep=True)
    
    # Convert 'Date' to datetime and filter for years 2013–2017
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]

    # Randomly sample 12,109 records (matching the paper's final record count)
    df = df.sample(n=12109, random_state=100).copy()

    print("Data shape after year filter and sampling:", df.shape)
    log_progress(f"Data loading completed in {time.time() - start_time:.2f} seconds")

    # 2. REDUCE TO RELEVANT COLUMNS (SIMILAR TO THE PAPER'S 18 INITIAL FEATURES)
    log_progress("Starting feature selection...")
    start_time = time.time()
    possible_columns = [
        'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description',
        'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward',
        'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year'
    ]

    # Keep only columns that exist in your dataset
    df = df[[col for col in possible_columns if col in df.columns]]

    # Drop duplicates and remove any missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    log_progress(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # 3. RECLASSIFY CRIMES INTO THE 4 CATEGORIES EXACTLY AS THE PAPER STATES
    def reclassify_crime(crime_type):
        """
        Map the original 'Primary Type' into the four categories:
        1) Forbidden Practices
        2) Theft
        3) Assault
        4) Public Peace Violation
        """
        # Convert to string uppercase
        ct = str(crime_type).upper()

        forbidden_practices = [
            'NARCOTICS', 'OTHER NARCOTIC VIOLATION', 'PROSTITUTION', 'GAMBLING', 'OBSCENITY'
        ]
        theft = [
            'BURGLARY', 'DECEPTIVE PRACTICE', 'MOTOR VEHICLE THEFT', 'ROBBERY'
        ]
        assault = [
            'CRIM SEXUAL ASSAULT', 'OFFENSE INVOLVING CHILDREN', 'SEX OFFENSE',
            'HOMICIDE', 'HUMAN TRAFFICKING'
        ]
        public_violation = [
            'WEAPONS VIOLATION', 'CRIMINAL DAMAGE', 'CRIMINAL TRESPASS', 'ARSON',
            'KIDNAPPING', 'STALKING', 'INTIMIDATION', 'PUBLIC INDECENCY', 'CRIMINAL DEFACEMENT'
        ]

        if ct in forbidden_practices:
            return 'FORBIDDEN_PRACTICES'
        elif ct in theft:
            return 'THEFT'
        elif ct in assault:
            return 'ASSAULT'
        elif ct in public_violation:
            return 'PUBLIC_PEACE_VIOLATION'
        else:
            return 'OTHER'

    df['Crime_Category'] = df['Primary Type'].apply(reclassify_crime)

    # Keep only the four categories described in the paper
    df = df[df['Crime_Category'] != 'OTHER']

    print("Data shape after reclassification & removing 'OTHER':", df.shape)
    print(df['Crime_Category'].value_counts())

    # 4. FEATURE SELECTION USING BACKWARD ELIMINATION
    log_progress("Starting feature selection...")
    start_time = time.time()
    candidate_features = [
        col for col in df.columns
        if col not in ['Crime_Category', 'Primary Type', 'Case Number', 'Date', 'Year']
    ]

    # Convert categorical columns to category codes
    for col in candidate_features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    X_all = df[candidate_features].copy()
    y_all = df['Crime_Category'].copy()

    def evaluate_feature_set(features):
        """Train a quick Decision Tree and return accuracy on a 70-30 split."""
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], y_all,
            test_size=0.3,
            random_state=100,
            stratify=y_all
        )
        
        dt_temp = DecisionTreeClassifier(criterion='entropy', random_state=100)
        dt_temp.fit(X_train, y_train)
        y_pred = dt_temp.predict(X_test)
        return accuracy_score(y_test, y_pred)

    current_features = candidate_features[:]

    improved = True
    while improved and len(current_features) > 9:
        improved = False
        best_accuracy = evaluate_feature_set(current_features)
        
        # Check if dropping each feature improves accuracy
        drop_candidates = []
        for f in current_features:
            temp_features = [x for x in current_features if x != f]
            acc = evaluate_feature_set(temp_features)
            drop_candidates.append((f, acc))
        
        # Sort by accuracy descending
        drop_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # If dropping a feature improves accuracy, remove it
        top_feature, top_acc = drop_candidates[0]
        if top_acc > best_accuracy:
            current_features.remove(top_feature)
            improved = True

    print("\nSelected features after approximate backward elimination:")
    print(current_features)
    log_progress(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # 5. TRAIN/TEST SPLIT WITH FINAL FEATURE SET
    X = df[current_features]
    y = df['Crime_Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=100,
        stratify=y
    )

    # Check distribution
    print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class distribution:\n", y_train.value_counts())
    print("Test class distribution:\n", y_test.value_counts())

    # 6. TRAIN THE CLASSIFIERS (DECISION TREE & NAIVE BAYES)
    log_progress("Training models...")
    start_time = time.time()
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)
    log_progress(f"Model training completed in {time.time() - start_time:.2f} seconds")

    # 7. EVALUATE THE RESULTS
    log_progress("Evaluating models...")
    start_time = time.time()
    def evaluate_model(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{name} RESULTS")
        print(f"Accuracy: {acc*100:.2f}% | Precision: {prec:.3f} | Recall: {rec:.3f}")
        
        # Class-by-class correct classification (compact format)
        for c in sorted(y_true.unique()):
            c_mask = (y_true == c)
            correct = (y_true[c_mask] == y_pred[c_mask]).sum()
            total = c_mask.sum()
            print(f"Class '{c}': Correct: {correct} | Incorrect: {total - correct} | Total: {total}")

    evaluate_model("Decision Tree (Entropy)", y_test, dt_pred)
    evaluate_model("Naive Bayes", y_test, nb_pred)
    
    elapsed_time = time.time() - start_implementation_time
    log_progress(f"Original implementation completed in {elapsed_time:.2f} seconds")
    
    return {
        "implementation": "Original (4 categories)",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred,
        "elapsed_time": elapsed_time
    }

###############################################################################
# IMPLEMENTATION 2 - ORIGINAL CATEGORIES (UNBALANCED)
###############################################################################
def original_categories_implementation(crime_df):
    """
    Implementation using original crime categories without resampling
    """
    start_implementation_time = time.time()
    log_progress("Starting implementation...")
    
    print("\n\n" + "="*80)
    print("IMPLEMENTATION 2 - ORIGINAL CATEGORIES (UNBALANCED)")
    print("="*80)
    
    # 1. LOAD AND FILTER THE DATA
    log_progress("Loading and filtering data...")
    start_time = time.time()

    # Create a deep copy to avoid modifying the original dataframe
    df = crime_df.copy(deep=True)

    # Convert 'Date' to datetime and filter for years 2013–2017
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]

    # Randomly sample 12,109 records
    df = df.sample(n=12109, random_state=100).copy()

    print("Data shape after year filter and sampling:", df.shape)
    log_progress(f"Data loading completed in {time.time() - start_time:.2f} seconds")

    # 2. REDUCE TO RELEVANT COLUMNS
    log_progress("Starting feature selection...")
    start_time = time.time()
    possible_columns = [
        'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description',
        'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward',
        'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year'
    ]

    # Keep only columns that exist in your dataset
    df = df[[col for col in possible_columns if col in df.columns]]

    # Drop duplicates and remove any missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    log_progress(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # 3. FILTER OUT LOW-FREQUENCY CRIME CATEGORIES
    categories_to_exclude = [
        'PUBLIC INDECENCY', 'NON-CRIMINAL', 'OTHER NARCOTIC VIOLATION', 'HUMAN TRAFFICKING',
        'NON - CRIMINAL', 'RITUALISM', 'NON-CRIMINAL (SUBJECT SPECIFIED)', 'DOMESTIC VIOLENCE',
        'OBSCENITY', 'CONCEALED CARRY LICENSE VIOLATION'
    ]

    df = df[~df['Primary Type'].isin(categories_to_exclude)]

    # Using original crime categories
    print("\nOriginal crime categories distribution:")
    print(df['Primary Type'].value_counts())

    # 4. FEATURE SELECTION USING BACKWARD ELIMINATION
    log_progress("Starting feature selection...")
    start_time = time.time()
    candidate_features = [
        col for col in df.columns
        if col not in ['Primary Type', 'Case Number', 'Date', 'Year']
    ]

    # Convert categorical columns to category codes
    for col in candidate_features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    X_all = df[candidate_features].copy()
    y_all = df['Primary Type'].copy()

    def evaluate_feature_set(features):
        """Train a quick Decision Tree and return accuracy on a 70-30 split."""
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], y_all,
            test_size=0.3,
            random_state=100,
            stratify=y_all
        )
        
        dt_temp = DecisionTreeClassifier(criterion='entropy', random_state=100)
        dt_temp.fit(X_train, y_train)
        y_pred = dt_temp.predict(X_test)
        return accuracy_score(y_test, y_pred)

    current_features = candidate_features[:]

    improved = True
    while improved and len(current_features) > 9:
        improved = False
        best_accuracy = evaluate_feature_set(current_features)
        
        # Check if dropping each feature improves accuracy
        drop_candidates = []
        for f in current_features:
            temp_features = [x for x in current_features if x != f]
            acc = evaluate_feature_set(temp_features)
            drop_candidates.append((f, acc))
        
        # Sort by accuracy descending
        drop_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # If dropping a feature improves accuracy, remove it
        top_feature, top_acc = drop_candidates[0]
        if top_acc > best_accuracy:
            current_features.remove(top_feature)
            improved = True

    print("\nSelected features after approximate backward elimination:")
    print(current_features)
    log_progress(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # 5. TRAIN/TEST SPLIT WITH FINAL FEATURE SET
    X = df[current_features]
    y = df['Primary Type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=100,
        stratify=y
    )

    # Check distribution
    print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class distribution:\n", y_train.value_counts())
    print("Test class distribution:\n", y_test.value_counts())

    # 6. TRAIN THE CLASSIFIERS
    log_progress("Training models...")
    start_time = time.time()
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)
    log_progress(f"Model training completed in {time.time() - start_time:.2f} seconds")

    # 7. EVALUATE THE RESULTS
    log_progress("Evaluating models...")
    start_time = time.time()
    def evaluate_model(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{name} RESULTS")
        print(f"Accuracy: {acc*100:.2f}% | Precision: {prec:.3f} | Recall: {rec:.3f}")
        
        # Class-by-class correct classification (compact format)
        for c in sorted(y_true.unique()):
            c_mask = (y_true == c)
            correct = (y_true[c_mask] == y_pred[c_mask]).sum()
            total = c_mask.sum()
            print(f"Class '{c}': Correct: {correct} | Incorrect: {total - correct} | Total: {total}")

    evaluate_model("Decision Tree (Entropy)", y_test, dt_pred)
    evaluate_model("Naive Bayes", y_test, nb_pred)
    
    elapsed_time = time.time() - start_implementation_time
    log_progress(f"Implementation completed in {elapsed_time:.2f} seconds")
    
    return {
        "implementation": "Original Categories (Unbalanced)",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred,
        "elapsed_time": elapsed_time
    }

###############################################################################
# IMPLEMENTATION 3 - BALANCED DATASET (2246 SAMPLES PER CATEGORY)
###############################################################################
def balanced_implementation_2246(crime_df):
    """
    Implementation using balanced dataset with 2246 samples per category
    """
    start_implementation_time = time.time()
    log_progress("Starting implementation...")
    
    print("\n\n" + "="*80)
    print("IMPLEMENTATION 3 - BALANCED DATASET (2246 SAMPLES PER CATEGORY)")
    print("="*80)
    
    # 1. LOAD AND FILTER THE DATA
    log_progress("Loading and filtering data...")
    start_time = time.time()

    # Create a deep copy to avoid modifying the original dataframe
    df = crime_df.copy(deep=True)

    # Convert 'Date' to datetime and filter for years 2013–2017
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]

    # Filter out low-frequency crime categories
    categories_to_exclude = [
        'PUBLIC INDECENCY', 'NON-CRIMINAL', 'OTHER NARCOTIC VIOLATION', 'HUMAN TRAFFICKING',
        'NON - CRIMINAL', 'RITUALISM', 'NON-CRIMINAL (SUBJECT SPECIFIED)', 'DOMESTIC VIOLENCE'
    ]

    df = df[~df['Primary Type'].isin(categories_to_exclude)]

    # 2. DEFINE COLUMNS AND CHECK CATEGORY COUNTS
    log_progress("Starting feature selection...")
    start_time = time.time()
    possible_columns = [
        'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description',
        'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward',
        'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year'
    ]

    # Filter categorical data and check which categories have at least 2246 samples after cleaning
    sufficient_categories = []
    category_counts = {}

    for category in df['Primary Type'].unique():
        category_df = df[df['Primary Type'] == category]
        
        # Drop any rows with missing values in our possible columns
        category_df = category_df.dropna(subset=[col for col in possible_columns if col in category_df.columns])
        
        # Drop duplicates to ensure clean data
        category_df = category_df.drop_duplicates()
        
        # Count remaining samples
        count = len(category_df)
        category_counts[category] = count
        
        # Check if we have at least 2246 samples
        if count >= 2246:
            sufficient_categories.append(category)

    print(f"Categories with at least 2246 clean samples: {len(sufficient_categories)}")
    for cat in sufficient_categories:
        print(f"{cat}: {category_counts[cat]}")

    # Filter to only include categories with sufficient samples
    df = df[df['Primary Type'].isin(sufficient_categories)]

    # Sample exactly 2246 from each category
    samples_per_category = 2246
    print(f"Taking exactly {samples_per_category} samples from each of {len(sufficient_categories)} categories")

    # Create an empty dataframe to store our balanced sample
    balanced_df = pd.DataFrame()

    # Sample equally from each category
    for category in sufficient_categories:
        category_df = df[df['Primary Type'] == category]
        
        # Drop any rows with missing values in our possible columns
        category_df = category_df.dropna(subset=[col for col in possible_columns if col in category_df.columns])
        
        # Drop duplicates to ensure clean data
        category_df = category_df.drop_duplicates()
        
        # Take exactly 2246 samples
        sampled = category_df.sample(n=samples_per_category, random_state=100)
        balanced_df = pd.concat([balanced_df, sampled])

    # Shuffle the final dataframe
    df = balanced_df.sample(frac=1, random_state=100).reset_index(drop=True)

    print("Data shape after balanced sampling:", df.shape)

    # Keep only columns that exist in your dataset
    df = df[[col for col in possible_columns if col in df.columns]]

    # Verify that we have exactly 2246 samples per category
    print("\nVerifying category distribution:")
    category_counts = df['Primary Type'].value_counts()
    print(category_counts)
    print(f"All categories have exactly {samples_per_category} samples: {(category_counts == samples_per_category).all()}")

    print("\nOriginal crime categories distribution:")
    print(df['Primary Type'].value_counts())
    log_progress(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # 3. FEATURE SELECTION
    log_progress("Starting feature selection...")
    start_time = time.time()
    candidate_features = [
        col for col in df.columns
        if col not in ['Primary Type', 'Case Number', 'Date', 'Year']
    ]

    # Convert categorical columns to category codes
    for col in candidate_features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    X_all = df[candidate_features].copy()
    y_all = df['Primary Type'].copy()

    def evaluate_feature_set(features):
        """Train a quick Decision Tree and return accuracy on a 70-30 split."""
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], y_all,
            test_size=0.3,
            random_state=100,
            stratify=y_all
        )
        
        dt_temp = DecisionTreeClassifier(criterion='entropy', random_state=100)
        dt_temp.fit(X_train, y_train)
        y_pred = dt_temp.predict(X_test)
        return accuracy_score(y_test, y_pred)

    current_features = candidate_features[:]

    improved = True
    while improved and len(current_features) > 9:
        improved = False
        best_accuracy = evaluate_feature_set(current_features)
        
        # Check if dropping each feature improves accuracy
        drop_candidates = []
        for f in current_features:
            temp_features = [x for x in current_features if x != f]
            acc = evaluate_feature_set(temp_features)
            drop_candidates.append((f, acc))
        
        # Sort by accuracy descending
        drop_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # If dropping a feature improves accuracy, remove it
        top_feature, top_acc = drop_candidates[0]
        if top_acc > best_accuracy:
            current_features.remove(top_feature)
            improved = True

    print("\nSelected features after approximate backward elimination:")
    print(current_features)
    log_progress(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # 4. TRAIN/TEST SPLIT WITH FINAL FEATURE SET
    X = df[current_features]
    y = df['Primary Type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=100,
        stratify=y
    )

    # Check distribution
    print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class distribution:\n", y_train.value_counts())
    print("Test class distribution:\n", y_test.value_counts())

    # 5. TRAIN THE CLASSIFIERS
    log_progress("Training models...")
    start_time = time.time()
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)
    log_progress(f"Model training completed in {time.time() - start_time:.2f} seconds")

    # 6. EVALUATE THE RESULTS
    log_progress("Evaluating models...")
    start_time = time.time()
    def evaluate_model(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{name} RESULTS")
        print(f"Accuracy: {acc*100:.2f}% | Precision: {prec:.3f} | Recall: {rec:.3f}")
        
        # Class-by-class correct classification (compact format)
        for c in sorted(y_true.unique()):
            c_mask = (y_true == c)
            correct = (y_true[c_mask] == y_pred[c_mask]).sum()
            total = c_mask.sum()
            print(f"Class '{c}': Correct: {correct} | Incorrect: {total - correct} | Total: {total}")

    evaluate_model("Decision Tree (Entropy)", y_test, dt_pred)
    evaluate_model("Naive Bayes", y_test, nb_pred)
    
    elapsed_time = time.time() - start_implementation_time
    log_progress(f"Implementation completed in {elapsed_time:.2f} seconds")
    
    return {
        "implementation": "Balanced (2246 per category)",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred,
        "elapsed_time": elapsed_time
    }

###############################################################################
# IMPLEMENTATION 4 - BALANCED DATASET (5000 SAMPLES PER CATEGORY)
###############################################################################
def balanced_implementation_5000(crime_df):
    """
    Implementation using balanced dataset with 5000 samples per category
    """
    start_implementation_time = time.time()
    log_progress("Starting implementation...")
    
    print("\n\n" + "="*80)
    print("IMPLEMENTATION 4 - BALANCED DATASET (5000 SAMPLES PER CATEGORY)")
    print("="*80)
    
    # 1. LOAD AND FILTER THE DATA
    log_progress("Loading and filtering data...")
    start_time = time.time()

    # Create a deep copy to avoid modifying the original dataframe
    df = crime_df.copy(deep=True)

    # Convert 'Date' to datetime and filter for years 2013–2017
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]

    # Filter out low-frequency crime categories
    categories_to_exclude = [
        'PUBLIC INDECENCY', 'NON-CRIMINAL', 'OTHER NARCOTIC VIOLATION', 'HUMAN TRAFFICKING',
        'NON - CRIMINAL', 'RITUALISM', 'NON-CRIMINAL (SUBJECT SPECIFIED)', 'DOMESTIC VIOLENCE'
    ]

    df = df[~df['Primary Type'].isin(categories_to_exclude)]

    # 2. DEFINE COLUMNS AND CHECK CATEGORY COUNTS
    log_progress("Starting feature selection...")
    start_time = time.time()
    possible_columns = [
        'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description',
        'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward',
        'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year'
    ]

    # Filter categorical data and check which categories have at least 5000 samples after cleaning
    sufficient_categories = []
    category_counts = {}

    for category in df['Primary Type'].unique():
        category_df = df[df['Primary Type'] == category]
        
        # Drop any rows with missing values in our possible columns
        category_df = category_df.dropna(subset=[col for col in possible_columns if col in category_df.columns])
        
        # Drop duplicates to ensure clean data
        category_df = category_df.drop_duplicates()
        
        # Count remaining samples
        count = len(category_df)
        category_counts[category] = count
        
        # Check if we have at least 5000 samples
        if count >= 5000:
            sufficient_categories.append(category)

    print(f"Categories with at least 5000 clean samples: {len(sufficient_categories)}")
    for cat in sufficient_categories:
        print(f"{cat}: {category_counts[cat]}")

    # Filter to only include categories with sufficient samples
    df = df[df['Primary Type'].isin(sufficient_categories)]

    # Sample exactly 5000 from each category
    samples_per_category = 5000
    print(f"Taking exactly {samples_per_category} samples from each of {len(sufficient_categories)} categories")

    # Create an empty dataframe to store our balanced sample
    balanced_df = pd.DataFrame()

    # Sample equally from each category
    for category in sufficient_categories:
        category_df = df[df['Primary Type'] == category]
        
        # Drop any rows with missing values in our possible columns
        category_df = category_df.dropna(subset=[col for col in possible_columns if col in category_df.columns])
        
        # Drop duplicates to ensure clean data
        category_df = category_df.drop_duplicates()
        
        # Take exactly 5000 samples
        sampled = category_df.sample(n=samples_per_category, random_state=100)
        balanced_df = pd.concat([balanced_df, sampled])

    # Shuffle the final dataframe
    df = balanced_df.sample(frac=1, random_state=100).reset_index(drop=True)

    print("Data shape after balanced sampling:", df.shape)

    # Keep only columns that exist in your dataset
    df = df[[col for col in possible_columns if col in df.columns]]

    # Verify that we have exactly 5000 samples per category
    print("\nVerifying category distribution:")
    category_counts = df['Primary Type'].value_counts()
    print(category_counts)
    print(f"All categories have exactly {samples_per_category} samples: {(category_counts == samples_per_category).all()}")

    print("\nOriginal crime categories distribution:")
    print(df['Primary Type'].value_counts())
    log_progress(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # 3. FEATURE SELECTION
    log_progress("Starting feature selection...")
    start_time = time.time()
    candidate_features = [
        col for col in df.columns
        if col not in ['Primary Type', 'Case Number', 'Date', 'Year']
    ]

    # Convert categorical columns to category codes
    for col in candidate_features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    X_all = df[candidate_features].copy()
    y_all = df['Primary Type'].copy()

    def evaluate_feature_set(features):
        """Train a quick Decision Tree and return accuracy on a 70-30 split."""
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], y_all,
            test_size=0.3,
            random_state=100,
            stratify=y_all
        )
        
        dt_temp = DecisionTreeClassifier(criterion='entropy', random_state=100)
        dt_temp.fit(X_train, y_train)
        y_pred = dt_temp.predict(X_test)
        return accuracy_score(y_test, y_pred)

    current_features = candidate_features[:]

    improved = True
    while improved and len(current_features) > 9:
        improved = False
        best_accuracy = evaluate_feature_set(current_features)
        
        # Check if dropping each feature improves accuracy
        drop_candidates = []
        for f in current_features:
            temp_features = [x for x in current_features if x != f]
            acc = evaluate_feature_set(temp_features)
            drop_candidates.append((f, acc))
        
        # Sort by accuracy descending
        drop_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # If dropping a feature improves accuracy, remove it
        top_feature, top_acc = drop_candidates[0]
        if top_acc > best_accuracy:
            current_features.remove(top_feature)
            improved = True

    print("\nSelected features after approximate backward elimination:")
    print(current_features)
    log_progress(f"Feature selection completed in {time.time() - start_time:.2f} seconds")

    # 4. TRAIN/TEST SPLIT WITH FINAL FEATURE SET
    X = df[current_features]
    y = df['Primary Type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=100,
        stratify=y
    )

    # Check distribution
    print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class distribution:\n", y_train.value_counts())
    print("Test class distribution:\n", y_test.value_counts())

    # 5. TRAIN THE CLASSIFIERS
    log_progress("Training models...")
    start_time = time.time()
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)
    log_progress(f"Model training completed in {time.time() - start_time:.2f} seconds")

    # 6. EVALUATE THE RESULTS
    log_progress("Evaluating models...")
    start_time = time.time()
    def evaluate_model(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{name} RESULTS")
        print(f"Accuracy: {acc*100:.2f}% | Precision: {prec:.3f} | Recall: {rec:.3f}")
        
        # Class-by-class correct classification (compact format)
        for c in sorted(y_true.unique()):
            c_mask = (y_true == c)
            correct = (y_true[c_mask] == y_pred[c_mask]).sum()
            total = c_mask.sum()
            print(f"Class '{c}': Correct: {correct} | Incorrect: {total - correct} | Total: {total}")

    evaluate_model("Decision Tree (Entropy)", y_test, dt_pred)
    evaluate_model("Naive Bayes", y_test, nb_pred)
    
    elapsed_time = time.time() - start_implementation_time
    log_progress(f"Implementation completed in {elapsed_time:.2f} seconds")
    
    return {
        "implementation": "Balanced (5000 per category)",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred,
        "elapsed_time": elapsed_time
    }

###############################################################################
# IMPLEMENTATION 5 - BALANCED DATASET WITH TEMPERATURE (5000 SAMPLES PER CATEGORY)
###############################################################################
def balanced_implementation_5000_temp(crime_df):
    """
    Implementation using balanced dataset with 5000 samples per category plus temperature data
    OPTIMIZED VERSION using pre-computed temperature lookup file
    
    Parameters:
    crime_df (DataFrame): The loaded Chicago crime dataset
    """
    start_implementation_time = time.time()
    log_progress("Starting implementation...")
    
    print("\n\n" + "="*80)
    print("IMPLEMENTATION 5 - BALANCED DATASET WITH TEMPERATURE (5000 SAMPLES PER CATEGORY)")
    print("="*80)
    
    # 1. LOAD TEMPERATURE LOOKUP FILE
    log_progress("Loading pre-computed temperature lookup file...")
    start_time = time.time()
    
    try:
        temp_lookup_df = pd.read_csv("data/temperature_lookup.csv")
        log_progress(f"Temperature lookup file loaded with {len(temp_lookup_df)} entries")
        
        # Convert date column to datetime for merging
        temp_lookup_df['date'] = pd.to_datetime(temp_lookup_df['date']).dt.date
        
        # Calculate mean temperature for missing values
        mean_temp = temp_lookup_df['temperature'].mean()
        log_progress(f"Mean temperature: {mean_temp:.2f}K")
        
    except FileNotFoundError:
        log_progress("ERROR: temperature_lookup.csv file not found. This file should be created first.")
        log_progress("Run the temperature lookup creation script before running this implementation.")
        return None
    
    log_progress(f"Temperature lookup data loaded in {time.time() - start_time:.2f} seconds")
    
    # 2. LOAD AND FILTER THE CRIME DATA
    log_progress("Filtering crime data...")
    start_time = time.time()
    
    # Create a deep copy to avoid modifying the original dataframe
    df = crime_df.copy(deep=True)
    log_progress(f"Raw crime data loaded: {df.shape[0]} records")

    # Convert 'Date' to datetime and filter for years 2013–2017
    log_progress("Converting dates and filtering by year...")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]
    log_progress(f"After year filtering: {df.shape[0]} records")
    
    # Extract date and hour from Date for temperature lookup
    df['date'] = df['Date'].dt.date
    df['hour'] = df['Date'].dt.hour

    # Filter out low-frequency crime categories
    categories_to_exclude = [
        'PUBLIC INDECENCY', 'NON-CRIMINAL', 'OTHER NARCOTIC VIOLATION', 'HUMAN TRAFFICKING',
        'NON - CRIMINAL', 'RITUALISM', 'NON-CRIMINAL (SUBJECT SPECIFIED)', 'DOMESTIC VIOLENCE'
    ]
    df = df[~df['Primary Type'].isin(categories_to_exclude)]
    log_progress(f"After category filtering: {df.shape[0]} records")
    log_progress(f"Crime data loading and filtering completed in {time.time() - start_time:.2f} seconds")
    
    # 3. MERGE CRIME DATA WITH TEMPERATURE DATA
    log_progress("Adding temperature data to crime records using lookup table...")
    start_time = time.time()
    
    # Efficient merge operation instead of row-by-row lookup
    df = pd.merge(
        df,
        temp_lookup_df,
        on=['date', 'hour'],
        how='left'
    )
    
    # Fill missing values with the mean temperature
    missing_temp = df['temperature'].isna().sum()
    if missing_temp > 0:
        log_progress(f"Filling {missing_temp} missing temperature values with mean: {mean_temp:.2f}K")
        df['temperature'] = df['temperature'].fillna(mean_temp)  # Modified to avoid SettingWithCopyWarning
    
    # Rename column to be consistent with previous implementations
    df = df.rename(columns={'temperature': 'Temperature'})  # Modified to avoid SettingWithCopyWarning
    
    log_progress(f"Temperature data added to crime records in {time.time() - start_time:.2f} seconds")
    
    # 4. DEFINE COLUMNS AND CHECK CATEGORY COUNTS
    log_progress("Analyzing category counts for balanced sampling...")
    possible_columns = [
        'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type', 'Description',
        'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward',
        'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Year', 'Temperature'
    ]
    
    # Filter categorical data and check which categories have at least 5000 samples after cleaning
    sufficient_categories = []
    category_counts = {}
    
    for category in df['Primary Type'].unique():
        category_df = df[df['Primary Type'] == category]
        
        # Drop any rows with missing values in our possible columns
        existing_cols = [col for col in possible_columns if col in category_df.columns]
        category_df = category_df.dropna(subset=existing_cols)
        
        # Drop duplicates to ensure clean data
        category_df = category_df.drop_duplicates()
        
        # Count remaining samples
        count = len(category_df)
        category_counts[category] = count
        
        # Check if we have at least 5000 samples
        if count >= 5000:
            sufficient_categories.append(category)
    
    log_progress(f"Categories with at least 5000 clean samples: {len(sufficient_categories)}")
    for cat in sufficient_categories:
        print(f"  {cat}: {category_counts[cat]}")
    
    # 5. CREATE BALANCED DATASET
    log_progress("Creating balanced dataset...")
    start_time = time.time()
    
    # Filter to only include categories with sufficient samples
    df = df[df['Primary Type'].isin(sufficient_categories)]
    
    # Sample exactly 5000 from each category
    samples_per_category = 5000
    log_progress(f"Taking exactly {samples_per_category} samples from each of {len(sufficient_categories)} categories")
    
    # Create an empty dataframe to store our balanced sample
    balanced_df = pd.DataFrame()
    
    # Sample equally from each category
    for category in sufficient_categories:
        log_progress(f"  Sampling {category}...")
        category_df = df[df['Primary Type'] == category]
        
        # Drop any rows with missing values in our possible columns
        existing_cols = [col for col in possible_columns if col in category_df.columns]
        category_df = category_df.dropna(subset=existing_cols)
        
        # Drop duplicates to ensure clean data
        category_df = category_df.drop_duplicates()
        
        # Take exactly 5000 samples
        if len(category_df) >= samples_per_category:
            sampled = category_df.sample(n=samples_per_category, random_state=100)
            balanced_df = pd.concat([balanced_df, sampled])
        else:
            log_progress(f"  Warning: Category {category} has fewer than {samples_per_category} samples after cleaning.")
            balanced_df = pd.concat([balanced_df, category_df])
    
    # Shuffle the final dataframe
    df = balanced_df.sample(frac=1, random_state=100).reset_index(drop=True)
    
    log_progress(f"Balanced dataset created with shape: {df.shape}")
    log_progress(f"Balanced sampling completed in {time.time() - start_time:.2f} seconds")
    
    # Keep only relevant columns for modeling
    model_columns = [col for col in possible_columns if col in df.columns]
    model_columns.extend(['date', 'hour'])  # Add date and hour columns for reference
    df = df[model_columns]
    
    # Verify the category distribution
    log_progress("Verifying category distribution:")
    category_counts = df['Primary Type'].value_counts()
    all_equal = all(count == samples_per_category for count in category_counts)
    log_progress(f"All categories have exactly {samples_per_category} samples: {all_equal}")
    
    # 6. FEATURE PREPARATION
    log_progress("Preparing features for modeling...")
    start_time = time.time()
    
    # Use the same features from Implementation 4 but add Temperature
    # From Implementation 4, we know these features performed well
    selected_features = [
        'Block', 'IUCR', 'Description', 'Location Description', 'Arrest', 
        'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 
        'FBI Code', 'X Coordinate', 'Y Coordinate', 'Temperature'
    ]
    
    # Make sure all selected features exist in our dataset
    selected_features = [f for f in selected_features if f in df.columns]
    log_progress(f"Using features from Implementation 4 plus Temperature: {selected_features}")
    
    # Convert categorical columns to category codes
    for col in selected_features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    
    log_progress(f"Feature preparation completed in {time.time() - start_time:.2f} seconds")
    
    # 7. CALCULATE FEATURE IMPORTANCE
    log_progress("Calculating feature importance with mutual information...")
    start_time = time.time()
    
    X = df[selected_features].copy()
    y = df['Primary Type'].copy()
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=100)
    mi_results = pd.DataFrame({
        'Feature': selected_features,
        'MI Score': mi_scores
    }).sort_values('MI Score', ascending=False)
    
    log_progress("Feature importance by mutual information:")
    print(mi_results)
    
    # Look at Temperature importance
    temp_importance = mi_results[mi_results['Feature'] == 'Temperature']
    if not temp_importance.empty:
        temp_rank = temp_importance.index[0] + 1
        log_progress(f"Temperature ranks {temp_rank} out of {len(selected_features)} features by importance")
        log_progress(f"Temperature MI Score: {temp_importance.iloc[0]['MI Score']:.6f}")
    
    log_progress(f"Feature importance calculation completed in {time.time() - start_time:.2f} seconds")
    
    # 8. TRAIN/TEST SPLIT
    log_progress("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=100,
        stratify=y
    )
    
    log_progress(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 9. TRAIN THE CLASSIFIERS
    log_progress("Training Decision Tree model...")
    start_time = time.time()
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)
    log_progress(f"Decision Tree training completed in {time.time() - start_time:.2f} seconds")
    
    log_progress("Training Naive Bayes model...")
    start_time = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    log_progress(f"Naive Bayes training completed in {time.time() - start_time:.2f} seconds")
    
    # 10. MAKE PREDICTIONS
    log_progress("Making predictions...")
    start_time = time.time()
    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)
    
    # 11. EVALUATE THE RESULTS
    log_progress("Evaluating model performance...")
    
    def evaluate_model(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        log_progress(f"\n{name} RESULTS")
        log_progress(f"Accuracy: {acc*100:.2f}% | Precision: {prec:.3f} | Recall: {rec:.3f}")
        
        # Class-by-class correct classification (compact format)
        for c in sorted(y_true.unique()):
            c_mask = (y_true == c)
            correct = (y_true[c_mask] == y_pred[c_mask]).sum()
            total = c_mask.sum()
            print(f"Class '{c}': Correct: {correct} | Incorrect: {total - correct} | Total: {total}")

    evaluate_model("Decision Tree (Entropy) with Temperature", y_test, dt_pred)
    evaluate_model("Naive Bayes with Temperature", y_test, nb_pred)
    
    # 12. ANALYZE TEMPERATURE EFFECT ON SPECIFIC CRIME TYPES
    log_progress("Analyzing temperature effect on crime types...")
    
    # Combine test set with predictions and temperature for analysis
    test_results = pd.DataFrame({
        'Primary_Type': y_test,
        'DT_Prediction': dt_pred,
        'NB_Prediction': nb_pred,
        'Temperature': X_test['Temperature'] if 'Temperature' in X_test.columns else None
    })
    
    if 'Temperature' in X_test.columns:
        # Calculate accuracy by temperature ranges
        log_progress("Analyzing model performance across temperature ranges:")
        
        # Create temperature bins (e.g., every 5 degrees)
        temp_min = test_results['Temperature'].min()
        temp_max = test_results['Temperature'].max()
        bin_size = 5  # Kelvin
        bins = np.arange(temp_min, temp_max + bin_size, bin_size)
        
        # Create a new copy to avoid SettingWithCopyWarning
        test_results_copy = test_results.copy()
        test_results_copy['Temp_Bin'] = pd.cut(test_results_copy['Temperature'], bins=bins)
        
        # Calculate accuracy by temperature bin - fixed to address DeprecationWarning
        def calculate_accuracy(group_data):
            """Calculate accuracy metrics for a group"""
            dt_correct = (group_data['Primary_Type'] == group_data['DT_Prediction'])
            nb_correct = (group_data['Primary_Type'] == group_data['NB_Prediction'])
            
            return {
                'DT_Accuracy': dt_correct.mean() * 100,
                'NB_Accuracy': nb_correct.mean() * 100,
                'Count': len(group_data)
            }
        
        # Apply the function to each group with include_groups=False to avoid warning
        temp_accuracy = test_results_copy.groupby('Temp_Bin', observed=True).apply(
            calculate_accuracy
        ).reset_index()
        
        log_progress("Accuracy by temperature range:")
        for _, row in temp_accuracy.iterrows():
            bin_range = row['Temp_Bin']
            stats = row[0]  # Access the dictionary with accuracies
            print(f"Temp range {bin_range}: DT: {stats['DT_Accuracy']:.2f}%, NB: {stats['NB_Accuracy']:.2f}%, Count: {stats['Count']}")
        
        # Check if any crime types have significant temperature correlations
        log_progress("Analyzing temperature correlation with prediction accuracy by crime type:")
        
        for crime_type in test_results['Primary_Type'].unique():
            # Create a new DataFrame for this crime type to avoid warnings
            crime_results = test_results[test_results['Primary_Type'] == crime_type].copy()
            
            # Only analyze if we have enough samples
            if len(crime_results) > 50:
                dt_correct = (crime_results['Primary_Type'] == crime_results['DT_Prediction']).astype(float)
                nb_correct = (crime_results['Primary_Type'] == crime_results['NB_Prediction']).astype(float)
                
                # Handle cases where all predictions are correct (no variation)
                if dt_correct.var() == 0 or nb_correct.var() == 0:
                    continue
                    
                # Calculate correlation between temperature and correct predictions
                # Using try/except to handle potential warnings about division by zero
                try:
                    dt_corr = np.corrcoef(crime_results['Temperature'].values, dt_correct.values)[0, 1]
                    nb_corr = np.corrcoef(crime_results['Temperature'].values, nb_correct.values)[0, 1]
                    
                    # Only report if correlation is notable and not NaN
                    if (abs(dt_corr) > 0.1 or abs(nb_corr) > 0.1) and not (np.isnan(dt_corr) or np.isnan(nb_corr)):
                        print(f"Crime type '{crime_type}':")
                        print(f"  DT correlation with temperature: {dt_corr:.4f}")
                        print(f"  NB correlation with temperature: {nb_corr:.4f}")
                except:
                    # If correlation calculation fails, just continue to the next crime type
                    continue
    
    log_progress("Implementation 5 completed!")
    
    elapsed_time = time.time() - start_implementation_time
    log_progress(f"Implementation completed in {elapsed_time:.2f} seconds")
    
    return {
        "implementation": "Balanced (5000/cat) + Temperature",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred,
        "has_temperature": True,
        "test_results": test_results if 'Temperature' in X_test.columns else None,
        "elapsed_time": elapsed_time
    }

###############################################################################
# CREATE VISUALIZATIONS FOR MODEL COMPARISONS
###############################################################################
def create_visualizations(results):
    # Create figures directory if it doesn't exist
    figures_dir = 'data/figures'
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Overall Accuracy Comparison
    plt.figure(figsize=(12, 8))
    
    implementations = [r["implementation"] for r in results]
    dt_accuracies = [r["dt_accuracy"] for r in results]
    nb_accuracies = [r["nb_accuracy"] for r in results]
    
    x = np.arange(len(implementations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, dt_accuracies, width, label='Decision Tree', color='#8884d8')
    rects2 = ax.bar(x + width/2, nb_accuracies, width, label='Naive Bayes', color='#82ca9d')
    
    ax.set_title('Model Accuracy Across Different Implementations', fontsize=18)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_ylim(80, 101)
    ax.set_xticks(x)
    ax.set_xticklabels(implementations, rotation=15, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'model_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 2. Naive Bayes Performance Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data for PUBLIC PEACE VIOLATION accuracy in each implementation
    ppv_accuracies = []
    
    # For implementation 1 (4 categories)
    mask = (results[0]["y_test"] == "PUBLIC_PEACE_VIOLATION")
    correct = (results[0]["y_test"][mask] == results[0]["nb_pred"][mask]).sum()
    total = mask.sum()
    ppv_accuracies.append((correct / total) * 100)
    
    # For implementation 2 (unbalanced original)
    if "PUBLIC PEACE VIOLATION" in results[1]["y_test"].unique():
        mask = (results[1]["y_test"] == "PUBLIC PEACE VIOLATION")
        correct = (results[1]["y_test"][mask] == results[1]["nb_pred"][mask]).sum()
        total = mask.sum()
        ppv_accuracies.append((correct / total) * 100 if total > 0 else 0)
    else:
        ppv_accuracies.append(0)
    
    # For implementation 3 (2246 balanced)
    if "PUBLIC PEACE VIOLATION" in results[2]["y_test"].unique():
        mask = (results[2]["y_test"] == "PUBLIC PEACE VIOLATION")
        correct = (results[2]["y_test"][mask] == results[2]["nb_pred"][mask]).sum()
        total = mask.sum()
        ppv_accuracies.append((correct / total) * 100 if total > 0 else 0)
    else:
        ppv_accuracies.append(0)
    
    # For implementation 4 (5000 balanced)
    if "PUBLIC PEACE VIOLATION" in results[3]["y_test"].unique():
        mask = (results[3]["y_test"] == "PUBLIC PEACE VIOLATION")
        correct = (results[3]["y_test"][mask] == results[3]["nb_pred"][mask]).sum()
        total = mask.sum()
        ppv_accuracies.append((correct / total) * 100 if total > 0 else 0)
    else:
        ppv_accuracies.append(0)
        
    # For implementation 5 (5000 balanced with temperature)
    if len(results) > 4 and "PUBLIC PEACE VIOLATION" in results[4]["y_test"].unique():
        mask = (results[4]["y_test"] == "PUBLIC PEACE VIOLATION")
        correct = (results[4]["y_test"][mask] == results[4]["nb_pred"][mask]).sum()
        total = mask.sum()
        ppv_accuracies.append((correct / total) * 100 if total > 0 else 0)
    else:
        ppv_accuracies.append(0)
    
    # Create the bar chart
    names = ["Original (4 cat)", "Original Categories", "Balanced (2246/cat)", "Balanced (5000/cat)", "Balanced + Temp"]
    
    ax.bar(names, ppv_accuracies, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'])
    ax.set_title('Naive Bayes Performance on PUBLIC PEACE VIOLATION Category', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_ylim(0, 105)
    
    # Add accuracy values on top of bars
    for i, v in enumerate(ppv_accuracies):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'naive_bayes_public_peace_violation.png'), dpi=300, bbox_inches='tight')
    
    # 3. Dataset Distribution Analysis
    # Original data distribution
    original_distribution = {
        'THEFT': 21.19,
        'BATTERY': 18.23,
        'CRIMINAL DAMAGE': 11.38,
        'NARCOTICS': 9.20,
        'ASSAULT': 6.65,
        'OTHER OFFENSE': 6.22,
        'BURGLARY': 5.32,
        'MOTOR VEHICLE THEFT': 5.07,
        'DECEPTIVE PRACTICE': 4.55,
        'ROBBERY': 3.76,
        'Other Categories': 8.43
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        original_distribution.values(), 
        labels=original_distribution.keys(),
        autopct='%1.1f%%',
        textprops={'fontsize': 9},
        colors=plt.cm.tab20.colors
    )
    
    # Make the percentage labels more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    ax.set_title('Original Chicago Crime Dataset Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'original_crime_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 4. Dataset Size Comparison
    dataset_sizes = [
        (results[0]["implementation"], 5340, 4),
        (results[1]["implementation"], 11955, 26),
        (results[2]["implementation"], 42674, 19),
        (results[3]["implementation"], 85000, 17)
    ]
    
    # Add implementation 5 if available
    if len(results) > 4:
        # Estimate the number of samples and categories from implementation 5
        # This assumes a structure similar to implementation 4
        dataset_sizes.append((results[4]["implementation"], 85000, 17))
    
    names = [d[0] for d in dataset_sizes]
    sample_sizes = [d[1] for d in dataset_sizes]
    category_counts = [d[2] for d in dataset_sizes]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Sample size bars (main axis)
    ax1.bar(names, sample_sizes, color='#8884d8', alpha=0.7)
    ax1.set_ylabel('Total Samples', fontsize=14)
    ax1.set_title('Dataset Sizes and Category Counts Across Implementations', fontsize=16)
    
    # Add second y-axis for category counts
    ax2 = ax1.twinx()
    ax2.plot(names, category_counts, 'o-', color='#82ca9d', linewidth=3, markersize=10)
    ax2.set_ylabel('Number of Categories', fontsize=14)
    
    # Add values on bars and points
    for i, v in enumerate(sample_sizes):
        ax1.text(i, v + 2000, f"{v:,}", ha='center', fontsize=10)
    
    for i, v in enumerate(category_counts):
        ax2.text(i, v + 0.5, str(v), ha='center', fontsize=10, color='#82ca9d')
    
    ax1.set_ylim(0, max(sample_sizes) * 1)
    ax1.set_ylim(0, max(sample_sizes) * 1.1)
    ax2.set_ylim(0, max(category_counts) * 1.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'dataset_size_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 5. Feature Importance Analysis
    feature_sets = [
        (results[0]["implementation"], ['Block', 'IUCR', 'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate']),
        (results[1]["implementation"], ['Block', 'IUCR', 'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward',]),
        (results[1]["implementation"], ['Block', 'IUCR', 'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate']),
        (results[2]["implementation"], ['IUCR', 'Description', 'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 'X Coordinate', 'Y Coordinate']),
        (results[3]["implementation"], ['Block', 'IUCR', 'Description', 'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate'])
    ]
    
    # Add implementation 5 if available
    if len(results) > 4:
        # Assuming the feature set is similar to implementation 4 but with Temperature added
        impl5_features = ['Block', 'IUCR', 'Description', 'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Temperature']
        feature_sets.append((results[4]["implementation"], impl5_features))
    
    # Count occurrences of each feature
    all_features = []
    for _, features in feature_sets:
        all_features.extend(features)
    
    feature_count = {}
    for feature in set(all_features):
        feature_count[feature] = sum(1 for _, features in feature_sets if feature in features)
    
    # Sort by count descending
    feature_count = {k: v for k, v in sorted(feature_count.items(), key=lambda item: item[1], reverse=True)}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = list(feature_count.keys())
    counts = list(feature_count.values())
    
    y_pos = np.arange(len(features))
    
    # Color Temperature differently if present
    colors = plt.cm.tab20c.colors[:len(features)]
    if 'Temperature' in features:
        temp_idx = features.index('Temperature')
        # Create a custom color list
        colors = [plt.cm.tab20c.colors[i % len(plt.cm.tab20c.colors)] for i in range(len(features))]
        colors[temp_idx] = '#ff7f0e'  # Highlight Temperature in orange
    
    ax.barh(y_pos, counts, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=12)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Occurrences Across Implementations', fontsize=14)
    ax.set_title('Feature Selection Across Different Implementations', fontsize=16)
    
    # Add count values on bars
    for i, v in enumerate(counts):
        ax.text(v + 0.1, i, str(v), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_selection_analysis.png'), dpi=300, bbox_inches='tight')
    
    # 6. NEW: Temperature Effect Analysis (if implementation 5 is available)
    if len(results) > 4 and "has_temperature" in results[4] and results[4]["has_temperature"]:
        test_results = results[4].get("test_results")
        
        if test_results is not None and 'Temperature' in test_results.columns:
            # Create a temperature vs. accuracy plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Bin the temperatures
            temp_min = test_results['Temperature'].min()
            temp_max = test_results['Temperature'].max()
            bin_size = 5  # Kelvin
            bins = np.arange(np.floor(temp_min), np.ceil(temp_max) + bin_size, bin_size)
            
            test_results['Temp_Bin'] = pd.cut(test_results['Temperature'], bins=bins)
            
            # Calculate accuracy by temperature bin
            temp_dt_accuracies = []
            temp_nb_accuracies = []
            temp_bin_centers = []
            temp_sample_counts = []
            
            for temp_bin in sorted(test_results['Temp_Bin'].unique()):
                bin_data = test_results[test_results['Temp_Bin'] == temp_bin]
                
                if len(bin_data) > 20:  # Only include bins with sufficient samples
                    dt_acc = (bin_data['Primary_Type'] == bin_data['DT_Prediction']).mean() * 100
                    nb_acc = (bin_data['Primary_Type'] == bin_data['NB_Prediction']).mean() * 100
                    
                    temp_dt_accuracies.append(dt_acc)
                    temp_nb_accuracies.append(nb_acc)
                    # Use the midpoint of the bin for plotting
                    temp_bin_centers.append((temp_bin.left + temp_bin.right) / 2)
                    temp_sample_counts.append(len(bin_data))
            
            # Plot accuracies by temperature
            ax.plot(temp_bin_centers, temp_dt_accuracies, 'o-', color='#8884d8', linewidth=2, markersize=8, label='Decision Tree')
            ax.plot(temp_bin_centers, temp_nb_accuracies, 'o-', color='#82ca9d', linewidth=2, markersize=8, label='Naive Bayes')
            
            # Plot sample count as bar chart on secondary axis
            ax2 = ax.twinx()
            ax2.bar(temp_bin_centers, temp_sample_counts, alpha=0.3, color='gray', width=3)
            ax2.set_ylabel('Number of Samples', fontsize=14)
            
            # Add reference line at 100% for Decision Tree
            ax.axhline(y=100, color='#8884d8', linestyle='--', alpha=0.5)
            
            ax.set_title('Model Accuracy by Temperature', fontsize=18)
            ax.set_xlabel('Temperature (Kelvin)', fontsize=14)
            ax.set_ylabel('Accuracy (%)', fontsize=14)
            ax.set_ylim(70, 102)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='lower left', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'temperature_effect_analysis.png'), dpi=300, bbox_inches='tight')
            
            # 7. NEW: Crime Type-Specific Temperature Effect
            # Select a few representative crime types
            major_crime_types = test_results['Primary_Type'].value_counts().nlargest(5).index.tolist()
            
            fig, axes = plt.subplots(len(major_crime_types), 1, figsize=(12, 4*len(major_crime_types)), sharex=True)
            
            for i, crime_type in enumerate(major_crime_types):
                ax = axes[i]
                crime_data = test_results[test_results['Primary_Type'] == crime_type]
                
                # Bin the temperatures
                crime_data['Temp_Bin'] = pd.cut(crime_data['Temperature'], bins=bins)
                
                # Calculate accuracy by temperature bin for this crime type
                type_dt_accuracies = []
                type_nb_accuracies = []
                type_bin_centers = []
                type_sample_counts = []
                
                for temp_bin in sorted(crime_data['Temp_Bin'].unique()):
                    bin_data = crime_data[crime_data['Temp_Bin'] == temp_bin]
                    
                    if len(bin_data) > 10:  # Only include bins with sufficient samples
                        dt_acc = (bin_data['Primary_Type'] == bin_data['DT_Prediction']).mean() * 100
                        nb_acc = (bin_data['Primary_Type'] == bin_data['NB_Prediction']).mean() * 100
                        
                        type_dt_accuracies.append(dt_acc)
                        type_nb_accuracies.append(nb_acc)
                        type_bin_centers.append((temp_bin.left + temp_bin.right) / 2)
                        type_sample_counts.append(len(bin_data))
                
                # Plot accuracies by temperature for this crime type
                ax.plot(type_bin_centers, type_dt_accuracies, 'o-', color='#8884d8', linewidth=2, markersize=8, label='Decision Tree')
                ax.plot(type_bin_centers, type_nb_accuracies, 'o-', color='#82ca9d', linewidth=2, markersize=8, label='Naive Bayes')
                
                # Plot sample count as bar chart in background
                ax2 = ax.twinx()
                ax2.bar(type_bin_centers, type_sample_counts, alpha=0.2, color='gray', width=3)
                ax2.set_ylabel('Samples', fontsize=12)
                
                ax.set_title(f'Accuracy by Temperature: {crime_type}', fontsize=14)
                ax.set_ylabel('Accuracy (%)', fontsize=12)
                ax.set_ylim(0, 105)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                if i == 0:
                    ax.legend(loc='lower left', fontsize=10)
            
            # Set xlabel only on bottom subplot
            axes[-1].set_xlabel('Temperature (Kelvin)', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'temperature_effect_by_crime_type.png'), dpi=300, bbox_inches='tight')
            
            # 8. NEW: Distribution of crimes by temperature
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot histogram of crimes by temperature
            for crime_type in major_crime_types:
                crime_data = test_results[test_results['Primary_Type'] == crime_type]
                ax.hist(crime_data['Temperature'], bins=20, alpha=0.5, label=crime_type)
            
            ax.set_title('Distribution of Crime Types by Temperature', fontsize=18)
            ax.set_xlabel('Temperature (Kelvin)', fontsize=14)
            ax.set_ylabel('Number of Incidents', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'crime_distribution_by_temperature.png'), dpi=300, bbox_inches='tight')

###############################################################################
# RUN ALL IMPLEMENTATIONS AND COMPARE RESULTS
###############################################################################
def run_all_and_compare():
    start_total_time = time.time()
    log_progress("Starting all implementations...")
    
    # Load the Chicago crime data once
    log_progress("Loading Chicago crime data...")
    data_load_start = time.time()
    crime_df = pd.read_csv("data/chicago_crime.csv", low_memory=False)
    log_progress(f"Chicago crime data loaded in {time.time() - data_load_start:.2f} seconds")
    
    # Run all implementations, passing the loaded dataframe to each
    results = []
    results.append(original_implementation(crime_df))
    results.append(original_categories_implementation(crime_df))
    results.append(balanced_implementation_2246(crime_df))
    results.append(balanced_implementation_5000(crime_df))
    results.append(balanced_implementation_5000_temp(crime_df))
    
    # Create visualizations
    log_progress("Creating visualizations...")
    create_visualizations(results)
    
    # Print comparison
    print("\n\n" + "="*80)
    print("COMPARISON OF ALL IMPLEMENTATIONS")
    print("="*80)
    
    print(f"{'Implementation':<35} | {'Decision Tree Accuracy':<25} | {'Naive Bayes Accuracy':<25}")
    print("-" * 90)
    
    for result in results:
        print(f"{result['implementation']:<35} | {result['dt_accuracy']:<25.2f} | {result['nb_accuracy']:<25.2f}")

    # Calculate total elapsed time
    total_elapsed_time = time.time() - start_total_time
    log_progress(f"All implementations completed in {total_elapsed_time:.2f} seconds")
    
    # Print detailed timing breakdown
    hours = int(total_elapsed_time // 3600)
    minutes = int((total_elapsed_time % 3600) // 60)
    seconds = total_elapsed_time % 60
    
    if hours > 0:
        log_progress(f"Total runtime: {hours}h {minutes}m {seconds:.2f}s")
    elif minutes > 0:
        log_progress(f"Total runtime: {minutes}m {seconds:.2f}s")
    else:
        log_progress(f"Total runtime: {seconds:.2f}s")

# Run all implementations and compare results
if __name__ == "__main__":
    run_all_and_compare()