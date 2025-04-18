import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif

###############################################################################
# ORIGINAL IMPLEMENTATION - 4 UMBRELLA CATEGORIES
###############################################################################
def original_implementation():
    """
    Original implementation using the 4 umbrella categories as described in the paper
    """
    print("\n\n" + "="*80)
    print("ORIGINAL IMPLEMENTATION - 4 UMBRELLA CATEGORIES")
    print("="*80)
    
    # 1. LOAD AND FILTER THE DATA
    df = pd.read_csv("chicago_crime.csv", low_memory=False)

    # Convert 'Date' to datetime and filter for years 2013–2017
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]

    # Randomly sample 12,109 records (matching the paper's final record count)
    df = df.sample(n=12109, random_state=100).copy()

    print("Data shape after year filter and sampling:", df.shape)

    # 2. REDUCE TO RELEVANT COLUMNS (SIMILAR TO THE PAPER'S 18 INITIAL FEATURES)
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
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)

    # 7. EVALUATE THE RESULTS
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
    
    return {
        "implementation": "Original (4 categories)",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred
    }

###############################################################################
# IMPLEMENTATION 2 - ORIGINAL CATEGORIES (UNBALANCED)
###############################################################################
def original_categories_implementation():
    """
    Implementation using original crime categories without resampling
    """
    print("\n\n" + "="*80)
    print("IMPLEMENTATION 2 - ORIGINAL CATEGORIES (UNBALANCED)")
    print("="*80)
    
    # 1. LOAD AND FILTER THE DATA
    df = pd.read_csv("chicago_crime.csv", low_memory=False)

    # Convert 'Date' to datetime and filter for years 2013–2017
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]

    # Randomly sample 12,109 records
    df = df.sample(n=12109, random_state=100).copy()

    print("Data shape after year filter and sampling:", df.shape)

    # 2. REDUCE TO RELEVANT COLUMNS
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
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)

    # 7. EVALUATE THE RESULTS
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
    
    return {
        "implementation": "Original Categories (Unbalanced)",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred
    }

###############################################################################
# IMPLEMENTATION 3 - BALANCED DATASET (2246 SAMPLES PER CATEGORY)
###############################################################################
def balanced_implementation_2246():
    """
    Implementation using balanced dataset with 2246 samples per category
    """
    print("\n\n" + "="*80)
    print("IMPLEMENTATION 3 - BALANCED DATASET (2246 SAMPLES PER CATEGORY)")
    print("="*80)
    
    # 1. LOAD AND FILTER THE DATA
    df = pd.read_csv("chicago_crime.csv", low_memory=False)

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

    # 3. FEATURE SELECTION
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
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)

    # 6. EVALUATE THE RESULTS
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
    
    return {
        "implementation": "Balanced (2246 per category)",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred
    }

###############################################################################
# IMPLEMENTATION 4 - BALANCED DATASET (5000 SAMPLES PER CATEGORY)
###############################################################################
def balanced_implementation_5000():
    """
    Implementation using balanced dataset with 5000 samples per category
    """
    print("\n\n" + "="*80)
    print("IMPLEMENTATION 4 - BALANCED DATASET (5000 SAMPLES PER CATEGORY)")
    print("="*80)
    
    # 1. LOAD AND FILTER THE DATA
    df = pd.read_csv("chicago_crime.csv", low_memory=False)

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

    # 3. FEATURE SELECTION
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
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)

    # 6. EVALUATE THE RESULTS
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
    
    return {
        "implementation": "Balanced (5000 per category)",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred
    }

###############################################################################
# IMPLEMENTATION 5 - BALANCED DATASET WITH TEMPERATURE (5000 SAMPLES PER CATEGORY)
###############################################################################
def balanced_implementation_5000_temp():
    """
    Implementation using balanced dataset with 5000 samples per category plus temperature data
    """
    print("\n\n" + "="*80)
    print("IMPLEMENTATION 5 - BALANCED DATASET WITH TEMPERATURE (5000 SAMPLES PER CATEGORY)")
    print("="*80)
    
    # 1. LOAD TEMPERATURE DATA
    print("Loading temperature data...")
    temp_df = pd.read_csv("temperature.csv")
    
    # Convert datetime column to datetime type and create hour column
    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'])
    temp_df['date'] = temp_df['datetime'].dt.date
    temp_df['hour'] = temp_df['datetime'].dt.hour
    
    # Extract Chicago temperature (ensure column exists)
    if 'Chicago' in temp_df.columns:
        temp_chicago = temp_df[['datetime', 'date', 'hour', 'Chicago']].copy()
        print("Chicago temperature data shape:", temp_chicago.shape)
    else:
        print("Warning: 'Chicago' column not found in temperature data. Using nearest available city.")
        # Use a nearby city if Chicago isn't available
        nearby_cities = ['Indianapolis', 'Detroit', 'Minneapolis', 'Saint Louis']
        for city in nearby_cities:
            if city in temp_df.columns:
                temp_chicago = temp_df[['datetime', 'date', 'hour', city]].copy()
                temp_chicago.rename(columns={city: 'Chicago'}, inplace=True)
                print(f"Using {city} temperature data as proxy. Shape:", temp_chicago.shape)
                break
        else:
            # If no nearby city, use the first temperature column available
            temp_col = [col for col in temp_df.columns if col not in ['datetime', 'date', 'hour']][0]
            temp_chicago = temp_df[['datetime', 'date', 'hour', temp_col]].copy()
            temp_chicago.rename(columns={temp_col: 'Chicago'}, inplace=True)
            print(f"Using {temp_col} temperature data as proxy. Shape:", temp_chicago.shape)
    
    # 2. LOAD AND FILTER THE CRIME DATA
    print("Loading and filtering crime data...")
    df = pd.read_csv("chicago_crime.csv", low_memory=False)

    # Convert 'Date' to datetime and filter for years 2013–2017
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
    df['Year'] = df['Date'].dt.year
    df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]
    
    # Extract date and hour from Date
    df['date'] = df['Date'].dt.date
    df['hour'] = df['Date'].dt.hour

    # Filter out low-frequency crime categories
    categories_to_exclude = [
        'PUBLIC INDECENCY', 'NON-CRIMINAL', 'OTHER NARCOTIC VIOLATION', 'HUMAN TRAFFICKING',
        'NON - CRIMINAL', 'RITUALISM', 'NON-CRIMINAL (SUBJECT SPECIFIED)', 'DOMESTIC VIOLENCE'
    ]

    df = df[~df['Primary Type'].isin(categories_to_exclude)]

    # 3. MERGE CRIME DATA WITH TEMPERATURE DATA
    print("Merging crime data with temperature data...")
    
    # Create a mapping function to find the nearest hour's temperature
    def find_nearest_temp(row):
        crime_date = row['date']
        crime_hour = row['hour']
        
        # Look for exact match
        exact_match = temp_chicago[(temp_chicago['date'] == crime_date) & 
                                  (temp_chicago['hour'] == crime_hour)]
        
        if not exact_match.empty:
            return exact_match.iloc[0]['Chicago']
        
        # If no exact match, find the closest hour on the same day
        same_day = temp_chicago[temp_chicago['date'] == crime_date]
        if not same_day.empty:
            closest_hour_idx = (same_day['hour'] - crime_hour).abs().idxmin()
            return same_day.loc[closest_hour_idx, 'Chicago']
        
        # If no data for that day, use the closest day's same hour
        all_dates = temp_chicago['datetime'].dt.date.unique()
        if len(all_dates) > 0:
            date_diffs = np.abs([(date - crime_date).days for date in all_dates])
            closest_date_idx = np.argmin(date_diffs)
            closest_date = all_dates[closest_date_idx]
            
            closest_day = temp_chicago[temp_chicago['date'] == closest_date]
            if not closest_day.empty:
                closest_hour_idx = (closest_day['hour'] - crime_hour).abs().idxmin()
                return closest_day.loc[closest_hour_idx, 'Chicago']
        
        # If all else fails, return the mean temperature
        return temp_chicago['Chicago'].mean()
    
    # Apply the mapping to get temperature for each crime (this may take a while)
    print("Finding temperature for each crime incident...")
    # For efficiency, create a lookup dictionary first
    temp_lookup = {}
    for _, row in temp_chicago.iterrows():
        key = (row['date'], row['hour'])
        temp_lookup[key] = row['Chicago']
    
    # Function to get temperature using the lookup dictionary
    def get_temp_from_lookup(row):
        key = (row['date'], row['hour'])
        if key in temp_lookup:
            return temp_lookup[key]
        
        # If exact hour not found, try to find closest hour on same day
        same_day_hours = [k[1] for k in temp_lookup.keys() if k[0] == row['date']]
        if same_day_hours:
            closest_hour = min(same_day_hours, key=lambda x: abs(x - row['hour']))
            return temp_lookup[(row['date'], closest_hour)]
        
        # If no data for that day, return the mean
        return np.mean(list(temp_lookup.values()))
    
    # Apply the temperature lookup
    df['Temperature'] = df.apply(get_temp_from_lookup, axis=1)
    
    # Check for missing temperature values and fill with the mean
    missing_temp = df['Temperature'].isna().sum()
    if missing_temp > 0:
        print(f"Found {missing_temp} crime records with missing temperature data. Filling with mean.")
        df['Temperature'].fillna(df['Temperature'].mean(), inplace=True)
    
    # 4. DEFINE COLUMNS AND CHECK CATEGORY COUNTS
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

    print(f"Categories with at least 5000 clean samples (including temperature): {len(sufficient_categories)}")
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
        existing_cols = [col for col in possible_columns if col in category_df.columns]
        category_df = category_df.dropna(subset=existing_cols)
        
        # Drop duplicates to ensure clean data
        category_df = category_df.drop_duplicates()
        
        # Take exactly 5000 samples
        if len(category_df) >= samples_per_category:
            sampled = category_df.sample(n=samples_per_category, random_state=100)
            balanced_df = pd.concat([balanced_df, sampled])
        else:
            print(f"Warning: Category {category} has fewer than {samples_per_category} samples after cleaning.")
            balanced_df = pd.concat([balanced_df, category_df])

    # Shuffle the final dataframe
    df = balanced_df.sample(frac=1, random_state=100).reset_index(drop=True)

    print("Data shape after balanced sampling with temperature:", df.shape)

    # Keep only relevant columns for modeling
    model_columns = [col for col in possible_columns if col in df.columns]
    model_columns.extend(['date', 'hour'])  # Add date and hour columns for reference
    df = df[model_columns]

    # Verify the category distribution
    print("\nVerifying category distribution:")
    category_counts = df['Primary Type'].value_counts()
    print(category_counts)
    all_equal = all(count == samples_per_category for count in category_counts)
    print(f"All categories have exactly {samples_per_category} samples: {all_equal}")

    # 5. FEATURE SELECTION
    candidate_features = [
        col for col in df.columns
        if col not in ['Primary Type', 'Case Number', 'Date', 'Year', 'date', 'hour']
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

    # Keep the same feature selection process, but don't remove Temperature
    current_features = candidate_features[:]

    improved = True
    while improved and len(current_features) > 9:
        improved = False
        best_accuracy = evaluate_feature_set(current_features)
        
        # Check if dropping each feature improves accuracy
        drop_candidates = []
        for f in current_features:
            # Don't consider removing Temperature feature
            if f == 'Temperature':
                continue
                
            temp_features = [x for x in current_features if x != f]
            acc = evaluate_feature_set(temp_features)
            drop_candidates.append((f, acc))
        
        # Sort by accuracy descending
        drop_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # If dropping a feature improves accuracy, remove it
        if drop_candidates:
            top_feature, top_acc = drop_candidates[0]
            if top_acc > best_accuracy:
                current_features.remove(top_feature)
                improved = True

    print("\nSelected features after approximate backward elimination (keeping Temperature):")
    print(current_features)
    
    # Calculate feature importance with mutual information
    if 'Temperature' in current_features:
        X_temp = df[current_features].copy()
        mi_scores = mutual_info_classif(X_temp, y_all, random_state=100)
        mi_results = pd.DataFrame({
            'Feature': current_features,
            'MI Score': mi_scores
        }).sort_values('MI Score', ascending=False)
        
        print("\nFeature importance by mutual information:")
        print(mi_results)
        
        # Specifically look at Temperature importance
        temp_importance = mi_results[mi_results['Feature'] == 'Temperature']
        if not temp_importance.empty:
            temp_rank = temp_importance.index[0] + 1
            print(f"\nTemperature ranks {temp_rank} out of {len(current_features)} features by importance")
            print(f"Temperature MI Score: {temp_importance.iloc[0]['MI Score']:.6f}")

    # 6. TRAIN/TEST SPLIT WITH FINAL FEATURE SET
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

    # 7. TRAIN THE CLASSIFIERS
    dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    nb_pred = nb.predict(X_test)

    # 8. EVALUATE THE RESULTS
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

    evaluate_model("Decision Tree (Entropy) with Temperature", y_test, dt_pred)
    evaluate_model("Naive Bayes with Temperature", y_test, nb_pred)
    
    # 9. ANALYZE TEMPERATURE EFFECT ON SPECIFIC CRIME TYPES
    # Combine test set with predictions and temperature for analysis
    test_results = pd.DataFrame({
        'Primary_Type': y_test,
        'DT_Prediction': dt_pred,
        'NB_Prediction': nb_pred,
        'Temperature': X_test['Temperature'] if 'Temperature' in X_test.columns else None
    })
    
    if 'Temperature' in X_test.columns:
        # Calculate accuracy by temperature ranges
        print("\nAnalyzing model performance across temperature ranges:")
        
        # Create temperature bins (e.g., every 5 degrees)
        temp_min = test_results['Temperature'].min()
        temp_max = test_results['Temperature'].max()
        bin_size = 5  # Kelvin
        bins = np.arange(temp_min, temp_max + bin_size, bin_size)
        
        test_results['Temp_Bin'] = pd.cut(test_results['Temperature'], bins=bins)
        
        # Calculate accuracy by temperature bin
        temp_accuracy = test_results.groupby('Temp_Bin').apply(
            lambda x: {
                'DT_Accuracy': accuracy_score(x['Primary_Type'], x['DT_Prediction']) * 100,
                'NB_Accuracy': accuracy_score(x['Primary_Type'], x['NB_Prediction']) * 100,
                'Count': len(x)
            }
        ).reset_index()
        
        print("\nAccuracy by temperature range:")
        for _, row in temp_accuracy.iterrows():
            bin_range = row['Temp_Bin']
            stats = row[0]  # Access the dictionary with accuracies
            print(f"Temp range {bin_range}: DT: {stats['DT_Accuracy']:.2f}%, NB: {stats['NB_Accuracy']:.2f}%, Count: {stats['Count']}")
        
        # Check if any crime types have significant temperature correlations
        print("\nAnalyzing temperature correlation with prediction accuracy by crime type:")
        
        for crime_type in test_results['Primary_Type'].unique():
            crime_results = test_results[test_results['Primary_Type'] == crime_type]
            
            # Only analyze if we have enough samples
            if len(crime_results) > 50:
                dt_correct = (crime_results['Primary_Type'] == crime_results['DT_Prediction'])
                nb_correct = (crime_results['Primary_Type'] == crime_results['NB_Prediction'])
                
                # Calculate correlation between temperature and correct predictions
                dt_corr = np.corrcoef(crime_results['Temperature'], dt_correct)[0, 1]
                nb_corr = np.corrcoef(crime_results['Temperature'], nb_correct)[0, 1]
                
                # Only report if correlation is notable
                if abs(dt_corr) > 0.1 or abs(nb_corr) > 0.1:
                    print(f"Crime type '{crime_type}':")
                    print(f"  DT correlation with temperature: {dt_corr:.4f}")
                    print(f"  NB correlation with temperature: {nb_corr:.4f}")
    
    return {
        "implementation": "Balanced (5000/cat) + Temperature",
        "dt_accuracy": accuracy_score(y_test, dt_pred) * 100,
        "nb_accuracy": accuracy_score(y_test, nb_pred) * 100,
        "y_test": y_test,
        "dt_pred": dt_pred,
        "nb_pred": nb_pred,
        "has_temperature": True,
        "test_results": test_results if 'Temperature' in X_test.columns else None
    }

###############################################################################
# CREATE VISUALIZATIONS FOR MODEL COMPARISONS
###############################################################################
def create_visualizations(results):
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
    plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    
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
    plt.savefig('naive_bayes_public_peace_violation.png', dpi=300, bbox_inches='tight')
    
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
    plt.savefig('original_crime_distribution.png', dpi=300, bbox_inches='tight')
    
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
    plt.savefig('dataset_size_comparison.png', dpi=300, bbox_inches='tight')
    
    # 5. Feature Importance Analysis
    feature_sets = [
        (results[0]["implementation"], ['Block', 'IUCR', 'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward', 'Community Area', 'FBI Code', 'X Coordinate', 'Y Coordinate']),
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
    plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
    
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
            plt.savefig('temperature_effect_analysis.png', dpi=300, bbox_inches='tight')
            
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
            plt.savefig('temperature_effect_by_crime_type.png', dpi=300, bbox_inches='tight')
            
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
            plt.savefig('crime_distribution_by_temperature.png', dpi=300, bbox_inches='tight')

###############################################################################
# RUN ALL IMPLEMENTATIONS AND COMPARE RESULTS
###############################################################################
def run_all_and_compare():
    # Run all implementations
    results = []
    results.append(original_implementation())
    results.append(original_categories_implementation())
    results.append(balanced_implementation_2246())
    results.append(balanced_implementation_5000())
    results.append(balanced_implementation_5000_temp())
    
    # Create visualizations
    create_visualizations(results)
    
    # Print comparison
    print("\n\n" + "="*80)
    print("COMPARISON OF ALL IMPLEMENTATIONS")
    print("="*80)
    
    print(f"{'Implementation':<35} | {'Decision Tree Accuracy':<25} | {'Naive Bayes Accuracy':<25}")
    print("-" * 90)
    
    for result in results:
        print(f"{result['implementation']:<35} | {result['dt_accuracy']:<25.2f} | {result['nb_accuracy']:<25.2f}")

# Run all implementations and compare results
if __name__ == "__main__":
    run_all_and_compare()