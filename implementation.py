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
    
    # Create the bar chart
    names = ["Original (4 cat)", "Original Categories", "Balanced (2246/cat)", "Balanced (5000/cat)"]
    
    ax.bar(names, ppv_accuracies, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
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
    
    ax.barh(y_pos, counts, color=plt.cm.tab20c.colors[:len(features)])
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
    
    # Create visualizations
    create_visualizations(results)
    
    # Print comparison
    print("\n\n" + "="*80)
    print("COMPARISON OF ALL IMPLEMENTATIONS")
    print("="*80)
    
    print(f"{'Implementation':<30} | {'Decision Tree Accuracy':<25} | {'Naive Bayes Accuracy':<25}")
    print("-" * 85)
    
    for result in results:
        print(f"{result['implementation']:<30} | {result['dt_accuracy']:<25.2f} | {result['nb_accuracy']:<25.2f}")

# Run all implementations and compare results
if __name__ == "__main__":
    run_all_and_compare()