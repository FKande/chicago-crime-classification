import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif

###############################################################################
# 1. LOAD AND FILTER THE DATA
###############################################################################
df = pd.read_csv("chicago_crime.csv", low_memory=False)

# Convert 'Date' to datetime and filter for years 2013–2017
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])  # remove rows where Date failed to parse
df['Year'] = df['Date'].dt.year
df = df[(df['Year'] >= 2013) & (df['Year'] <= 2017)]

# Randomly sample 12,109 records (matching the paper's final record count)
df = df.sample(n=12109, random_state=100).copy()

print("Data shape after year filter and sampling:", df.shape)

###############################################################################
# 2. REDUCE TO RELEVANT COLUMNS (SIMILAR TO THE PAPER'S 18 INITIAL FEATURES)
###############################################################################
# The paper mentioned 18 attributes originally (e.g., ID, Case Number, Date, 
# Block, IUCR, Primary Type, Description, Location, Arrest, Domestic, Beat, 
# District, Ward, CommunityArea, FBI Code, X Coordinate, Y Coordinate, Year, ...).
# We'll pick a representative subset that aligns with typical Chicago data columns.

possible_columns = [
    'Case Number',
    'Date',
    'Block',
    'IUCR',
    'Primary Type',
    'Description',
    'Location Description',
    'Arrest',
    'Domestic',
    'Beat',
    'District',
    'Ward',
    'Community Area',
    'FBI Code',
    'X Coordinate',
    'Y Coordinate',
    'Year'
]

# Keep only columns that exist in your dataset
df = df[[col for col in possible_columns if col in df.columns]]

# Drop duplicates and remove any missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

###############################################################################
# 3. RECLASSIFY CRIMES INTO THE 4 CATEGORIES EXACTLY AS THE PAPER STATES
###############################################################################
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
        'NARCOTICS',
        'OTHER NARCOTIC VIOLATION',
        'PROSTITUTION',
        'GAMBLING',
        'OBSCENITY'
    ]
    theft = [
        'BURGLARY',
        'DECEPTIVE PRACTICE',
        'MOTOR VEHICLE THEFT',
        'ROBBERY'
    ]
    assault = [
        'CRIM SEXUAL ASSAULT',
        'OFFENSE INVOLVING CHILDREN',
        'SEX OFFENSE',
        'HOMICIDE',
        'HUMAN TRAFFICKING'
    ]
    public_violation = [
        'WEAPONS VIOLATION',
        'CRIMINAL DAMAGE',
        'CRIMINAL TRESPASS',
        'ARSON',
        'KIDNAPPING',
        'STALKING',
        'INTIMIDATION',
        'PUBLIC INDECENCY',
        'CRIMINAL DEFACEMENT'
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

###############################################################################
# 4. SIMULATE BACKWARD FEATURE SELECTION USING MUTUAL INFORMATION
#    (APPROXIMATES THE "INFORMATION GAIN" MENTIONED IN THE PAPER)
###############################################################################
# The final subset in the paper was 9 features plus the target.
# They specifically mention using 'Block', 'Location Description', 'Domestic',
# 'Beat', 'District Ward', 'Community Area', 'X Coordinates', 'Y Coordinates',
# plus 'Primary Type' (target).
#
# However, let's demonstrate an approximate procedure:
###############################################################################

# 4.1 Prepare a baseline set of potential input features (excluding the target)
candidate_features = [
    col for col in df.columns
    if col not in ['Crime_Category', 'Primary Type', 'Case Number', 'Date', 'Year']
]

# We will convert any categorical columns to category codes 
# (this is a minimal approach so that mutual_info_classif can handle them).
for col in candidate_features:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

X_all = df[candidate_features].copy()
y_all = df['Crime_Category'].copy()

def evaluate_feature_set(features):
    """
    Train a quick Decision Tree on the selected features, 
    return accuracy on a 70-30 split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], y_all,
        test_size=0.3,
        random_state=100,
        stratify=y_all
    )
    
    # Minimal tree setup
    dt_temp = DecisionTreeClassifier(
        criterion='entropy', 
        random_state=100
    )
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
    
    # If the top of drop_candidates > best_accuracy, then we drop that feature
    # Because it means dropping that feature improved accuracy
    top_feature, top_acc = drop_candidates[0]
    if top_acc > best_accuracy:
        # drop that feature
        current_features.remove(top_feature)
        improved = True

print("\nSelected features after approximate backward elimination:")
print(current_features)

###############################################################################
# 5. TRAIN/TEST SPLIT WITH FINAL FEATURE SET
###############################################################################
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

###############################################################################
# 6. TRAIN THE CLASSIFIERS (DECISION TREE & NAIVE BAYES)
###############################################################################
# Decision Tree approximating J48
dt = DecisionTreeClassifier(
    criterion='entropy',  # ID3-like, not exactly C4.5
    random_state=100
)
dt.fit(X_train, y_train)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
nb_pred = nb.predict(X_test)

###############################################################################
# 7. EVALUATE THE RESULTS
###############################################################################
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{name} RESULTS")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    
    # Class-by-class correct classification
    for c in sorted(y_true.unique()):
        c_mask = (y_true == c)
        correct = (y_true[c_mask] == y_pred[c_mask]).sum()
        total = c_mask.sum()
        print(f"\nClass '{c}':")
        print(f"   Correct:   {correct}")
        print(f"   Incorrect: {total - correct}")
        print(f"   Total:     {total}")

evaluate_model("Decision Tree (Entropy)", y_test, dt_pred)
evaluate_model("Naive Bayes", y_test, nb_pred)
