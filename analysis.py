"""
analysis.py
-----------
End-to-end data science pipeline for the Student Performance project.

Steps (each step matches a course topic — useful for VIVA prep):
    1. Data Collection   -> read the CSV
    2. Data Inspection   -> .head(), .info(), .describe()
    3. Data Cleaning     -> check missing values, fix data types
    4. Feature Eng.      -> create target 'passed', encode categoricals
    5. EDA + plots       -> histograms, heatmap, boxplots
    6. Train/Test Split  -> 80/20 split
    7. Model Training    -> Logistic Regression, Decision Tree, Random Forest
    8. Model Evaluation  -> accuracy, confusion matrix, classification report
    9. Save Best Model   -> pickle for the Streamlit dashboard

Run:    python analysis.py
Output: plots/*.png  and  model.pkl
"""

# =============================================================================
# 0) IMPORT LIBRARIES
# =============================================================================
# pandas    -> tables / DataFrames (think Excel inside Python)
# numpy     -> fast math on arrays
# matplotlib + seaborn -> plotting libraries
# sklearn   -> machine learning models and helpers
# pickle    -> save Python objects (our trained model) to a file
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# Make sure the folder for saved plots exists.
os.makedirs("plots", exist_ok=True)

# Set a clean default style for all seaborn plots.
sns.set_style("whitegrid")


# =============================================================================
# 1) LOAD THE DATA
# =============================================================================
# The UCI file uses semicolons (;) as the separator instead of commas.
# So we tell pandas: sep=";"
print("\n" + "=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_csv("data/student-mat.csv", sep=";")
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")


# =============================================================================
# 2) INSPECT THE DATA
# =============================================================================
# Always look at your data before doing anything!
print("\n" + "=" * 60)
print("STEP 2: INSPECTING DATA")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn data types and non-null counts:")
print(df.info())

print("\nSummary statistics for numeric columns:")
print(df.describe())


# =============================================================================
# 3) DATA CLEANING — CHECK MISSING VALUES
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: CLEANING")
print("=" * 60)

# .isnull().sum() counts how many missing values each column has.
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "None! Dataset is clean.")

# This dataset is famously clean (no missing values), so no imputation needed.
# In a real project, we'd use df.fillna(...) or df.dropna(...) here.


# =============================================================================
# 4) FEATURE ENGINEERING
# =============================================================================
# We'll turn the project into a CLASSIFICATION problem:
#     "Will the student PASS or FAIL the final exam?"
#
# G3 is the final grade (0-20 in the Portuguese system).
# Pass = G3 >= 10  (this is the standard pass mark in Portugal).
print("\n" + "=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

df["passed"] = (df["G3"] >= 10).astype(int)  # 1 = pass, 0 = fail

print(f"\nClass balance:")
print(df["passed"].value_counts())
print(f"Pass rate = {df['passed'].mean() * 100:.1f}%")


# =============================================================================
# 5) EXPLORATORY DATA ANALYSIS (EDA) + VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: EDA + PLOTS")
print("=" * 60)

# --- Plot 1: Distribution of final grades --------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=20, kde=True, color="steelblue")
plt.axvline(10, color="red", linestyle="--", label="Pass mark (10)")
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig("plots/01_grade_distribution.png", dpi=120)
plt.close()
print("Saved plots/01_grade_distribution.png")

# --- Plot 2: Pass vs Fail counts -----------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(x="passed", data=df, hue="passed",
              palette=["salmon", "seagreen"], legend=False)
plt.title("Pass vs Fail")
plt.xticks([0, 1], ["Fail", "Pass"])
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("plots/02_pass_fail_count.png", dpi=120)
plt.close()
print("Saved plots/02_pass_fail_count.png")

# --- Plot 3: Correlation heatmap of numeric features ---------------------
# A correlation matrix tells us how strongly two variables move together.
# Values range -1 (perfect negative) to +1 (perfect positive).
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 9))
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.savefig("plots/03_correlation_heatmap.png", dpi=120)
plt.close()
print("Saved plots/03_correlation_heatmap.png")

# --- Plot 4: Grade by study time -----------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=df, hue="studytime",
            palette="Blues", legend=False)
plt.title("Final Grade vs Weekly Study Time")
plt.xlabel("Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("plots/04_grade_by_studytime.png", dpi=120)
plt.close()
print("Saved plots/04_grade_by_studytime.png")

# --- Plot 5: Grade by past failures --------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="failures", y="G3", data=df, hue="failures",
            palette="Reds", legend=False)
plt.title("Final Grade vs Number of Past Failures")
plt.xlabel("Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("plots/05_grade_by_failures.png", dpi=120)
plt.close()
print("Saved plots/05_grade_by_failures.png")

# --- Plot 6: Absences vs grade -------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x="absences", y="G3", hue="passed", data=df,
                palette={0: "red", 1: "green"}, alpha=0.6)
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("plots/06_absences_vs_grade.png", dpi=120)
plt.close()
print("Saved plots/06_absences_vs_grade.png")

# --- Plot 7: Grade by gender ---------------------------------------------
plt.figure(figsize=(6, 5))
sns.boxplot(x="sex", y="G3", data=df, hue="sex",
            palette="Set2", legend=False)
plt.title("Final Grade by Gender")
plt.xlabel("Sex (F=Female, M=Male)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("plots/07_grade_by_gender.png", dpi=120)
plt.close()
print("Saved plots/07_grade_by_gender.png")


# =============================================================================
# 6) PREPARE FEATURES FOR MODELING
# =============================================================================
# We pick a focused set of features that are easy for a user to fill in
# on the dashboard. We exclude G1, G2, and G3 — predicting G3 from G1/G2
# is too easy (correlation > 0.9). The interesting question is:
#
#     "Can we predict pass/fail from BACKGROUND factors only?"
print("\n" + "=" * 60)
print("STEP 6: FEATURE SELECTION + ENCODING")
print("=" * 60)

feature_cols = [
    "sex",         # Female / Male
    "age",         # 15 – 22
    "studytime",   # 1 – 4
    "failures",    # 0 – 3
    "absences",    # 0 – 93
    "Medu",        # Mother's education 0 – 4
    "Fedu",        # Father's education 0 – 4
    "goout",       # Going out with friends 1 – 5
    "Dalc",        # Workday alcohol 1 – 5
    "Walc",        # Weekend alcohol 1 – 5
    "health",      # Current health 1 – 5
    "internet",    # yes / no
]

X = df[feature_cols].copy()
y = df["passed"].copy()

# Convert text categories ("yes"/"no", "F"/"M") to numbers using one-hot.
# pd.get_dummies creates a new column for each category.
X = pd.get_dummies(X, columns=["sex", "internet"], drop_first=True)
print(f"\nFinal feature matrix shape: {X.shape}")
print("Final feature columns:", list(X.columns))


# =============================================================================
# 7) TRAIN / TEST SPLIT
# =============================================================================
# We hold out 20% of data for TESTING. The model never sees this data
# during training, so the test accuracy reflects real-world performance.
print("\n" + "=" * 60)
print("STEP 7: TRAIN/TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # for reproducibility (always same split)
    stratify=y,          # keep pass/fail ratio same in both sets
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# Standardize numeric features (mean=0, std=1).
# Logistic Regression works much better when features are on the same scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================================================================
# 8) TRAIN MULTIPLE MODELS AND COMPARE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: TRAINING MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    # Logistic Regression uses scaled features; tree-based models don't need scaling.
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = (model, acc, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))


# =============================================================================
# 9) PICK BEST MODEL + FEATURE IMPORTANCE PLOT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9: FEATURE IMPORTANCE")
print("=" * 60)

# Pick the model with the highest accuracy.
best_name = max(results, key=lambda k: results[k][1])
best_model, best_acc, _ = results[best_name]
print(f"\nBest model: {best_name} (accuracy = {best_acc:.3f})")

# For Random Forest / Decision Tree, we can directly read 'feature_importances_'.
# This tells us which features the model relied on most.
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="teal")
    plt.title(f"Feature Importance ({best_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("plots/08_feature_importance.png", dpi=120)
    plt.close()
    print("Saved plots/08_feature_importance.png")
    print("\nTop features:")
    print(importances.sort_values(ascending=False).head(8))


# =============================================================================
# 10) SAVE THE MODEL FOR THE STREAMLIT APP
# =============================================================================
# We bundle the model + scaler + column order into a single dictionary
# so the app can recreate the exact preprocessing.
print("\n" + "=" * 60)
print("STEP 10: SAVE MODEL")
print("=" * 60)

# We'll save the Random Forest specifically — it's robust and gives feature importance.
rf_model = results["Random Forest"][0]

bundle = {
    "model":   rf_model,
    "scaler":  scaler,                      # in case you want to scale later
    "columns": list(X.columns),             # exact column order matters!
    "feature_cols_raw": feature_cols,       # original column names
}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Saved model.pkl")
print("\nDone! Now run:  streamlit run app.py")
