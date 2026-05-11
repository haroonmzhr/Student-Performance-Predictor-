**Student Performance Predictor**


**Project Type:** Data-Driven Insights Prototype
**Domain:** Education Analytics

---

What this project does

Predicts whether a student will **pass or fail** the final math exam based on background factors (study time, past failures, absences, family education, etc.) — using the **UCI Student Performance** dataset of 395 Portuguese students.

The deliverables match every item in the assignment rubric:

| Rubric Requirement                 | Where it's done                          |
| ---------------------------------- | ---------------------------------------- |
| Data collection                    | `download_data.py`                       |
| Data preprocessing & cleaning      | `analysis.py` Steps 3–4                  |
| EDA + ≥3 insights + visualizations | `analysis.py` Step 5 → `plots/` folder   |
| Machine learning model             | `analysis.py` Steps 6–8                  |
| Model evaluation                   | `analysis.py` Step 8 (accuracy + report) |
| Prototype with interactive UI      | `app.py` (Streamlit)                     |

---

How to run (3 commands)

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Download the dataset

```powershell
python download_data.py
```
This creates `data/student-mat.csv` (only needs to be run once).

### 3. Run the analysis (generates plots and trains the model)

```powershell
python analysis.py
```
This creates 8 PNGs in `plots/` and saves the trained model as `model.pkl`.

### 4. Launch the dashboard

```powershell
streamlit run app.py
```
Your browser opens automatically at `http://localhost:8501`.

---

Project Structure

```
IDS PROJECT/
├── data/
│   └── student-mat.csv           # Dataset (395 students × 33 columns)
├── plots/
│   ├── 01_grade_distribution.png
│   ├── 02_pass_fail_count.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_grade_by_studytime.png
│   ├── 05_grade_by_failures.png
│   ├── 06_absences_vs_grade.png
│   ├── 07_grade_by_gender.png
│   └── 08_feature_importance.png
├── download_data.py              # Fetches dataset from UCI
├── analysis.py                   # End-to-end pipeline (EDA + model)
├── app.py                        # Streamlit dashboard
├── model.pkl                     # Trained Random Forest (created by analysis.py)
├── requirements.txt              # Python libraries
├── README.md                     # ← You are here
└── REPORT_OUTLINE.md             # Template for your PDF report
```

---

Concepts Covered 

| Concept                  | Where in the code                  | One-line explanation                                              |
| ------------------------ | ---------------------------------- | ----------------------------------------------------------------- |
| Reading CSV with pandas  | `analysis.py` Step 1               | `pd.read_csv()` loads tabular data into a DataFrame.              |
| Data inspection          | `analysis.py` Step 2               | `.head()`, `.info()`, `.describe()` show structure and stats.     |
| Missing-value check      | `analysis.py` Step 3               | `.isnull().sum()` counts NaNs per column.                         |
| Feature engineering      | `analysis.py` Step 4               | We created `passed = (G3 >= 10)` — a binary target.               |
| Histogram                | Plot 1                             | Shows the distribution of a single numeric variable.              |
| Bar / count plot         | Plot 2                             | Shows category counts (pass vs fail).                             |
| Correlation heatmap      | Plot 3                             | Visualizes pairwise correlation between numeric features.         |
| Box plot                 | Plots 4, 5, 7                      | Compares the distribution of a numeric variable across groups.    |
| Scatter plot             | Plot 6                             | Shows the relationship between two numeric variables.             |
| One-hot encoding         | `analysis.py` Step 6               | `pd.get_dummies` turns categories into 0/1 columns.               |
| Train/test split         | `analysis.py` Step 7               | Holding out 20 % keeps the test honest.                           |
| Feature scaling          | `analysis.py` Step 7               | `StandardScaler` puts features on the same scale (mean 0, std 1). |
| Logistic Regression      | `analysis.py` Step 8               | Linear model that outputs class probabilities.                    |
| Decision Tree            | `analysis.py` Step 8               | If/else tree of conditions — easy to interpret.                   |
| Random Forest            | `analysis.py` Step 8               | Many decision trees voting → more robust predictions.             |
| Accuracy                 | `analysis.py` Step 8               | (correct predictions) ÷ (total predictions).                      |
| Confusion matrix         | `analysis.py` Step 8               | Table of TP, FP, TN, FN.                                          |
| Classification report    | `analysis.py` Step 8               | Precision, recall, F1-score per class.                            |
| Feature importance       | `analysis.py` Step 9 → Plot 8      | Which inputs the model relied on most.                            |
| Model serialization      | `analysis.py` Step 10              | `pickle.dump()` saves the model to disk.                          |
| Streamlit caching        | `app.py` (`@st.cache_data`)        | Avoids reloading data on every interaction.                       |

---

Three Key Insights from the Data

1. **Past failures are the strongest predictor of future failure.** Students with 1+ past failures average 2–3 grade points lower.
2. **Attendance matters a lot.** Beyond ~25 absences, almost no student passes.
3. **Study time has a positive but smaller effect** than failures or absences — students who study >5 h/week clearly score higher, but the boost is moderate.

---

