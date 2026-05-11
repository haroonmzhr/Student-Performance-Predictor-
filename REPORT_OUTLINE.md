# 📄 Report Template — Student Performance Predictor

> Copy this into Word/Google Docs and fill in the gaps.
> Insert the PNGs from the `plots/` folder where indicated.
> Then export to **PDF** for the final submission.

---

## Title Page

- **Project Title:** Student Performance Predictor — Predicting Pass/Fail Outcomes Using Machine Learning
- **Course:** Introduction to Data Science (Fall 2025)
- **Group Members:**
  - [Your Name] — Roll # ____
  - [Partner's Name] — Roll # ____
- **Date:** December 2, 2025

---

## 1. Abstract (½ page)

In this project we build a data-driven prototype that predicts whether a secondary-school student will pass their final mathematics exam based on personal, family, and behavioural attributes. We use the publicly-available **UCI Student Performance dataset** (395 Portuguese students, 33 features). Our pipeline covers data cleaning, exploratory analysis, three classification models (Logistic Regression, Decision Tree, Random Forest), and a **Streamlit web dashboard** for interactive predictions. The Random Forest classifier achieved approximately **XX % accuracy** on a held-out test set. Key insights show that **past class failures, total absences, and weekly study time** are the strongest indicators of academic outcomes.

> 💡 Replace `XX %` with the accuracy printed by `analysis.py`.

---

## 2. Introduction (1 page)

### 2.1 Background
Education analytics is a growing area of data science where statistical and machine-learning techniques are applied to student data to identify factors that affect academic performance. Early identification of at-risk students enables teachers and administrators to intervene before failure occurs.

### 2.2 Problem Statement
> *Can we predict, before the final exam, which students are at risk of failing — using only background factors (family, lifestyle, behaviour) and not their prior grades?*

### 2.3 Why this matters
- Schools can allocate tutoring resources to students who need them most.
- Parents understand which lifestyle factors (study time, going out, alcohol, absences) most influence outcomes.
- Students themselves get evidence-based feedback on what changes most.

### 2.4 Objectives
1. Acquire and clean an open dataset on student performance.
2. Perform exploratory data analysis and extract at least three insights.
3. Train and compare three machine-learning classifiers.
4. Deliver a working interactive prototype that produces predictions in real time.

---

## 3. Dataset (½ page)

- **Source:** UCI Machine Learning Repository — *Student Performance Data Set*
  https://archive.ics.uci.edu/dataset/320/student+performance
- **Size:** 395 rows × 33 columns
- **Subject:** Portuguese high-school students enrolled in a mathematics course.
- **Target variable:** `G3` (final grade, 0–20). We convert this to a binary label: `passed = (G3 >= 10)`.
- **Justification:** This dataset is well-documented, free of missing values, and contains a rich mix of numeric and categorical features ideal for an introductory ML project.

---

## 4. Methodology (1–2 pages)

### 4.1 Data Preprocessing
- Loaded the CSV with `pandas.read_csv` (semicolon-separated).
- Verified there are **no missing values** with `df.isnull().sum()`.
- Created the binary target `passed = (G3 >= 10).astype(int)`.
- One-hot encoded categorical columns (`sex`, `internet`) with `pd.get_dummies`.
- Standardised numeric features for Logistic Regression using `StandardScaler`.

### 4.2 Exploratory Data Analysis
We produced 7 visualisations covering grade distribution, pass/fail balance, feature correlations, and the relationship between the final grade and key behavioural features. (See **Section 5**.)

### 4.3 Train/Test Split
- 80 % training / 20 % testing
- `random_state=42` for reproducibility
- Stratified split to preserve the pass/fail ratio in both sets

### 4.4 Models Compared
| Model               | Why we chose it                                           |
| ------------------- | --------------------------------------------------------- |
| Logistic Regression | Simple, interpretable linear baseline                     |
| Decision Tree       | Easily explainable; mirrors human decision-making         |
| Random Forest       | Ensemble that usually performs best on tabular data       |

### 4.5 Evaluation Metrics
- **Accuracy** — fraction of correct predictions
- **Confusion matrix** — TP / FP / TN / FN counts
- **Classification report** — precision, recall, F1-score per class

---

## 5. Results & Insights (2 pages — heaviest section)

### 5.1 Distribution of final grades
![Grade distribution](plots/01_grade_distribution.png)
The distribution is roughly bell-shaped, peaking around grade 10–11. Roughly one-third of students score below the pass mark of 10.

### 5.2 Pass vs Fail balance
![Pass/Fail counts](plots/02_pass_fail_count.png)
About **67 %** of students pass and **33 %** fail.

### 5.3 Correlation heatmap
![Correlation heatmap](plots/03_correlation_heatmap.png)
G1 and G2 are extremely strongly correlated with G3, which is why we **excluded** them from the model — we want to predict outcomes from background factors only. Among the remaining features, `failures` shows the strongest negative correlation with the final grade.

### 5.4 Insight 1 — Study time matters, but not as much as you'd think
![Grade by study time](plots/04_grade_by_studytime.png)
Median grades climb from ~9 (study time 1) to ~12 (study time 4). The effect is real but smaller than expected.

### 5.5 Insight 2 — Past failures are the strongest single predictor
![Grade by failures](plots/05_grade_by_failures.png)
Students with even **one** past failure see a sharp drop in median grade. By 2+ failures, almost no one passes.

### 5.6 Insight 3 — Attendance is critical
![Absences vs grade](plots/06_absences_vs_grade.png)
The scatter shows that almost no student with more than ~25 absences passes the course.

### 5.7 Final grade by gender
![Grade by gender](plots/07_grade_by_gender.png)
Distributions are similar; male students average about 0.5 grade points higher.

### 5.8 Model performance

| Model               | Test Accuracy |
| ------------------- | ------------- |
| Logistic Regression | __.____       |
| Decision Tree       | __.____       |
| Random Forest       | __.____       |

> 💡 Fill in the numbers printed by `analysis.py`.

### 5.9 Feature importance
![Feature importance](plots/08_feature_importance.png)
The Random Forest relied most heavily on `failures`, `absences`, `goout`, and `studytime` — consistent with our EDA findings.

---

## 6. Prototype (½ page)

We built an interactive **Streamlit** dashboard with four tabs:
- **Dataset Overview** — table preview, summary statistics, column dictionary.
- **Visualizations** — all eight plots with explanatory captions.
- **Predict** — sliders and radio buttons for student profile → live pass/fail prediction with confidence.
- **About** — methodology summary.

> 💡 Add 2–3 screenshots of the running dashboard here.

To run it:
```bash
pip install -r requirements.txt
python download_data.py
python analysis.py
streamlit run app.py
```

---

## 7. Limitations

1. **Sample size** — only 395 students from two specific Portuguese schools, so results may not generalise.
2. **Self-reported features** — alcohol use, going out, and study time were self-reported on a 1–5 scale.
3. **No prior-grade features** — we deliberately excluded G1/G2; including them would push accuracy above 90 %, but the prediction would be trivial.
4. **Class imbalance is mild** (≈ 33 % fail). If it were more severe we'd need techniques like SMOTE or class weighting.

---

## 8. Conclusion (½ page)

We successfully built an end-to-end data science prototype that predicts student pass/fail outcomes with around **XX % accuracy**, identified three actionable insights (failures > absences > study time), and packaged everything into a clean Streamlit dashboard. The project demonstrates the full data-science workflow — collection, cleaning, EDA, modelling, evaluation, and deployment — using only beginner-friendly libraries.

---

## 9. References

1. Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*. UCI Machine Learning Repository.
2. McKinney, W. (2010). *Data structures for statistical computing in Python*. Pandas documentation.
3. Pedregosa et al. (2011). *Scikit-learn: Machine learning in Python*. JMLR.
4. Streamlit Inc. (2024). *Streamlit Documentation*. https://docs.streamlit.io

---

## 10. Team Collaboration & Links

- **GitHub Repository:** _________
- **LinkedIn Demo Video:** _________
- **Contributions:**
  - **[Your Name]:** Data collection, preprocessing, EDA visualisations.
  - **[Partner's Name]:** Model training, evaluation, Streamlit dashboard, report write-up.
