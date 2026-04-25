# 🎓 Student Performance Predictor

A machine learning project that analyzes student data to predict academic performance using **Exploratory Data Analysis (EDA)** and classification models.

---

## 📌 Project Overview

This project investigates the factors that influence student academic outcomes — including study habits, attendance, parental education, and internet access — and builds a predictive model to classify student performance as **Low**, **Medium**, or **High**.

---

## 🧠 What This Project Covers

- **Data Generation** — Synthetic dataset modeled on UCI Student Performance structure
- **Exploratory Data Analysis** — Visualizing distributions, correlations, and patterns
- **Feature Engineering** — Encoding categorical variables, selecting relevant features
- **Model Training** — Random Forest & Logistic Regression classifiers
- **Model Evaluation** — Accuracy, classification report, confusion matrix
- **Feature Importance** — Understanding which factors matter most
- **Prediction** — Predicting performance for a new student profile

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `study_hours` | Weekly study hours (1–10) |
| `absences` | Number of school absences |
| `parent_edu` | Parental education level |
| `internet` | Internet access at home |
| `extra_support` | Extra academic support |
| `health` | Student health rating (1–5) |
| `prev_grade` | Previous semester grade (0–20) |
| `final_grade` | Final grade (0–20) |
| `performance` | Target: Low / Medium / High |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Matplotlib / Seaborn** — Data visualization
- **Scikit-learn** — Machine learning models

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python student_performance.py
```

---

## 📈 Output

Running the script generates:

| File | Description |
|---|---|
| `eda_analysis.png` | EDA charts — distributions, scatter plots, box plots |
| `model_evaluation.png` | Confusion matrices for both models |
| `feature_importance.png` | Feature importance from Random Forest |

---

## 🔍 Key Findings

- **Study hours** and **previous grade** are the strongest predictors of performance
- Students with **internet access** tend to perform better
- **High absences** strongly correlate with low performance
- **Random Forest** outperforms Logistic Regression on this dataset

---

## 🧪 Sample Prediction

```python
new_student = {
    'study_hours'  : 6,
    'absences'     : 3,
    'parent_edu'   : 2,   # secondary
    'internet'     : 1,   # yes
    'extra_support': 0,   # no
    'health'       : 4,
    'prev_grade'   : 14
}
# Predicted Performance → High
```

---

## 📁 Project Structure

```
student-performance-predictor/
│
├── student_performance.py   # Main project script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── eda_analysis.png         # Generated after running
├── model_evaluation.png     # Generated after running
└── feature_importance.png   # Generated after running
```

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**[Prajwal R]**
- GitHub: [@Prajwal2222](https://github.com/your_username)

---

> Built as part of a structured AI/ML learning journey. Feedback welcome.
