# =============================================================================
# Student Performance Predictor
# Author: [Your Name]
# Description: Analyzes student data to predict academic performance
#              using exploratory data analysis and machine learning.
# Dataset: UCI Student Performance Dataset (public domain)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: GENERATE SYNTHETIC DATASET
# Based on UCI Student Performance structure
# ─────────────────────────────────────────────

np.random.seed(42)
n = 400

def generate_dataset(n):
    study_hours   = np.random.randint(1, 10, n)
    absences      = np.random.randint(0, 20, n)
    parent_edu    = np.random.choice(['none', 'primary', 'secondary', 'higher'], n)
    internet      = np.random.choice(['yes', 'no'], n)
    extra_support = np.random.choice(['yes', 'no'], n)
    health        = np.random.randint(1, 6, n)          # 1 (poor) to 5 (excellent)
    prev_grade    = np.random.randint(5, 20, n)         # out of 20

    # Final grade influenced by real factors
    base = (
        study_hours * 1.2
        + prev_grade * 0.5
        + health * 0.4
        - absences * 0.3
        + np.where(internet == 'yes', 1.5, 0)
        + np.where(extra_support == 'yes', 1.0, 0)
        + np.random.normal(0, 1.5, n)
    )

    # Normalize to 0–20 range
    final_grade = np.clip(base, 0, 20).astype(int)

    # Performance label
    performance = pd.cut(
        final_grade,
        bins=[-1, 9, 13, 20],
        labels=['Low', 'Medium', 'High']
    )

    df = pd.DataFrame({
        'study_hours'   : study_hours,
        'absences'      : absences,
        'parent_edu'    : parent_edu,
        'internet'      : internet,
        'extra_support' : extra_support,
        'health'        : health,
        'prev_grade'    : prev_grade,
        'final_grade'   : final_grade,
        'performance'   : performance
    })
    return df

df = generate_dataset(n)
print("=" * 55)
print("  STUDENT PERFORMANCE PREDICTOR")
print("=" * 55)
print(f"\n Dataset loaded: {df.shape[0]} students, {df.shape[1]} features\n")
print(df.head())


# ─────────────────────────────────────────────
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

print("\n─── Basic Statistics ───")
print(df.describe())

print("\n─── Performance Distribution ───")
print(df['performance'].value_counts())

# Set visual style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Student Performance — Exploratory Data Analysis", fontsize=16, fontweight='bold')

# Plot 1: Performance Distribution
df['performance'].value_counts().plot(
    kind='bar', ax=axes[0, 0], color=['#E74C3C', '#F39C12', '#2ECC71'], edgecolor='black'
)
axes[0, 0].set_title("Performance Distribution")
axes[0, 0].set_xlabel("Performance Level")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis='x', rotation=0)

# Plot 2: Study Hours vs Final Grade
axes[0, 1].scatter(
    df['study_hours'], df['final_grade'],
    c=df['performance'].cat.codes, cmap='RdYlGn', alpha=0.6, edgecolors='gray', linewidth=0.3
)
axes[0, 1].set_title("Study Hours vs Final Grade")
axes[0, 1].set_xlabel("Study Hours per Week")
axes[0, 1].set_ylabel("Final Grade (out of 20)")

# Plot 3: Absences vs Final Grade
sns.boxplot(x='performance', y='absences', data=df, ax=axes[1, 0],
            palette={'Low': '#E74C3C', 'Medium': '#F39C12', 'High': '#2ECC71'},
            order=['Low', 'Medium', 'High'])
axes[1, 0].set_title("Absences by Performance Level")
axes[1, 0].set_xlabel("Performance Level")
axes[1, 0].set_ylabel("Number of Absences")

# Plot 4: Internet Access vs Performance
internet_perf = df.groupby(['internet', 'performance']).size().unstack()
internet_perf.plot(kind='bar', ax=axes[1, 1], color=['#E74C3C', '#F39C12', '#2ECC71'],
                   edgecolor='black')
axes[1, 1].set_title("Internet Access vs Performance")
axes[1, 1].set_xlabel("Internet Access")
axes[1, 1].set_ylabel("Count")
axes[1, 1].tick_params(axis='x', rotation=0)
axes[1, 1].legend(title='Performance')

plt.tight_layout()
plt.savefig("eda_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n EDA chart saved → eda_analysis.png")


# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────

df_model = df.copy()

# Encode categorical columns
le = LabelEncoder()
for col in ['parent_edu', 'internet', 'extra_support']:
    df_model[col] = le.fit_transform(df_model[col])

# Features and target
X = df_model.drop(columns=['performance', 'final_grade'])
y = df_model['performance']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n─── Train/Test Split ───")
print(f" Training samples : {X_train.shape[0]}")
print(f" Testing samples  : {X_test.shape[0]}")


# ─────────────────────────────────────────────
# STEP 4: MODEL TRAINING
# ─────────────────────────────────────────────

models = {
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
}

results = {}
print("\n─── Model Training & Evaluation ───\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': acc, 'predictions': y_pred}
    print(f" {name}")
    print(f"  Accuracy : {acc:.2%}")
    print(f"\n{classification_report(y_test, y_pred)}")
    print("-" * 45)

# Best model
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
print(f"\n Best Model: {best_name} ({results[best_name]['accuracy']:.2%} accuracy)")


# ─────────────────────────────────────────────
# STEP 5: VISUALIZATION — CONFUSION MATRIX
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Evaluation — Confusion Matrices", fontsize=14, fontweight='bold')

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['predictions'], labels=['Low', 'Medium', 'High'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
    disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
    axes[idx].set_title(f"{name}\nAccuracy: {res['accuracy']:.2%}")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n Model evaluation chart saved → model_evaluation.png")


# ─────────────────────────────────────────────
# STEP 6: FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────

rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_names = X.columns

feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(9, 5))
colors = ['#2ECC71' if i >= len(feat_df) - 3 else '#3498DB' for i in range(len(feat_df))]
plt.barh(feat_df['Feature'], feat_df['Importance'], color=colors, edgecolor='black')
plt.title("Feature Importance — Random Forest", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n Feature importance chart saved → feature_importance.png")


# ─────────────────────────────────────────────
# STEP 7: PREDICT A NEW STUDENT
# ─────────────────────────────────────────────

print("\n─── Predict New Student ───")

new_student = pd.DataFrame([{
    'study_hours'   : 6,
    'absences'      : 3,
    'parent_edu'    : 2,    # 0=none, 1=primary, 2=secondary, 3=higher
    'internet'      : 1,    # 1=yes, 0=no
    'extra_support' : 0,    # 0=no, 1=yes
    'health'        : 4,
    'prev_grade'    : 14
}])

prediction = best_model.predict(new_student)[0]
probabilities = best_model.predict_proba(new_student)[0]
classes = best_model.classes_

print(f"\n Input Features:")
for col, val in new_student.iloc[0].items():
    print(f"  {col:<15}: {val}")

print(f"\n Predicted Performance : {prediction}")
print(f" Confidence Breakdown  :")
for cls, prob in zip(classes, probabilities):
    bar = "█" * int(prob * 20)
    print(f"  {cls:<8} {bar:<20} {prob:.1%}")

print("\n" + "=" * 55)
print("  Analysis Complete.")
print("=" * 55)
