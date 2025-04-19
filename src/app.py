from utils import db_connect
engine = db_connect()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

df = pd.read_csv("Ejercicio_RANDOM_FOREST.csv")

cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)
for col in cols_with_invalid_zeros:
    df[col] = df[col].fillna(df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=rf_clf.classes_).plot(cmap="Blues")
plt.title("Matriz de Confusión - Random Forest")
plt.show()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)
grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
best_rf_clf = grid_search.best_estimator_

y_pred_best = best_rf_clf.predict(X_test)
print("Mejor Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nMejor Reporte de clasificación:\n", classification_report(y_test, y_pred_best))

cm_best = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm_best, display_labels=best_rf_clf.classes_).plot(cmap="Blues")
plt.title("Matriz de Confusión - Random Forest Optimizado")
plt.show()

rf_balanced = RandomForestClassifier(n_estimators=100, 
                                     class_weight='balanced', 
                                     random_state=42)
rf_balanced.fit(X_train, y_train)

y_pred_bal = rf_balanced.predict(X_test)
print("Accuracy con balanceo:", accuracy_score(y_test, y_pred_bal))
print("\nReporte de clasificación balanceado:\n", classification_report(y_test, y_pred_bal))

cm_bal = confusion_matrix(y_test, y_pred_bal)
ConfusionMatrixDisplay(cm_bal, display_labels=rf_balanced.classes_).plot(cmap="Blues")
plt.title("Matriz de Confusión - Random Forest (class_weight='balanced')")
plt.show()


