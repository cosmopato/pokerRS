from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os
import random

# =============================
# Cargar y preprocesar dataset
# =============================
df = pd.read_csv(os.path.join(os.getcwd(), "dataset", "poker_data.csv"))
df["Card1"] = df["Card1"].replace({"10": "T"})
df["Card2"] = df["Card2"].replace({"10": "T"})
valores = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
df["Card1_val"] = df["Card1"].map(valores)
df["Card2_val"] = df["Card2"].map(valores)

# Etiquetas codificadas
le_pos = LabelEncoder()
le_type = LabelEncoder()
le_action = LabelEncoder()

# Añadir un poco de ruido (si fuera útil para testear sobreajuste)
for i in df.sample(frac=0.05, random_state=42).index:
    df.at[i, 'Action'] = random.choice(['Fold', 'Limp', 'Raise'])

df["Position_enc"] = le_pos.fit_transform(df["Position"])
df["HandType_enc"] = le_type.fit_transform(df["HandType"])
df["Action_enc"] = le_action.fit_transform(df["Action"])

# Features y Target
X = df[["Card1_val", "Card2_val", "HandType_enc", "Position_enc", "BigBlinds"]]
y = df["Action_enc"]

# =============================
# GridSearchCV con CalibratedClassifierCV
# =============================
param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [None, 10, 20],
    'estimator__min_samples_split': [2, 5]
}

base_rf = RandomForestClassifier(random_state=42)
modelo_calibrado = CalibratedClassifierCV(base_rf, cv=5)

grid = GridSearchCV(
    estimator=modelo_calibrado,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=1
)

# Entrenamiento
grid.fit(X, y)
mejor_modelo = grid.best_estimator_

print("✅ Mejores parámetros:", grid.best_params_)

# =============================
# Guardar modelo y codificadores
# =============================
os.makedirs("modelos", exist_ok=True)
joblib.dump(mejor_modelo, os.path.join("modelos", "modelo_poker.pkl"))
joblib.dump(le_pos, os.path.join("modelos", "le_pos.pkl"))
joblib.dump(le_type, os.path.join("modelos", "le_type.pkl"))
joblib.dump(le_action, os.path.join("modelos", "le_action.pkl"))
