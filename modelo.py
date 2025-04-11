from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
import os
import random

# Cargar y preprocesar el dataset
df = pd.read_csv(os.path.join(os.getcwd(),"dataset","poker_data.csv"))
df["Card1"] = df["Card1"].replace({"10": "T"})
df["Card2"] = df["Card2"].replace({"10": "T"})
valores = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
df["Card1_val"] = df["Card1"].map(valores)
df["Card2_val"] = df["Card2"].map(valores)

# Codificadores
le_pos = LabelEncoder()
le_type = LabelEncoder()
le_action = LabelEncoder()

for i in df.sample(frac=0.05, random_state=42).index:
    df.at[i, 'Action'] = random.choice(['Fold', 'Limp', 'Raise'])

df["Position_enc"] = le_pos.fit_transform(df["Position"])
df["HandType_enc"] = le_type.fit_transform(df["HandType"])
df["Action_enc"] = le_action.fit_transform(df["Action"])

# Features y target
X = df[["Card1_val", "Card2_val", "HandType_enc", "Position_enc", "BigBlinds"]]
y = df["Action_enc"]

modelo_cv = CalibratedClassifierCV(estimator=RandomForestClassifier(), cv=5)

# Ahora entrenamos y guardamos
modelo_cv.fit(X, y)

# Guardar
os.makedirs("modelos", exist_ok=True)
joblib.dump(modelo_cv, os.path.join("modelos", "modelo_poker.pkl"))
joblib.dump(le_pos, os.path.join("modelos", "le_pos.pkl"))
joblib.dump(le_type, os.path.join("modelos", "le_type.pkl"))
joblib.dump(le_action, os.path.join("modelos", "le_action.pkl"))
