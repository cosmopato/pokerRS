import random
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Definimos las posiciones en la mesa
POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

# Definimos los palos y valores de las cartas
SUITS = ["hearts", "diamonds", "clubs", "spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

# Función para generar una mano aleatoria
def generate_hand():
    card1 = (random.choice(RANKS), random.choice(SUITS))
    card2 = (random.choice(RANKS), random.choice(SUITS))
    while card2 == card1:  # Asegurar que no sean la misma carta
        card2 = (random.choice(RANKS), random.choice(SUITS))
    return card1, card2

# Función para clasificar la mano
def classify_hand(card1, card2):
    rank1, suit1 = card1
    rank2, suit2 = card2
    if rank1 == rank2:
        return "Pair"
    elif suit1 == suit2:
        return "Suited"
    else:
        return "Off-suited"

# Función para asignar una acción recomendada basada en heurísticas básicas
def recommend_action(hand_type, position):
    if hand_type == "Pair" and position in ["CO", "BTN", "SB"]:
        return "Raise"
    elif hand_type == "Suited" and position in ["MP", "CO", "BTN"]:
        return "Call"
    else:
        return "Fold"

# Generamos el dataset
num_hands = 1000000  # Puedes aumentar este número para hacerlo más pesado
data = []

for _ in range(num_hands):
    position = random.choice(POSITIONS)
    card1, card2 = generate_hand()
    hand_type = classify_hand(card1, card2)
    action = recommend_action(hand_type, position)
    data.append([position, card1[0], card2[0], hand_type, action])

# Convertimos a DataFrame
df = pd.DataFrame(data, columns=["Position", "Card1", "Card2", "Hand Type", "Action"])

# Preprocesamiento: Convertir variables categóricas en números
label_encoders = {}
for col in ["Position", "Card1", "Card2", "Hand Type", "Action"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separar datos en entrenamiento y prueba
X = df.drop(columns=["Action"])
y = df["Action"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Guardar el modelo entrenado y los label encoders
joblib.dump(model, "poker_recommender.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Modelo guardado como poker_recommender.pkl")

# Función para predecir la acción recomendada
def recommend_play(position, card1, card2):
    # Cargar modelo y encoders
    model = joblib.load("poker_recommender.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    
    # Convertir entrada en formato numérico
    position_encoded = label_encoders["Position"].transform([position])[0]
    card1_encoded = label_encoders["Card1"].transform([card1])[0]
    card2_encoded = label_encoders["Card2"].transform([card2])[0]
    hand_type = classify_hand((card1, None), (card2, None))
    hand_type_encoded = label_encoders["Hand Type"].transform([hand_type])[0]
    
    # Predecir acción
    input_data = [[position_encoded, card1_encoded, card2_encoded, hand_type_encoded]]
    action_encoded = model.predict(input_data)[0]
    action = label_encoders["Action"].inverse_transform([action_encoded])[0]
    
    return action

# Interfaz de usuario en consola
if __name__ == "__main__":
    print("Sistema de Recomendación de Poker")
    position = input("Ingresa la posición (UTG, MP, CO, BTN, SB, BB): ")
    card1 = input("Ingresa la primera carta (ejemplo: A, K, Q, J, 10, 9...2): ")
    card2 = input("Ingresa la segunda carta: ")
    
    recommended_action = recommend_play(position, card1, card2)
    print(f"Acción recomendada: {recommended_action}")
