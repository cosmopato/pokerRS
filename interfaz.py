import streamlit as st
import pandas as pd
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import base64
import os
import plotly.express as px
import random
import plotly.graph_objects as go

# Función para fondo con imagen local en base64
def obtener_fondo_base64(ruta_imagen):
    with open(ruta_imagen, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Ruta robusta a imagen local
ruta_imagen = os.path.join(os.path.dirname(__file__), "ChatGPT Image 11 abr 2025, 10_30_44.png")
bg_base64 = obtener_fondo_base64(ruta_imagen)

st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .recomendacion-box {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border: 2px solid gold;
            border-radius: 12px;
            font-size: 20px;
            color: white;
            margin-top: 20px;
        }}
        h1, h2, h3, h4 {{
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
        }}
    </style>
""", unsafe_allow_html=True)

# Sonido al hacer clic
def reproducir_sonido():
    st.markdown("""
        <audio autoplay>
            <source src="https://www.fesliyanstudios.com/play-mp3/387" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)

# Título
st.markdown("""
# ♦️ **Sistema de Recomendación Preflop de Poker** ♣️
Descubre la mejor acción en todo momento 🃏🎲
""")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv(os.path.join(os.getcwd(),"dataset","poker_data.csv"))
    df["Card1"] = df["Card1"].replace({"10": "T"})
    df["Card2"] = df["Card2"].replace({"10": "T"})
    df["ManoNormalizada"] = df.apply(lambda row: normalizar_mano(row["Card1"], row["Suit1"], row["Card2"], row["Suit2"]), axis=1)
    return df

# Funciones

def traducir_carta(valor):
    return 'T' if valor == '10' else valor

def normalizar_mano(c1, s1, c2, s2):
    v1 = traducir_carta(c1)
    v2 = traducir_carta(c2)
    cartas = sorted([v1, v2], reverse=True, key=lambda x: '23456789TJQKA'.index(x))
    if v1 == v2:
        tipo = 'Pair'
    elif s1 == s2:
        tipo = 'Suited'
    else:
        tipo = 'Off-suited'
    return f"{cartas[0]}{cartas[1]}_{tipo}"

# Cargar modelo y codificadores
modelo = joblib.load(os.path.join("modelos","modelo_poker.pkl"))
le_pos = joblib.load(os.path.join("modelos","le_pos.pkl"))
le_type = joblib.load(os.path.join("modelos","le_type.pkl"))
le_action = joblib.load(os.path.join("modelos","le_action.pkl"))

def predecir_accion(c1, s1, c2, s2, posicion, bigblinds):
    valores = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
               '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    v1, v2 = valores[c1], valores[c2]
    tipo = 'Pair' if c1 == c2 else 'Suited' if s1 == s2 else 'Off-suited'
    entrada = [[v1, v2, le_type.transform([tipo])[0],
                le_pos.transform([posicion])[0], bigblinds]]
    
    probs = modelo.predict_proba(entrada)[0]
    pred_idx = np.argmax(probs)
    accion = le_action.inverse_transform([pred_idx])[0]
    confianza = probs[pred_idx]
    return normalizar_mano(c1, s1, c2, s2), tipo, accion, confianza, probs

# UI
st.markdown("## Selecciona tu mano")
df = cargar_datos()

col1, col2 = st.columns(2)
with col1:
    c1 = st.selectbox("Carta 1", list("23456789TJQKA"))
    s1 = st.selectbox("Palo Carta 1", ["♠", "♥", "♦", "♣"])
with col2:
    c2 = st.selectbox("Carta 2", list("23456789TJQKA"))
    s2 = st.selectbox("Palo Carta 2", ["♠", "♥", "♦", "♣"])

posicion = st.selectbox("Posición en la mesa", sorted(df["Position"].unique()))
bigblinds = st.slider("Big Blinds", min_value=1, max_value=200, value=50)

if st.button("🎯 Recomendar Acción"):
    reproducir_sonido()
    mano, tipo, recomendacion, confianza, probs = predecir_accion(c1, s1, c2, s2, posicion, bigblinds)

    st.markdown(f"### Mano: `{mano}`")
    st.markdown(f"""
    <div class='recomendacion-box'>
        <strong>Acción recomendada:</strong> <span style='color: gold;'>{recomendacion}</span><br>
        <small style='color: white;'>Confianza del modelo: {confianza*100:.1f}%</small>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar todas las probabilidades
    acciones_labels = le_action.inverse_transform(np.arange(len(probs)))
    probs_porcentajes = [round(p * 100, 2) for p in probs]
    df_probs = pd.DataFrame({
        "Acción": acciones_labels,
        "Probabilidad (%)": probs_porcentajes,
    }).sort_values(by="Probabilidad (%)", ascending=False)

    st.markdown("### 📊 Distribución de probabilidades del modelo")
    st.dataframe(df_probs, use_container_width=True)

    fig_probs = go.Figure(go.Bar(
    x=probs_porcentajes,
    y=acciones_labels,
    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    orientation='h',
    marker=dict(color='gold', line=dict(color='black', width=1)),
    text=[f"{p:.2f}%" for p in probs_porcentajes],
    textposition='auto'
    ))

    fig_probs.update_layout(
    title="📊 Distribución de probabilidades del modelo",
    xaxis_title="Probabilidad (%)",
    yaxis_title="Acción",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
    )

    st.plotly_chart(fig_probs, use_container_width=True)


if st.button("📊 Ver frecuencia de tipos de manos"):
    tipo_manos = df["HandType"].value_counts().reset_index()
    tipo_manos.columns = ["Tipo de Mano", "Frecuencia"]

    fig = px.pie(
        tipo_manos,
        names="Tipo de Mano",
        values="Frecuencia",
        title="Distribución de tipos de manos (preflop)",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    #### Clasificación de manos (preflop):
    - **Pair**: Pareja (ej. 9♠9♦)
    - **Suited**: Dos cartas del mismo palo (ej. A♠K♠)
    - **Off-suited**: Dos cartas de distinto palo (ej. Q♠J♦)
    > Esta clasificación no representa manos hechas (como full house o poker), sino la composición inicial preflop.
    """)

if st.button("🔝 Ver ranking de manos de poker"):
    st.markdown("""
    ### Ranking clásico de manos de poker (de mayor a menor):
    1. **Escalera Real** (A♠ K♠ Q♠ J♠ 10♠)
    2. **Escalera de Color** (5 cartas consecutivas del mismo palo)
    3. **Poker** (Cuatro cartas del mismo valor)
    4. **Full House** (Una trío + una pareja)
    5. **Color** (5 cartas del mismo palo sin orden)
    6. **Escalera** (5 cartas consecutivas de cualquier palo)
    7. **Trío** (Tres cartas del mismo valor)
    8. **Doble Pareja** (Dos pares diferentes)
    9. **Pareja** (Dos cartas del mismo valor)
    10. **Carta Alta** (Cuando no se forma nada)
    """)

st.markdown("---")

pos_frecuencia = st.selectbox("Selecciona una posición para ver el top 10 de manos", sorted(df["Position"].unique()), key="top_manos")
if st.button("📈 Ver top 10 combinaciones más frecuentes por posición"):
    df_filtrado = df[df["Position"] == pos_frecuencia]
    top_manos = df_filtrado["ManoNormalizada"].value_counts().head(10).reset_index()
    top_manos.columns = ["Mano", "Frecuencia"]
    fig = px.bar(top_manos, x="Mano", y="Frecuencia", color="Frecuencia",
                 color_continuous_scale="Bluered_r", title=f"Top 10 combinaciones más frecuentes en {pos_frecuencia}")
    fig.update_layout(xaxis_title="Mano Normalizada", yaxis_title="Frecuencia")
    st.plotly_chart(fig, use_container_width=True)

# Heatmap estilo matriz de manos
st.markdown("---")
st.markdown("## 🧩 Visualizar matriz de decisiones por posición 🧩")
posicion_matriz = st.selectbox("Elige una posición para mostrar la matriz de manos", sorted(df["Position"].unique()), key="heatmap")
if st.button("🧠 Generar matriz por modelo"):
    valores = list("AKQJT98765432")
    acciones = {}
    for i, r in enumerate(valores):
        for j, c in enumerate(valores):
            if i == j:
                mano = f"{r}{c}"
                tipo = 'Pair'
            elif i < j:
                mano = f"{c}{r}"
                tipo = 'Off-suited'
            else:
                mano = f"{r}{c}"
                tipo = 'Suited'
            v1, v2 = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}[r], {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}[c]
            entrada = [[v1, v2, le_type.transform([tipo])[0], le_pos.transform([posicion_matriz])[0], 50]]
            pred = modelo.predict(entrada)[0]
            accion = le_action.inverse_transform([pred])[0]
            acciones[(i, j)] = accion
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = {"Raise": "deepskyblue", "Limp": "gold", "Fold": "#222222"}
    for (i, j), accion in acciones.items():
        mano = f"{valores[i]}{valores[j]}" if i >= j else f"{valores[j]}{valores[i]}"
        color = cmap.get(accion, "gray")
        ax.add_patch(plt.Rectangle((j, 12 - i), 1, 1, color=color, ec="black"))
        ax.text(j + 0.5, 12 - i + 0.5, mano, ha='center', va='center', fontsize=8, color='black')
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Matriz de acciones para posición: {posicion_matriz}", fontsize=14)
    st.pyplot(fig)
