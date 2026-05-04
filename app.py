import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os

# ============================================================
# CARGAR MODELO Y ENCODERS DESDE GOOGLE DRIVE
# ============================================================

@st.cache_resource
def cargar_modelo():
    # descargar modelo si no existe localmente.
    if not os.path.exists('modelo_rf.pkl'):
        gdown.download(
            'https://drive.google.com/uc?id=1JuhjmtHpltMdkKE6ZmzXKEdQA4LoFEfs',
            'modelo_rf.pkl', quiet=False
        )
    # descargar encoders si no existen localmente.
    if not os.path.exists('encoders.pkl'):
        gdown.download(
            'https://drive.google.com/uc?id=1E31U1Bv3l3nQJezn-Ix3sNC7x_seL5zO',
            'encoders.pkl', quiet=False
        )
    with open('modelo_rf.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return modelo, encoders

modelo, encoders = cargar_modelo()

# ============================================================
# INTERFAZ
# ============================================================

st.title("🥦 Predictor de Precios de Alimentos")
st.markdown("Consulta el precio estimado de alimentos básicos en Colombia según el mercado, departamento y fecha.")
st.divider()

# selectores.
alimento     = st.selectbox("Alimento", encoders['le_alimento'].classes_)
departamento = st.selectbox("Departamento", encoders['le_depto'].classes_)
municipio    = st.selectbox("Municipio", encoders['le_municipio'].classes_)
mercado      = st.selectbox("Mercado mayorista", encoders['le_mercado'].classes_)
grupo        = st.selectbox("Grupo", encoders['le_grupo'].classes_)
mes          = st.slider("Mes", 1, 12, 2)
semana       = st.slider("Semana del año", 1, 52, 8)

st.divider()

# botón de predicción.
if st.button("Estimar precio"):
    # codificar entradas.
    alimento_cod     = encoders['le_alimento'].transform([alimento])[0]
    departamento_cod = encoders['le_depto'].transform([departamento])[0]
    municipio_cod    = encoders['le_municipio'].transform([municipio])[0]
    mercado_cod      = encoders['le_mercado'].transform([mercado])[0]
    grupo_cod        = encoders['le_grupo'].transform([grupo])[0]

    # construir entrada del modelo.
    entrada = pd.DataFrame([[alimento_cod, departamento_cod, municipio_cod,
                             mercado_cod, grupo_cod, mes, semana]],
                           columns=['Alimento_cod', 'Departamento_cod', 'Municipio_cod',
                                    'Mercado_cod', 'Grupo_cod', 'Mes', 'Semana'])
    # predecir.
    precio = modelo.predict(entrada)[0]

    st.success(f"💰 Precio estimado: **${precio:,.0f} COP/kg**")
    st.caption(f"{alimento} en {municipio}, {departamento} — semana {semana}, mes {mes}")