import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MercaIA",
    page_icon="🥦",
    layout="wide"
)

# ============================================================
# CARGAR MODELO Y ENCODERS
# ============================================================

@st.cache_resource
def cargar_modelo():
    if not os.path.exists('modelo_rf.pkl'):
        gdown.download(
            'https://drive.google.com/uc?id=1JuhjmtHpltMdkKE6ZmzXKEdQA4LoFEfs',
            'modelo_rf.pkl', quiet=False
        )
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
# MENÚ LATERAL
# ============================================================

with st.sidebar:
    st.markdown("""
        <h2 style='color: #1DB954;'>🥦 MercaIA</h2>
        <p style='color: #AAAAAA; font-size: 13px;'>Mercado Inteligente con IA</p>
        <hr style='border: 1px solid #1DB954;'>
    """, unsafe_allow_html=True)

    seccion = st.radio(
        "Navegación",
        ["🔍 Predicción de precios",
         "📊 Estadísticas del dataset",
         "📋 Acerca del proyecto"]
    )

    st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)
    st.markdown("""
        <p style='color: #555; font-size: 11px; text-align: center;'>
        Datos: SIPSA - DANE<br>
        Modelo: Random Forest<br><br>
        Torres · Castro · Seña · Arias
        </p>
    """, unsafe_allow_html=True)

# ============================================================
# SECCIÓN 1 — PREDICCIÓN DE PRECIOS
# ============================================================

if seccion == "🔍 Predicción de precios":

    st.markdown("""
        <h2 style='color: #1DB954;'>🔍 Predicción de Precios</h2>
        <p style='color: #AAAAAA;'>Consulta el precio estimado de alimentos básicos en mercados mayoristas de Colombia.</p>
        <hr style='border: 0.5px solid #333;'>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algoritmo", "Random Forest")
    col2.metric("R²", "0.70")
    col3.metric("MAE", "$1,409 COP/kg")
    col4.metric("Registros", "551,631")

    st.markdown("")
    st.markdown("#### Selecciona los parámetros")

    col_izq, col_der = st.columns(2)
    with col_izq:
        alimento     = st.selectbox("🥕 Alimento", sorted(encoders['le_alimento'].classes_))
        departamento = st.selectbox("🗺️ Departamento", sorted(encoders['le_depto'].classes_))
        municipio    = st.selectbox("📍 Municipio", sorted(encoders['le_municipio'].classes_))
    with col_der:
        mercado  = st.selectbox("🏪 Mercado mayorista", sorted(encoders['le_mercado'].classes_))
        grupo    = st.selectbox("📦 Grupo", sorted(encoders['le_grupo'].classes_))
        mes      = st.slider("📅 Mes", 1, 3, 3)
        semana   = st.slider("📆 Semana del año", 1, 14, 12)

    st.markdown("")
    predecir = st.button("⚡ Estimar precio", use_container_width=True)

    if predecir:
        alimento_cod     = encoders['le_alimento'].transform([alimento])[0]
        departamento_cod = encoders['le_depto'].transform([departamento])[0]
        municipio_cod    = encoders['le_municipio'].transform([municipio])[0]
        mercado_cod      = encoders['le_mercado'].transform([mercado])[0]
        grupo_cod        = encoders['le_grupo'].transform([grupo])[0]

        entrada = pd.DataFrame([[alimento_cod, departamento_cod, municipio_cod,
                                 mercado_cod, grupo_cod, mes, semana]],
                               columns=['Alimento_cod', 'Departamento_cod', 'Municipio_cod',
                                        'Mercado_cod', 'Grupo_cod', 'Mes', 'Semana'])
        precio = modelo.predict(entrada)[0]

        st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)
        st.markdown("#### 💰 Resultado")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Alimento", alimento)
        col_r2.metric("Ubicación", f"{municipio}, {departamento}")
        col_r3.metric("Precio estimado", f"${precio:,.0f} COP/kg")

        st.success(f"El precio estimado de **{alimento}** en **{municipio}, {departamento}** para la semana {semana} del mes {mes} es de **${precio:,.0f} COP/kg**.")
        st.caption("⚠️ Precios de mercados mayoristas. Pueden diferir del precio al consumidor final.")

        # gráfica variación por semana.
        st.markdown("#### 📈 Variación estimada por semana")
        semanas = list(range(1, 15))
        precios_semanas = []
        for s in semanas:
            e = pd.DataFrame([[alimento_cod, departamento_cod, municipio_cod,
                               mercado_cod, grupo_cod, mes, s]],
                             columns=['Alimento_cod', 'Departamento_cod', 'Municipio_cod',
                                      'Mercado_cod', 'Grupo_cod', 'Mes', 'Semana'])
            precios_semanas.append(modelo.predict(e)[0])

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0F1923')
        ax.set_facecolor('#0F1923')
        ax.plot(semanas, precios_semanas, color='#1DB954', linewidth=2.5, marker='o', markersize=5)
        ax.axvline(x=semana, color='#FF4B4B', linestyle='--', linewidth=1.5, label=f'Semana seleccionada: {semana}')
        ax.set_xlabel("Semana del año", color='white')
        ax.set_ylabel("Precio estimado (COP/kg)", color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1C2B3A', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig)

        # gráfica variación por mes.
        st.markdown("#### 📊 Comparación de precio por mes")
        meses_nombres = ['Ene', 'Feb', 'Mar']
        precios_meses = []
        for m in range(1, 4):
            e = pd.DataFrame([[alimento_cod, departamento_cod, municipio_cod,
                               mercado_cod, grupo_cod, m, 7]],
                             columns=['Alimento_cod', 'Departamento_cod', 'Municipio_cod',
                                      'Mercado_cod', 'Grupo_cod', 'Mes', 'Semana'])
            precios_meses.append(modelo.predict(e)[0])

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor('#0F1923')
        ax2.set_facecolor('#0F1923')
        colores = ['#1DB954' if m + 1 != mes else '#FF4B4B' for m in range(3)]
        ax2.bar(meses_nombres, precios_meses, color=colores)
        ax2.set_xlabel("Mes", color='white')
        ax2.set_ylabel("Precio estimado (COP/kg)", color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig2)

# ============================================================
# SECCIÓN 2 — ESTADÍSTICAS DEL DATASET
# ============================================================

elif seccion == "📊 Estadísticas del dataset":

    st.markdown("""
        <h2 style='color: #1DB954;'>📊 Estadísticas del Dataset</h2>
        <p style='color: #AAAAAA;'>Resumen del dataset utilizado para entrenar el modelo.</p>
        <hr style='border: 0.5px solid #333;'>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total registros", "572,350")
    col2.metric("Registros válidos", "551,631")
    col3.metric("Departamentos", "32")
    col4.metric("Alimentos únicos", "192")

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🗓️ Cobertura temporal")
        st.info("Enero 2026 — Marzo 2026\n\n**3 meses** de registros de precios en mercados mayoristas de Colombia.")

        st.markdown("#### 📦 Grupos de alimentos")
        grupos = ['Verduras y Hortalizas', 'Tubérculos y Plátanos',
                  'Frutas', 'Carnes', 'Granos y Cereales',
                  'Lácteos', 'Huevos', 'Otros']
        valores = [245205, 130000, 80000, 50000, 30000, 10000, 5000, 1631]
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor('#0F1923')
        ax3.set_facecolor('#0F1923')
        ax3.barh(grupos, valores, color='#1DB954')
        ax3.set_xlabel("Registros", color='white')
        ax3.tick_params(colors='white')
        for spine in ax3.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig3)

    with col_b:
        st.markdown("#### 💰 Distribución de precios")
        precios_ref = [100, 600, 1500, 3424, 4800, 40000]
        etiquetas   = ['Mín', 'Q25', 'Mediana', 'Promedio', 'Q75', 'Máx']
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        fig4.patch.set_facecolor('#0F1923')
        ax4.set_facecolor('#0F1923')
        colores_bar = ['#555', '#1DB954', '#1DB954', '#FF4B4B', '#1DB954', '#555']
        ax4.bar(etiquetas, precios_ref, color=colores_bar)
        ax4.set_ylabel("COP/kg", color='white')
        ax4.tick_params(colors='white')
        for spine in ax4.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig4)

        st.markdown("#### 🏆 Alimentos más frecuentes")
        alimentos_top = {
            'Plátano hartón verde': 22406,
            'Tomate chonto': 21764,
            'Arveja verde': 18364,
            'Papa superior': 16536,
            'Papa criolla': 15744
        }
        for alimento_nombre, cantidad in alimentos_top.items():
            st.markdown(f"**{alimento_nombre}** — {cantidad:,} registros")

# ============================================================
# SECCIÓN 3 — ACERCA DEL PROYECTO
# ============================================================

elif seccion == "📋 Acerca del proyecto":

    st.markdown("""
        <h2 style='color: #1DB954;'>📋 Acerca del Proyecto</h2>
        <hr style='border: 0.5px solid #333;'>
    """, unsafe_allow_html=True)

    col_info, col_equipo = st.columns(2)

    with col_info:
        st.markdown("#### 🎯 Objetivo")
        st.write("Desarrollar un modelo de inteligencia artificial capaz de predecir el precio de alimentos básicos en mercados mayoristas de Colombia, con el fin de apoyar la toma de decisiones de consumidores y pequeños comerciantes.")

        st.markdown("#### 🌍 Relación con ODS")
        st.success("ODS 2 — Hambre Cero\n\n**Meta 2.c:** Facilitar el acceso oportuno a información sobre los mercados alimentarios para limitar la extrema volatilidad de los precios.")

        st.markdown("#### 🤖 Modelo utilizado")
        st.info("**Random Forest Regressor**\n\n- 100 árboles de decisión\n- R² = 0.70\n- MAE = $1,409 COP/kg\n- RMSE = $2,669 COP/kg\n- Entrenado con 441,304 registros")

        st.markdown("#### ⚠️ Limitaciones")
        st.warning("- Dataset limitado a enero–marzo 2026\n- Precios de mercados mayoristas únicamente\n- Con más datos históricos la precisión mejoraría significativamente")

    with col_equipo:
        st.markdown("#### 👥 Equipo")
        integrantes = [
            ("Deimer Torres", "Líder del proyecto"),
            ("Maria Palmet Castro", "Analista de datos"),
            ("Harley Seña", "Desarrollador del sistema inteligente"),
            ("Joseph Arias", "Desarrollador de interfaz")
        ]
        for nombre, rol in integrantes:
            st.markdown(f"**{nombre}**\n\n_{rol}_")
            st.markdown("---")

        st.markdown("#### 🔗 Enlaces")
        st.markdown("🌐 [Aplicación web](https://predictor-sipsa.streamlit.app)")
        st.markdown("💻 [Código fuente](https://github.com/Deimer0819/sipsa-app)")
        st.markdown("📂 [Dataset SIPSA - DANE](https://microdatos.dane.gov.co/index.php/catalog/697/get-microdata)")