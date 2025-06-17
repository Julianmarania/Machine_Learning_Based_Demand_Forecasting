import streamlit as st
import os
from datetime import datetime
import pandas as pd
import joblib
import re
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import matplotlib.pyplot as plt

##############################
# Paso 0: Configuración inicial
##############################

# Determinar el directorio base de manera robusta
current_dir = os.path.abspath(os.path.dirname(__file__))

# Función para cargar datos con caché
@st.cache_data
def load_data():
    csv_path = os.path.join(current_dir, "..", "data", "raw", "Final_XGBoost_data_processed.csv")
    try:
        data = pd.read_csv(csv_path)
        # Convertir la columna 'month' de timestamp a número de mes
        if 'month' in data.columns:
            data['month'] = pd.to_datetime(data['month']).dt.month
        return data
    except FileNotFoundError:
        st.error("No se encontró el archivo CSV. Asegúrate de que esté en la ruta correcta.")
        return None

# Función para cargar modelo con caché
@st.cache_resource
def load_model(product_id, store_id):
    model_path = os.path.join(current_dir, "..", "models", f"Final_model_XGBOOST_{product_id}_{store_id}.sav")
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo en: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# Función para obtener productos y sus tiendas asociadas desde los nombres de los modelos
@st.cache_data
def get_available_models():
    model_dir = os.path.join(current_dir, "..", "models")
    product_to_stores = {}
    products = set()
    model_pattern = re.compile(r"Final_model_XGBOOST_(.+)_([A-Z]{2}_\d+)\.sav")
    
    try:
        for filename in os.listdir(model_dir):
            match = model_pattern.match(filename)
            if match:
                product_id = match.group(1)
                store_id = match.group(2)
                products.add(product_id)
                if product_id not in product_to_stores:
                    product_to_stores[product_id] = set()
                product_to_stores[product_id].add(store_id)
        return sorted(products), product_to_stores
    except FileNotFoundError:
        st.error(f"No se encontró la carpeta de modelos en: {model_dir}")
        return [], {}
    except Exception as e:
        st.error(f"Error al escanear modelos: {str(e)}")
        return [], {}

##################################
# Paso 1: Autenticación con nombre
##################################

# Mostrar logo de Walmart al inicio
walmart_logo_path = os.path.join(current_dir, "..", "data", "raw", "Walmart_logo.svg.png")
if os.path.exists(walmart_logo_path):
    st.image(walmart_logo_path, use_container_width=True)
else:
    st.error("No se encontró la imagen del logo de Walmart.")

# Inicializar el estado de la sesión para la autenticación
if "username" not in st.session_state:
    st.session_state["username"] = None
if "login_time" not in st.session_state:
    st.session_state["login_time"] = None
if "selected_state_full" not in st.session_state:
    st.session_state["selected_state_full"] = None

# Si el usuario no ha ingresado un nombre, mostrar el formulario de ingreso
if st.session_state["username"] is None:
    st.markdown("<h3 style='text-align: center;'>Bienvenido!</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Por favor, introduce tu nombre para continuar:</p>", unsafe_allow_html=True)
    
    username = st.text_input("Nombre de usuario", key="username_input")
    
    if st.button("Ingresar"):
        if username.strip() == "":
            st.error("Por favor, ingresa un nombre válido.")
        else:
            # Registrar el nombre y la hora de ingreso
            st.session_state["username"] = username
            st.session_state["login_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

# Si el usuario ya ingresó un nombre, mostrar el resto de la app
if st.session_state["username"] is not None:
    # Mostrar mensaje de bienvenida con nombre, fecha y hora
    st.markdown(
        f"""
        <p style='text-align: center; color: #0071CE;'>
            Bienvenido, {st.session_state["username"]}. Has ingresado el {st.session_state["login_time"]}.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Desplegable para seleccionar modo
    mode = st.selectbox(
        "Selecciona el modo:",
        ["Predicción de la demanda", "Informe de la predicción"],
        key="mode_select"
    )

    if mode == "Predicción de la demanda":
        ##############################
        # Paso 2: Añadir título
        ##############################

        st.markdown(
            """
            <h1 style='color: #0071CE; text-align: center; white-space: nowrap; margin-top: 0; margin-bottom: 20px;'>
                Predicción de la demanda
            </h1>
            """,
            unsafe_allow_html=True
        )

        ################################
        # Paso 3: Sidebar para selección
        ################################

        # Obtener productos y mapeo de productos a tiendas desde los nombres de los modelos
        products, product_to_stores = get_available_models()
        if not products or not product_to_stores:
            st.error("No se encontraron modelos válidos. Verifica la carpeta de modelos.")
            st.stop()

        # Mapa de estados
        state_mapping = {
            'CA': 'California',
            'TX': 'Texas',
            'WI': 'Wisconsin'
        }

        # Crear el sidebar
        st.sidebar.header("Parámetros a predecir")

        # Botón para cerrar sesión
        if st.sidebar.button("Cerrar sesión"):
            st.session_state["username"] = None
            st.session_state["login_time"] = None
            st.session_state["selected_state_full"] = None
            st.rerun()

        # Selección de producto
        selected_product = st.sidebar.selectbox("Selecciona un producto:", products, key="product_select")

        # Obtener tiendas disponibles para el producto seleccionado
        available_stores = sorted(product_to_stores.get(selected_product, []))
        available_stores_display = [f"Tienda {store.split('_')[1]}" for store in available_stores]
        
        # Selección de tienda (formato "Tienda X")
        selected_store_display = st.sidebar.selectbox("Selecciona una tienda:", available_stores_display, key="store_select") if available_stores_display else None
        selected_store = available_stores[available_stores_display.index(selected_store_display)] if selected_store_display else None

        # Mostrar el estado de la tienda seleccionada
        if selected_store:
            store_state_code = selected_store.split('_')[0]
            store_state_full = state_mapping.get(store_state_code, "Desconocido")
            st.sidebar.write(f"Estado: {store_state_full}")
        else:
            st.sidebar.warning("No hay tiendas disponibles para el producto seleccionado.")
            st.sidebar.write("Estado: No seleccionado")

        # Selección de mes y año (fijado a Mayo de 2016)
        st.sidebar.subheader("Período de predicción")
        selected_month = st.sidebar.selectbox("Mes", ["Mayo"], disabled=True, key="month_select")
        selected_year = st.sidebar.selectbox("Año", [2016], disabled=True, key="year_select")

        ##################################
        # Paso 4: Mapa interactivo de EEUU
        ##################################

        if selected_store:
            # Determinar el estado seleccionado a partir de la tienda
            store_state_code = selected_store.split('_')[0]
            selected_state_full = state_mapping.get(store_state_code, "Desconocido")

            # Crear mapa interactivo con Plotly
            us_states = [
                {'name': 'Alabama', 'code': 'AL'}, {'name': 'Alaska', 'code': 'AK'}, {'name': 'Arizona', 'code': 'AZ'},
                {'name': 'Arkansas', 'code': 'AR'}, {'name': 'California', 'code': 'CA'}, {'name': 'Colorado', 'code': 'CO'},
                {'name': 'Connecticut', 'code': 'CT'}, {'name': 'Delaware', 'code': 'DE'}, {'name': 'Florida', 'code': 'FL'},
                {'name': 'Georgia', 'code': 'GA'}, {'name': 'Hawaii', 'code': 'HI'}, {'name': 'Idaho', 'code': 'ID'},
                {'name': 'Illinois', 'code': 'IL'}, {'name': 'Indiana', 'code': 'IN'}, {'name': 'Iowa', 'code': 'IA'},
                {'name': 'Kansas', 'code': 'KS'}, {'name': 'Kentucky', 'code': 'KY'}, {'name': 'Louisiana', 'code': 'LA'},
                {'name': 'Maine', 'code': 'ME'}, {'name': 'Maryland', 'code': 'MD'}, {'name': 'Massachusetts', 'code': 'MA'},
                {'name': 'Michigan', 'code': 'MI'}, {'name': 'Minnesota', 'code': 'MN'}, {'name': 'Mississippi', 'code': 'MS'},
                {'name': 'Missouri', 'code': 'MO'}, {'name': 'Montana', 'code': 'MT'}, {'name': 'Nebraska', 'code': 'NE'},
                {'name': 'Nevada', 'code': 'NV'}, {'name': 'New Hampshire', 'code': 'NH'}, {'name': 'New Jersey', 'code': 'NJ'},
                {'name': 'New Mexico', 'code': 'NM'}, {'name': 'New York', 'code': 'NY'}, {'name': 'North Carolina', 'code': 'NC'},
                {'name': 'North Dakota', 'code': 'ND'}, {'name': 'Ohio', 'code': 'OH'}, {'name': 'Oklahoma', 'code': 'OK'},
                {'name': 'Oregon', 'code': 'OR'}, {'name': 'Pennsylvania', 'code': 'PA'}, {'name': 'Rhode Island', 'code': 'RI'},
                {'name': 'South Carolina', 'code': 'SC'}, {'name': 'South Dakota', 'code': 'SD'}, {'name': 'Tennessee', 'code': 'TN'},
                {'name': 'Texas', 'code': 'TX'}, {'name': 'Utah', 'code': 'UT'}, {'name': 'Vermont', 'code': 'VT'},
                {'name': 'Virginia', 'code': 'VA'}, {'name': 'Washington', 'code': 'WA'}, {'name': 'West Virginia', 'code': 'WV'},
                {'name': 'Wisconsin', 'code': 'WI'}, {'name': 'Wyoming', 'code': 'WY'}
            ]
            state_df = pd.DataFrame(us_states)
            state_df['color'] = state_df['name'].apply(lambda x: 'Seleccionado' if x == selected_state_full else 'No seleccionado')

            # Crear el mapa base con px.choropleth
            fig = px.choropleth(
                state_df,
                locations='code',
                locationmode="USA-states",
                color='color',
                scope="usa",
                color_discrete_map={
                    'Seleccionado': '#0071CE',
                    'No seleccionado': '#E0E0E0'
                },
                labels={'color': 'Estado'},
                title=f"Estado Seleccionado: {selected_state_full}"
            )
            fig.update_layout(
                plot_bgcolor='rgba(240, 240, 240, 1)',
                paper_bgcolor='rgba(240, 240, 240, 1)',
                font=dict(color='black', family="Arial, sans-serif"),
                title=dict(
                    font=dict(size=20),
                    x=0.5,
                    xanchor='center'
                ),
                geo=dict(
                    bgcolor='rgba(240, 240, 240, 1)',
                    lakecolor='rgba(255, 255, 255, 1)',
                    landcolor='rgba(200, 200, 200, 0.5)',
                    subunitcolor='rgba(0, 0, 0, 0.8)',
                    showlakes=True,
                    showsubunits=True,
                    showland=True
                ),
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig.update_traces(showscale=False)

            # Mostrar el mapa sin interacción de clics
            st.plotly_chart(fig, use_container_width=True, key="map")


        ##################################
        # Paso 5: Cargar modelo y predecir
        ##################################

        if selected_store:
            # Cargar el modelo correspondiente
            model = load_model(selected_product, selected_store)
            if model is not None:
                # Cargar datos del CSV
                data = load_data()
                if data is None:
                    st.stop()

                # Preparar datos para la predicción (Mayo de 2016)
                try:
                    # Filtrar datos para la tienda y producto seleccionados
                    filtered_data = data[(data['store_id'] == selected_store) & (data['item_id'] == selected_product)].copy()
                    
                    if filtered_data.empty:
                        st.warning("No se encontraron datos para la combinación de tienda y producto seleccionada en el CSV.")
                    else:
                        # Definir las columnas de características esperadas por el modelo
                        feature_columns = [
                            'event_name_1', 'snap', 'sell_price', 'lag_1', 'lag_2', 'lag_3', 
                            'lag_6', 'lag_12', 'rolling_mean_3', 'year', 
                            'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 
                            'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
                        ]
                        
                        # Verificar que todas las columnas requeridas estén presentes
                        missing_columns = [col for col in feature_columns if col not in filtered_data.columns]
                        if missing_columns:
                            st.error(f"Faltan las siguientes columnas en los datos: {missing_columns}")
                            st.stop()
                        
                        # Preparar datos para la predicción de Mayo 2016
                        last_row = filtered_data.iloc[-1]
                        prediction_data = pd.DataFrame({
                            'event_name_1': [last_row['event_name_1']],
                            'snap': [last_row['snap']],
                            'sell_price': [last_row['sell_price']],
                            'lag_1': [last_row['sales']],
                            'lag_2': [filtered_data['sales'].iloc[-2] if len(filtered_data) >= 2 else filtered_data['sales'].mean()],
                            'lag_3': [filtered_data['sales'].iloc[-3] if len(filtered_data) >= 3 else filtered_data['sales'].mean()],
                            'lag_6': [filtered_data['sales'].iloc[-6] if len(filtered_data) >= 6 else filtered_data['sales'].mean()],
                            'lag_12': [filtered_data['sales'].iloc[-12] if len(filtered_data) >= 12 else filtered_data['sales'].mean()],
                            'rolling_mean_3': [filtered_data['sales'].tail(3).mean()],
                            'year': [2016],
                            'month_1': [0], 'month_2': [0], 'month_3': [0], 'month_4': [0], 'month_5': [0],
                            'month_6': [1], 'month_7': [0], 'month_8': [0], 'month_9': [0], 'month_10': [0],
                            'month_11': [0], 'month_12': [0]
                        })
                        
                        # Realizar la predicción para Mayo 2016
                        predicted_log = model.predict(prediction_data[feature_columns])[0]
                        predicted_demand = np.expm1(predicted_log)
                        
                        # Obtener demanda real (usamos el último mes disponible, e.g., Mayo 2016)
                        real_demand = last_row['sales']
                        
                        # Crear gráfico de barras 2D con efecto pseudo-3D
                        fig = go.Figure()

                        # Definir posiciones y dimensiones para las barras
                        bar_width = 0.3
                        depth_offset = 0.05 * max(real_demand, predicted_demand)

                        # Barra principal para demanda real (cara frontal)
                        fig.add_trace(go.Bar(
                            x=[0],
                            y=[real_demand],
                            name='Demanda Real (Mayo 2016)',
                            marker_color='rgb(55, 83, 109)',
                            marker_line_color='rgb(8, 48, 107)',
                            marker_line_width=2,
                            opacity=0.9,
                            width=bar_width,
                            text=[f"{real_demand:.2f}"],
                            textposition='auto'
                        ))

                        # Cara superior para demanda real
                        fig.add_trace(go.Scatter(
                            x=[-bar_width / 2, bar_width / 2, bar_width / 2 + 0.1, -bar_width / 2 + 0.1, -bar_width / 2],
                            y=[real_demand, real_demand, real_demand + depth_offset, real_demand + depth_offset, real_demand],
                            mode='lines',
                            fill='toself',
                            fillcolor='rgb(75, 103, 129)',
                            line=dict(color='rgb(8, 48, 107)', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        # Cara lateral para demanda real
                        fig.add_trace(go.Scatter(
                            x=[bar_width / 2, bar_width / 2 + 0.1, bar_width / 2 + 0.1, bar_width / 2, bar_width / 2],
                            y=[0, 0, real_demand + depth_offset, real_demand, 0],
                            mode='lines',
                            fill='toself',
                            fillcolor='rgb(45, 63, 89)',
                            line=dict(color='rgb(8, 48, 107)', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        # Barra principal para demanda predicha
                        fig.add_trace(go.Bar(
                            x=[1],
                            y=[predicted_demand],
                            name='Demanda Predicha (Mayo 2016)',
                            marker_color='rgb(255, 140, 0)',
                            marker_line_color='rgb(204, 102, 0)',
                            marker_line_width=2,
                            opacity=0.9,
                            width=bar_width,
                            text=[f"{predicted_demand:.2f}"],
                            textposition='auto'
                        ))

                        # Cara superior para demanda predicha
                        fig.add_trace(go.Scatter(
                            x=[1 - bar_width / 2, 1 + bar_width / 2, 1 + bar_width / 2 + 0.1, 1 - bar_width / 2 + 0.1, 1 - bar_width / 2],
                            y=[predicted_demand, predicted_demand, predicted_demand + depth_offset, predicted_demand + depth_offset, predicted_demand],
                            mode='lines',
                            fill='toself',
                            fillcolor='rgb(255, 160, 50)',
                            line=dict(color='rgb(204, 102, 0)', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        # Cara lateral para demanda predicha
                        fig.add_trace(go.Scatter(
                            x=[1 + bar_width / 2, 1 + bar_width / 2 + 0.1, 1 + bar_width / 2 + 0.1, 1 + bar_width / 2, 1 + bar_width / 2],
                            y=[0, 0, predicted_demand + depth_offset, predicted_demand, 0],
                            mode='lines',
                            fill='toself',
                            fillcolor='rgb(215, 110, 0)',
                            line=dict(color='rgb(204, 102, 0)', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        # Actualizar el diseño
                        fig.update_layout(
                            title={
                                'text': "Demanda Real vs. Predicha para Mayo 2016",
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'font': dict(size=20, color='white', family="Arial, sans-serif")
                            },
                            xaxis_title="Mes",
                            yaxis_title="Demanda",
                            barmode='group',
                            bargap=0.2,
                            bargroupgap=0.1,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(30,30,60,1)',
                            font=dict(color='white', family="Arial, sans-serif"),
                            xaxis=dict(
                                title=dict(text="Mes", font=dict(size=16)),
                                tickfont=dict(size=14),
                                tickvals=[0, 1],
                                ticktext=['Demanda Real', 'Demanda Predicha'],
                                gridcolor='rgba(255,255,255,0.1)'
                            ),
                            yaxis=dict(
                                title=dict(text="Demanda", font=dict(size=16)),
                                tickfont=dict(size=14),
                                gridcolor='rgba(255,255,255,0.1)',
                                range=[0, max(real_demand, predicted_demand) * 1.2]
                            ),
                            legend=dict(
                                x=0.75,
                                y=1.1,
                                font=dict(size=12, color='white'),
                                bgcolor='rgba(0,0,0,0)',
                                orientation='h'
                            ),
                            height=600,
                            margin=dict(l=50, r=50, t=100, b=50)
                        )

                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error al realizar la predicción: {str(e)}")
        else:
            st.info("Por favor, selecciona una tienda para realizar la predicción.")

    elif mode == "Informe de la predicción":
        ##################################
        # Paso 6: Informe de la predicción
        ##################################

        st.markdown(
            """
            <h1 style='color: #0071CE; text-align: center; white-space: nowrap; margin-top: 0; margin-bottom: 20px;'>
                Informe de la predicción
            </h1>
            """,
            unsafe_allow_html=True
        )

        # Obtener productos y mapeo de productos a tiendas
        products, product_to_stores = get_available_models()
        if not products or not product_to_stores:
            st.error("No se encontraron modelos válidos. Verifica la carpeta de modelos.")
            st.stop()

        # Mapa de estados
        state_mapping = {
            'CA': 'California',
            'TX': 'Texas',
            'WI': 'Wisconsin'
        }
        all_states = sorted(state_mapping.values())

        # Selección de estado
        st.subheader("Selecciona estado, tiendas y productos para el informe")
        selected_state_full = st.selectbox(
            "Estado:",
            all_states,
            key="report_state"
        )

        # Filtrar tiendas disponibles para el estado seleccionado
        state_code = next((code for code, name in state_mapping.items() if name == selected_state_full), None)
        available_stores = []
        store_to_id = {}
        if state_code:
            for product in products:
                for store in product_to_stores.get(product, []):
                    if store.startswith(state_code):
                        store_number = store.split('_')[1]
                        store_display = f"Tienda {store_number} ({selected_state_full})"
                        if store_display not in available_stores:
                            available_stores.append(store_display)
                            store_to_id[store_display] = store
        available_stores = sorted(available_stores)

        # Selección de tiendas
        selected_stores_display = st.multiselect(
            "Tiendas:",
            available_stores,
            key="report_stores"
        )
        selected_stores = [store_to_id[store_display] for store_display in selected_stores_display]

        # Filtrar productos disponibles para las tiendas seleccionadas
        available_products = set()
        for store in selected_stores:
            for product in products:
                if store in product_to_stores.get(product, []):
                    available_products.add(product)
        available_products = sorted(available_products)

        # Selección de productos
        selected_products = st.multiselect(
            "Productos:",
            available_products,
            key="report_products"
        )

        if st.button("Generar Informe"):
            if not selected_state_full or not selected_stores or not selected_products:
                st.error("Por favor, selecciona un estado, al menos una tienda y al menos un producto.")
            else:
                import tempfile
                import time
                import matplotlib.pyplot as plt

                # Crear directorio temporal para imágenes
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Crear buffer para el PDF
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
                    story = []
                    styles = getSampleStyleSheet()

                    # Estilos personalizados
                    title_style = ParagraphStyle(
                        'Title',
                        parent=styles['Title'],
                        fontSize=16,
                        textColor=colors.HexColor('#0071CE'),
                        spaceAfter=20,
                        alignment=1  # Centrado
                    )
                    heading_style = ParagraphStyle(
                        'Heading2',
                        parent=styles['Heading2'],
                        fontSize=12,
                        spaceAfter=10
                    )
                    normal_style = styles['Normal']

                    # Título del informe
                    story.append(Paragraph(f"Informe de predicciones - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", title_style))
                    story.append(Paragraph(f"Usuario: {st.session_state['username']}", normal_style))
                    story.append(Spacer(1, 12))

                    # Tabla de predicciones
                    table_data = [['Producto', 'Estado', 'Tienda', 'Demanda real (mayo 2016)', 'Demanda predicha (Mayo 2016)', '% Error']]
                    image_paths = []

                    for product in selected_products:
                        for store in selected_stores:
                            # Verificar si el producto se vende en la tienda
                            if store not in product_to_stores.get(product, []):
                                continue  # Ignorar combinación inválida

                            # Verificar que la tienda pertenece al estado seleccionado
                            store_state_code = store.split('_')[0]
                            if store_state_code != state_code:
                                continue  # Ignorar si la tienda no coincide con el estado

                            # Cargar modelo
                            model = load_model(product, store)
                            if model is None:
                                continue

                            # Cargar datos
                            data = load_data()
                            if data is None:
                                continue

                            # Preparar datos para la predicción
                            filtered_data = data[(data['store_id'] == store) & (data['item_id'] == product)].copy()
                            if filtered_data.empty:
                                continue

                            # Definir columnas de características
                            feature_columns = [
                                'event_name_1', 'snap', 'sell_price', 'lag_1', 'lag_2', 'lag_3',
                                'lag_6', 'lag_12', 'rolling_mean_3', 'year',
                                'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
                                'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
                            ]
                            missing_columns = [col for col in feature_columns if col not in filtered_data.columns]
                            if missing_columns:
                                continue

                            # Preparar datos para predicción
                            last_row = filtered_data.iloc[-1]
                            prediction_data = pd.DataFrame({
                                'event_name_1': [last_row['event_name_1']],
                                'snap': [last_row['snap']],
                                'sell_price': [last_row['sell_price']],
                                'lag_1': [last_row['sales']],
                                'lag_2': [filtered_data['sales'].iloc[-2] if len(filtered_data) >= 2 else filtered_data['sales'].mean()],
                                'lag_3': [filtered_data['sales'].iloc[-3] if len(filtered_data) >= 3 else filtered_data['sales'].mean()],
                                'lag_6': [filtered_data['sales'].iloc[-6] if len(filtered_data) >= 6 else filtered_data['sales'].mean()],
                                'lag_12': [filtered_data['sales'].iloc[-12] if len(filtered_data) >= 12 else filtered_data['sales'].mean()],
                                'rolling_mean_3': [filtered_data['sales'].tail(3).mean()],
                                'year': [2016],
                                'month_1': [0], 'month_2': [0], 'month_3': [0], 'month_4': [0], 'month_5': [0],
                                'month_6': [1], 'month_7': [0], 'month_8': [0], 'month_9': [0], 'month_10': [0],
                                'month_11': [0], 'month_12': [0]
                            })

                            # Realizar predicción
                            try:
                                predicted_log = model.predict(prediction_data[feature_columns])[0]
                                predicted_demand = np.expm1(predicted_log)
                                real_demand = last_row['sales']
                            except Exception:
                                continue

                            # Calcular % de error
                            error_percent = abs(real_demand - predicted_demand) / real_demand * 100 if real_demand != 0 else 0

                            # Añadir a la tabla
                            table_data.append([
                                product,
                                selected_state_full,
                                f"Tienda {store.split('_')[1]}",
                                f"{real_demand:.2f}",
                                f"{predicted_demand:.2f}",
                                f"{error_percent:.2f}%"
                            ])

                            # Generar gráfico con matplotlib
                            fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1E1E3C')
                            ax.set_facecolor('#1E1E3C')
                            bar_width = 0.35
                            x = [0, 1]
                            bars = ax.bar([i - bar_width/2 for i in x], [real_demand, predicted_demand], 
                                         bar_width, color=['#37536D', '#FF8C00'], edgecolor='#08306B')
                            
                            # Añadir texto encima de las barras
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.2f}', ha='center', va='bottom', color='white')

                            # Configurar ejes
                            ax.set_xticks(x)
                            ax.set_xticklabels(['Demanda real', 'Demanda predicha'], color='white')
                            ax.set_ylabel('Demanda', color='white')
                            ax.set_title(f'Demanda Real vs. Predicha - {product}, {selected_state_full}, Tienda {store.split("_")[1]}', 
                                        color='white', pad=20)
                            ax.tick_params(axis='y', colors='white')
                            ax.spines['bottom'].set_color('white')
                            ax.spines['left'].set_color('white')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)

                            # Guardar gráfico como imagen
                            img_path = os.path.join(tmpdirname, f"plot_{product}_{store}_{len(image_paths)}.png")
                            try:
                                plt.savefig(img_path, format='png', bbox_inches='tight', facecolor='#1E1E3C', edgecolor='none')
                                time.sleep(1)
                                if os.path.exists(img_path):
                                    image_paths.append(img_path)
                                else:
                                    st.warning(f"No se generó la imagen para {product}, {selected_state_full}, Tienda {store.split('_')[1]}")
                            except Exception as e:
                                st.warning(f"Error al generar imagen para {product}, {selected_state_full}, Tienda {store.split('_')[1]}: {str(e)}")
                            finally:
                                plt.close(fig)

                    # Crear tabla en PDF
                    if len(table_data) > 1:  # Si hay datos
                        table = Table(table_data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0071CE')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                        ]))
                        story.append(Paragraph("Resumen de predicciones", heading_style))
                        story.append(table)
                        story.append(Spacer(1, 12))

                        # Añadir gráficos
                        if image_paths:
                            story.append(Paragraph("Gráficos de Demanda", heading_style))
                            for img_path in image_paths:
                                if os.path.exists(img_path):
                                    img = Image(img_path, width=450, height=300)
                                    img.hAlign = 'CENTER'
                                    story.append(img)
                                    story.append(Spacer(1, 12))

                        # Generar PDF
                        doc.build(story)
                        buffer.seek(0)

                        # Descargar PDF directamente
                        st.download_button(
                            label="Descargar informe",
                            data=buffer,
                            file_name=f"Informe_predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="download_pdf"
                        )
                    else:
                        st.warning("No se encontraron datos válidos para las combinaciones seleccionadas.")