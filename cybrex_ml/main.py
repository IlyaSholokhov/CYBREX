import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
from set_backgrounds import set_background
import os
from recommendations import sales_recommendations
from sklearn.linear_model import Ridge
from feature_engineering import feature_engineering
from models import (load_data,
                    load_data_prod, 
                    CybrexXGBoost, 
                    CybrexProphet)
import streamlit as st
from set_logo import set_logo
import base64

SVG_LOGO_PATH = "logo/logo.svg"
SVG_LOGO_PATH_1 = "logo/logo_label.svg"


set_logo(SVG_LOGO_PATH)

# if "show_main" not in st.session_state:
#     st.session_state.show_main = True
# if "ml_sales_forecast_main" not in st.session_state:
#     st.session_state.ml_sales_forecast_main = False
# if "ml_sales_forecast_data" not in st.session_state:
#     st.session_state.ml_sales_forecast_data = False
# if "ml_sales_forecast_models" not in st.session_state:
#     st.session_state.ml_sales_forecast_models = False
if "running" not in st.session_state:
    st.session_state.running = False
if "running_prod" not in st.session_state:
    st.session_state.running_prod = False

st.set_page_config(page_title="CYBREX.ML",
                   layout="wide")

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

set_background('background')
st.markdown(
    """
    <style>
    /* Кнопка Streamlit */

    div.stButton > button:first-child {
        background-color: #FF0000;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1.4em;
        font-size: 1.5vw;
        font-weight: 600;
        cursor: pointer;
        transition: 0.3s;
        
    }
    div.stButton > button:hover {
        background-color: #FF0000;
        transform: scale(1.1);
    }
    html {
        scroll-behavior: smooth;
    }
    </style>
    """, unsafe_allow_html=True)

# # Приветственный экран для ML-платформы
# if st.session_state.show_main:
# st.image("logo/logo_label.svg", use_column_width=False, width=150)

with open("logo/logo_label_1.svg", "rb") as f:
        svg = base64.b64encode(f.read()).decode()
st.markdown(
    f"""
    <div id='responsive-div'>

    <style>
        /* Базовые стили */
        #responsive-div {{
            width: 100%;
            height: 100vh; /* по умолчанию на весь экран */
        }}

        /* 💻 Большие экраны */
        @media (min-width: 1200px) {{
            #responsive-div {{
                height: 80vh;
            }}
        }}

        /* 💻 Ноутбуки */
        @media (min-width: 992px) and (max-width: 1199px) {{
            #responsive-div {{
                height: 50vh;
            }}
        }}

        /* 📱 Планшеты */
        @media (min-width: 768px) and (max-width: 991px) {{
            #responsive-div {{
                height: 30vh;
            }}
        }}

        /* 📱 Телефоны */
        @media (max-width: 767px) {{
            #responsive-div {{
                height: 70vh;
            }}
        }}
    </style>
    
    <div style='text-align: center; padding: 15px; border-radius: 15px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1); animation: fadeIn 2s;'>
        <img src="data:image/svg+xml;base64,{svg}" style="display:block; margin:0 auto; animation: fadeIn 2s; transform: scale(3)">
        <h1 style='text-align: center; font-size: 80px; margin-top: 100px'>
            Вас приветствует CYBREX.AI
        </h1>
        <p style='text-align: center; font-size: 18px;'>
            Комплекс решений для анализа данных.
        </p>
    </div>

    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; padding: 15px; border-radius: 15px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1); animation: fadeIn 2s;'>
        <h1 style='text-align: center; font-size: 80px;'>
            ML — прогноз продаж
        </h1>
        <p style='text-align: center; font-size: 20px;'>
            Сервис, позволяющий оценить будущие продажи на основе исторических данных.<br>
        </p>
    </div>

    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True
)
df = pd.read_csv('demo_datasets/synthetic_sales.csv')
st.dataframe(df, height=210)
st.markdown(
    """
        <p style='text-align: left; font-size: 20px;'>
            Набор данных содержит исторические продажи за 2024 год.
            Мы использовали его для обучения модели машинного обучения и 
            получения прогноза продаж на следующий период.
            Так выглядят данные, которые вы можете загрузить для анализа.
            А наш алгоритм позаботится обо всем остальном!
            Он автоматически обработает данные, обучит модель и предоставит вам прогноз.
            Как же будут выглядеть данные после обработки? Смотрите ниже👇
        </p>
    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True)
fig = px.line(df, 
                x="ds", 
                y="sales", 
                color="sku",
                title="Временной ряд исторических продаж")
fig.update_layout(height=600, 
                    title_font_size=30,
                    plot_bgcolor="rgba(0,0,0,0.2)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    """
        <p style='text-align: left; font-size: 20px;'>
            После обработки данных нашим алгоритмом, они будут выглядеть следующим образом:
        </p>
    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
        }
    </style>
    """,
    unsafe_allow_html=True)
df_fe = feature_engineering(df)
st.dataframe(df_fe, height=250)
st.markdown(
    """
        <p style='text-align: left; font-size: 20px;'>
            Как видите, к исходным данным были добавлены новые признаки:
            день недели, месяц, лаги и скользящие средние.
            Эти дополнительные признаки помогают модели машинного обучения лучше понимать
            сезонные и временные зависимости в данных, что улучшает качество прогноза продаж.
        </p>
    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
        }
    </style>""",
    unsafe_allow_html=True)

grouped_df_by_day = df_fe.groupby('dayofweek').agg({'sales': 'mean'}).reset_index()
grouped_df_by_month = df_fe.groupby('month').agg({'sales': 'mean'}).reset_index()


fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type':'domain'}, {'type':'domain'}]],
)

# Диаграмма по дням недели
fig.add_trace(
    go.Pie(
        labels=["Понедельник","Вторник","Среда","Четверг","Пятница","Суббота","Воскресенье"],
        values=grouped_df_by_day['sales'],
        textinfo='percent+label',
        name='День недели',
        marker=dict(colors=px.colors.sequential.Teal),
        showlegend=False
    ),
    row=1, col=1
)

# Диаграмма по месяцам
fig.add_trace(
    go.Pie(
        labels=["Январь","Февраль","Март","Апрель","Май","Июнь",
                "Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"],
        values=grouped_df_by_month['sales'],
        textinfo='percent+label',
        name='Месяц',
        marker=dict(colors=px.colors.sequential.Teal),
        showlegend=False
    ),
    row=1, col=2
)

fig.update_layout(
    title_text="Распределение средних продаж по дням недели и месяцам",
    title_font_size=30,
    height=800,
    plot_bgcolor="rgba(0,0,0,0.2)",
    paper_bgcolor="rgba(0,0,0,0)", 
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <div style='text-align: center; padding: 15px; border-radius: 15px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1); animation: fadeIn 2s;'>
        <h1 style='text-align: center; font-size: 60px;'>
            Результаты моделирования
        </h1>
    </div>
    <p style='text-align: center; font-size: 20px;'>
        Выберите, какие признаки вы хотите использовать для обучения модели:
    </p>
    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
    /* Увеличиваем размер шрифта и расстояния между чекбоксами */
    div[data-testid="stCheckbox"] label p {
        font-size: 20px !important;      /* Размер шрифта */
        font-weight: 600 !important;     /* Полужирный текст */
        
    }

    /* Добавляем немного отступов и делаем аккуратный hover */
    div[data-testid="stCheckbox"] {
        margin-right: 25px !important;
        padding: 8px 14px !important;
        border-radius: 10px !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stCheckbox"]:hover {
        border: 2px solid #e8e8e8 !important;
    }
    </style>
    """, unsafe_allow_html=True
)

features = []

col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1.1, 1.1, 1])
with col1:
    dayofweek = st.checkbox("День недели", value=True)
    if dayofweek:
        features.append("dayofweek")
with col2:
    month = st.checkbox("Месяц", value=True)
    if month:
        features.append("month")
with col3:
    promo = st.checkbox("Скидки", value=True)
    if promo:
        features.append("promo")
with col4:
    lag_1 = st.checkbox("Цена 1\n\rдень назад", value=True)
    if lag_1:
        features.append("lag_1")
with col5:
    lag_7 = st.checkbox("Цена 7\n\rдней назад", value=True)
    if lag_7:
        features.append("lag_7")
with col6:
    ma7 = st.checkbox("Скользящее среднее \n\r 7 дней", value=True)
    if ma7:
        features.append("ma7")
with col7:
    ma30 = st.checkbox("Скользящее среднее \n\r 30 дней", value=True)
    if ma30:
        features.append("ma30")

st.markdown(
    """
    <p style='text-align: center; font-size: 20px;'>
        Выберите, на какой период вы хотите получить предсказание (количество дней):
    </p>
    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True
)
forecast_days = st.slider(
    label=" ",
    min_value=1,
    max_value=30,
    value=14,
    step=1
)

st.markdown("""
    <style>
    /* Центрируем радиокнопки */
    div[data-testid="stRadio"] > div {
        text-align: center;
    }
    /* Стили текста */
    div[data-testid="stRadio"] label p {
        font-size: 20px !important;      
        font-weight: 600 !important;     
    }

    /* Стили самих блоков */
    div[data-testid="stRadio"] {
        display: inline-block !important;   /* Горизонтально */
        margin-right: 25px !important;
        padding: 8px 14px !important;
        border-radius: 10px !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
# --- Радиокнопки (выбор только одного варианта) ---
sku = st.radio(
    "Выберите, для какого SKU вы хотите получить предсказание:",
    ("SKU-1", "SKU-2", "SKU-3"),
    horizontal=True  # важно: горизонтальное расположение
)

st.markdown(
    """
        <p style='text-align: left; font-size: 20px;'>
            Отлично! На этом этапе вы выбрали признаки, которые будут использоваться для обучения модели машинного обучения. Давайте обучим ее и посмотрим результат👇
        </p>
    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
        }
    </style>""",
    unsafe_allow_html=True)

# CSS-оформление
st.markdown("""
<style>
.progress-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    margin: 50px 0;
}
.progress-line {
    position: absolute;
    top: 50%;
    left: 0;
    width: 100%;
    height: 6px;
    background-color: #e0e0e0;
    z-index: 1;
    border-radius: 3px;
}
.progress-line-fill {
    position: absolute;
    top: 50%;
    left: 0;
    height: 6px;
    background-color: #4da6ff;
    z-index: 2;
    border-radius: 3px;
    transition: width 0.6s ease;
}
.step {
    position: relative;
    z-index: 3;
    text-align: center;
    width: 25%;
}
.step-circle {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background-color: #e0e0e0;
    margin: 0 auto 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    transition: all 0.3s ease;
}
.step.active .step-circle {
    background-color: #4da6ff;
}
.step.completed .step-circle {
    background-color: #2e8b57;
}
.step-label {
    font-size: 50px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# Кнопка запуска
if not st.session_state.running:
    col1, col2, col3 = st.columns([1.3, 1, 1])
    with col2:
        if st.button("Анализ и прогнозирование продаж"):
            st.session_state.running = True
            st.rerun()
else:        
    # Инициализация и настройка моделей
    с_xgb = CybrexXGBoost()
    с_prophet = CybrexProphet()

    with st.spinner(f"Загрузка данных..."):
        # time.sleep(2)
        df, train_df, test_df = load_data(forecast_days, sku=sku)
    
    with st.spinner(f"Обучение модели..."):
        # time.sleep(2)
        с_xgb.tune_hyperparameters_xgb(train_df, features, target="sales")
        с_xgb.train_xgboost(train_df, features)

        с_prophet.tune_hyperparameters_prophet(train_df)
        с_prophet.train_prophet(train_df)

    with st.spinner(f"Получение предсказаний..."):
        # time.sleep(2)
        c_xgb_preds = с_xgb.predict(test_df, features)
        c_prophet_preds = с_prophet.predict(df, forecast_days)
        
        meta_features = pd.DataFrame({
            "xgb": c_xgb_preds,
            "prophet": c_prophet_preds
        })
        meta_model = Ridge(alpha=0.1)
        meta_model.fit(meta_features, test_df['sales'])
        ensemble_pred = meta_model.predict(meta_features)

    with st.spinner(f"Подготовка результатов..."):
        # time.sleep(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["ds"],
            y=df["sales"],
            mode='lines+markers',
            line=dict(color='#87CEEB'),
            name='История продаж',
            hovertemplate='Дата: %{x}<br>Продажи: %{y}<extra></extra>'
        ))
        # Прогноз ансамбля
        fig.add_trace(go.Scatter(
            x=test_df["ds"].iloc[-forecast_days:],
            y=ensemble_pred,
            mode='lines+markers',
            name='Прогноз модели',
            hovertemplate='Дата: %{x}<br>Предсказание модели: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=[df["ds"].iloc[-forecast_days], df["ds"].iloc[-forecast_days]],
            y=[min(df["sales"].min(), ensemble_pred.min()), max(df["sales"].max(), ensemble_pred.max())],
            mode='lines',
            name='Начало прогноза',
            hoverinfo='skip'
        ))
        fig.update_layout(
            title=f"Прогноз продаж на следующие {forecast_days} дней",
            title_font_size=30,
            height=800,
            xaxis_title="Дата",
            yaxis_title="Продажи",
            hovermode='x unified',  # все подсказки при наведении по вертикали
        )
    st.success("✅ Все этапы успешно завершены!")
    st.session_state.running = False
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
    """
        <p style='text-align: center; font-size: 20px;'>
            А вот и результат! На графике вы видите исторические продажи (голубая линия) и прогноз модели (красная линия) на выбранный вами период.<br>
            Чтобы получить такой же прогноз с другими параметрами, просто начните выбирать их заново!
        </p>
        
    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
        }
    </style>""",
    unsafe_allow_html=True)

# [1.05, 9.5, 2.15]
# col, col1, col2 = st.columns([0.01, 13.95, 1.56])
# with col1:
#     if st.button("⬅ Назад"):
#         st.session_state.ml_sales_forecast_data = False
#         st.session_state.ml_sales_forecast_main = True
#         st.rerun()
# with col2:
#     if st.button("Далее ➡"):
#         st.session_state.ml_sales_forecast_data = False
#         st.session_state.ml_sales_forecast_models = True
#         st.rerun()

# Экран с интерактивом для пользователя
# elif st.session_state.ml_sales_forecast_models:
    # Добавляем плавную прокрутку вниз
st.markdown("""
    <script>
    window.scrollTo({top: 400, behavior: 'smooth'});
    </script>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; padding: 15px; border-radius: 15px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1); animation: fadeIn 2s;'>
        <h1 style='text-align: center; font-size: 80px;'>
            CYBREX.ML
        </h1>
        <p style='text-align: center; font-size: 18px;'>
            Сервис, позволяющий оценить будущие продажи на основе исторических данных.
            Попробуйте наш сервис со своими данными. <br><br>
            Просто загрузите их!  А наш алгоритм сам сформирует отчет и составит прогноз.👇
        </p>
    </div>
    <style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Контейнер загрузчика */
    [data-testid="stFileUploader"] {
        width: 100%;              /* ширина загрузчика */
        margin: 0 auto;          /* центрирование */
        background-color: rgba(255, 255, 255, 0.15); /* полупрозрачный фон */
        border-radius: 12px;     /* скруглённые углы */
        padding: 20px;
        transition: all 0.3s ease;
    }

    /* При наведении */
    [data-testid="stFileUploader"]:hover {
        background-color: rgba(255, 255, 255, 0.25);
        transform: scale(1.01);
    }

    /* Текст в загрузчике */
    [data-testid="stFileUploader"] label {
        font-size: 20px;
        font-weight: 600;
        color: #E3E3E3;
    }

    /* Кнопка выбора файла */
    [data-testid="stFileUploader"] section div div button {
        background-color: #212121 !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        padding: 8px 16px;
        cursor: pointer;
    }

    /* При наведении на кнопку */
    [data-testid="stFileUploader"] section div div button:hover {
        background-color: #212121 !important;
    }
    .custom-success {
        background-color: rgba(46, 204, 113, 0.15);
        border-left: 4px solid #2ecc71;
        color: #2ecc71;
        padding: 0.8em 1em;
        border-radius: 8px;
        margin-top: 12px;
        text-align: center;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Загрузите CSV-файл. \n\r Для демо-версии доступна загрузка истории продаж не более, чем для 5 продуктов.", type=["csv", "xlsx"])

if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file, parse_dates=["ds"])
    
    if uploaded_df['sku'].unique().__len__() <= 5:
        st.success("✅ Файл успешно загружен!")
        fig = px.line(uploaded_df,
                        x="ds",
                        y="sales",
                        color="sku",
                        title="Временной ряд исторических продаж")
        fig.update_layout(height=600,
                            title_font_size=30,
                            plot_bgcolor="rgba(0,0,0,0.2)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <p style='text-align: center; font-size: 20px;'>
                Выберите, какие признаки вы хотите использовать для обучения модели:
            </p>
            <style>
            @keyframes fadeIn {
                from {opacity: 0; transform: translateY(-20px);}
                to {opacity: 1; transform: translateY(0);}
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
            <style>
            /* Увеличиваем размер шрифта и расстояния между чекбоксами */
            div[data-testid="stCheckbox"] label p {
                font-size: 20px !important;      /* Размер шрифта */
                font-weight: 600 !important;     /* Полужирный текст */
                
            }

            /* Добавляем немного отступов и делаем аккуратный hover */
            div[data-testid="stCheckbox"] {
                margin-right: 25px !important;
                padding: 8px 14px !important;
                border-radius: 10px !important;
                transition: all 0.2s ease-in-out;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
            }
            div[data-testid="stCheckbox"]:hover {
                border: 2px solid #e8e8e8 !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        
        features = []

        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1.4, 1, 1, 1.2, 1])
        with col1:
            dayofweek = st.checkbox("День недели", value=True)
            if dayofweek:
                features.append("dayofweek")
        with col2:
            month = st.checkbox("Месяц", value=True)
            if month:
                features.append("month")
        with col3:
            promo = st.checkbox("Скидки", value=True)
            if promo:
                features.append("promo")
        with col4:
            lag_1 = st.checkbox("Цена 1 \n\rдень назад", value=True)
            if lag_1:
                features.append("lag_1")
        with col5:
            lag_7 = st.checkbox("Цена 7 \n\rдень назад", value=True)
            if lag_7:
                features.append("lag_7")
        with col6:
            ma7 = st.checkbox("Скользящее среднее \n\r 7 дней", value=True)
            if ma7:
                features.append("ma7")
        with col7:
            ma30 = st.checkbox("Скользящее среднее \n\r 30 дней", value=True)
            if ma30:
                features.append("ma30")

        st.markdown(
            """
            <p style='text-align: center; font-size: 20px;'>
                Выберите, на какой период вы хотите получить предсказание (количество дней):
            </p>
            <style>
            @keyframes fadeIn {
                from {opacity: 0; transform: translateY(-20px);}
                to {opacity: 1; transform: translateY(0);}
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        forecast_days = st.slider(
            label=" ",
            min_value=1,
            max_value=30,
            value=14,
            step=1
        )


        st.markdown("""
            <style>
            /* Центрируем радиокнопки */
            div[data-testid="stRadio"] > div {
                text-align: center;
            }
            /* Стили текста */
            div[data-testid="stRadio"] label p {
                font-size: 20px !important;      
                font-weight: 600 !important;     
            }

            /* Стили самих блоков */
            div[data-testid="stRadio"] {
                display: inline-block !important;   /* Горизонтально */
                margin-right: 25px !important;
                padding: 8px 14px !important;
                border-radius: 10px !important;
                transition: all 0.2s ease-in-out;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
            }
            </style>
            """, unsafe_allow_html=True)
        
        # --- Радиокнопки (выбор только одного варианта) ---
        sku = st.radio(
            "Выберите, для какого SKU вы хотите получить предсказание:",
            tuple(uploaded_df["sku"].unique()),
            horizontal=True  # важно: горизонтальное расположение
        )


        st.markdown("""
            <style>
            .progress-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                position: relative;
                margin: 50px 0;
            }
            .progress-line {
                position: absolute;
                top: 50%;
                left: 0;
                width: 100%;
                height: 6px;
                background-color: #e0e0e0;
                z-index: 1;
                border-radius: 3px;
            }
            .progress-line-fill {
                position: absolute;
                top: 50%;
                left: 0;
                height: 6px;
                background-color: #4da6ff;
                z-index: 2;
                border-radius: 3px;
                transition: width 0.6s ease;
            }
            .step {
                position: relative;
                z-index: 3;
                text-align: center;
                width: 25%;
            }
            .step-circle {
                width: 28px;
                height: 28px;
                border-radius: 50%;
                background-color: #e0e0e0;
                margin: 0 auto 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .step.active .step-circle {
                background-color: #4da6ff;
            }
            .step.completed .step-circle {
                background-color: #2e8b57;
            }
            .step-label {
                font-size: 50px;
                font-weight: 600;
            }
            </style>
            """, unsafe_allow_html=True)
        
        if not st.session_state.running_prod:
            col1, col2, col3 = st.columns([1.3, 1, 1])
            with col2:
                if st.button("Анализ и прогнозирование продаж"):
                    st.session_state.running_prod = True
                    st.rerun()
        else:        
            # Инициализация и настройка моделей
            с_xgb = CybrexXGBoost()
            с_prophet = CybrexProphet()
            with st.spinner(f"Загрузка данных..."):
                train_df = load_data_prod(uploaded_df, sku=sku)
                
            with st.spinner(f"Обучение модели..."):
                с_xgb.tune_hyperparameters_xgb(train_df, features, target="sales")
                с_xgb.train_xgboost(train_df, features)

                с_prophet.tune_hyperparameters_prophet(train_df)
                с_prophet.train_prophet(train_df)

                meta_features = pd.DataFrame({
                        "xgb": с_xgb.predict(train_df, features),
                        "prophet": с_prophet.predict_for_meta(train_df)
                    })
                meta_model = Ridge(alpha=0.3)
                meta_model.fit(meta_features, train_df['sales'])
                

            with st.spinner(f"Получение предсказаний..."):
                last_date = train_df["ds"].max()
                future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
                for future_date in future_dates:
                    new_row = pd.DataFrame([{"ds": future_date, "sku": train_df["sku"][0], "promo":0}])
                    train_df = pd.concat([train_df, new_row]).reset_index(drop=True)
                    train_df = feature_engineering(train_df)

                    test_df = train_df[train_df["ds"] == future_date]

                    c_xgb_preds_sales = с_xgb.predict(test_df, features)


                    c_prophet_preds = с_prophet.predict(test_df, 1)

                    
                    ensemble_pred = meta_model.coef_[0]*c_xgb_preds_sales + meta_model.coef_[1]*c_prophet_preds
                    train_df.loc[train_df["ds"] == future_date, "sales"] = ensemble_pred[0]

            with st.spinner(f"Подготовка результатов..."):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train_df[train_df["ds"] <= last_date]["ds"],
                    y=train_df[train_df["ds"] <= last_date]["sales"],
                    mode='lines+markers',
                    line=dict(color='#87CEEB'),
                    name='История продаж',
                    hovertemplate='Дата: %{x}<br>Продажи: %{y}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=train_df[train_df["ds"] >= last_date]["ds"],
                    y=train_df[train_df["ds"] >= last_date]["sales"],
                    mode='lines+markers',
                    name='Прогноз модели',
                    hovertemplate='Дата: %{x}<br>Предсказание модели: %{y}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=[train_df[train_df["ds"] == last_date]["ds"], train_df[train_df["ds"] == last_date]["ds"]],
                    y=[train_df["sales"].min(), train_df["sales"].max()],
                    mode='lines',
                    name='Начало прогноза',
                    hoverinfo='skip'
                ))
                fig.update_layout(
                    title=f"Прогноз продаж на следующие {forecast_days} дней",
                    title_font_size=30,
                    height=600,
                    xaxis_title="Дата",
                    yaxis_title="Продажи",
                    hovermode='x unified',  # все подсказки при наведении по вертикали
                )
            st.success("✅ Все этапы успешно завершены!")
            st.session_state.running_prod = False
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ В файле содержится больше 5 вариантов продукции.")


        


    # [1.35, 3.5, 2]
    # col, col1, col2 = st.columns([0.05, 13, 1.5])
    # with col1:
    #     if st.button("⬅ Назад"):
    #         st.session_state.ml_sales_forecast_models = False
    #         st.session_state.ml_sales_forecast_data = True
    #         st.rerun()
    # with col2:
    #     if st.button("Далее ➡"):
    #         st.session_state.ml_sales_forecast_data = False
    #         st.session_state.ml_sales_forecast_models = True