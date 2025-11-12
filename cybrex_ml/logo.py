import base64
import streamlit as st
import streamlit.components.v1 as components
import os

SVG_LOGO_PATH = "logo.svg"

def set_logo(svgs_path):

    if "show_main" not in st.session_state:
        st.session_state.show_main = True
    if "ml_sales_forecast_main" not in st.session_state:
        st.session_state.ml_sales_forecast_main = False
    if "ml_sales_forecast_data" not in st.session_state:
        st.session_state.ml_sales_forecast_data = False
    if "ml_sales_forecast_models" not in st.session_state:
        st.session_state.ml_sales_forecast_models = False
    if "running" not in st.session_state:
        st.session_state.running = False
    if "running_prod" not in st.session_state:
        st.session_state.running_prod = False


    with open(os.path.join(svgs_path, SVG_LOGO_PATH), "rb") as f:
        svg_b64 = base64.b64encode(f.read()).decode()
    # top: 15px for production
    # z-index: 999;
    st.markdown(f"""
    <style>
        .header {{
            position: fixed;
            top: 15px;
            left: 0;
            right: 0;
            height: 76px;
            border-bottom: 1px solid #000000;
            border-top: 1px solid #000000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 40px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            z-index: 999;
            backdrop-filter: blur(10px); /* эффект размытия */
            -webkit-backdrop-filter: blur(10px);
        }}
        .header-title {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 20px;
            font-weight: 600;
            color: #333;
            font-family: "Segoe UI", sans-serif;
            opacity: 0; /* изначально скрыта */
            transition: opacity 0.5s ease, transform 0.5s ease;
            pointer-events: none; /* чтобы не мешала кликам */
        }}
        .header-title.visible {{
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }}
        .header img {{
            height: 45px;
            width: auto;
            transition: height 0.3s ease;
        }}
        
        @media (max-width: 768px) {{
            .header {{   
                height: 55px;
                padding: 0 20px;
            }}
            .header img {{
                height: 35px;
            }}
            .header-title {{
                font-size: 16px;
            }}
        }}
    </style>

    <div class="header">
        <img src="data:image/svg+xml;base64,{svg_b64}" alt="Logo">
        <div class="header-title">CYBREX.ML — Аналитическая платформа</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Главная"):
        st.session_state.show_main = True
        st.session_state.ml_sales_forecast_main = False
        st.session_state.ml_sales_forecast_data = False
        st.session_state.ml_sales_forecast_models = False
        st.rerun()
    elif st.button('ML - прогноз продаж'):
        st.session_state.show_main = False
        st.session_state.ml_sales_forecast_main = True
        st.session_state.ml_sales_forecast_data = False
        st.session_state.ml_sales_forecast_models = False
        st.rerun()
    elif st.button("Данные"):
        st.session_state.show_main = False
        st.session_state.ml_sales_forecast_main = False
        st.session_state.ml_sales_forecast_data = True
        st.session_state.ml_sales_forecast_models = False
        st.rerun()
    elif st.button("Попробовать"):
        st.session_state.show_main = False
        st.session_state.ml_sales_forecast_main = False
        st.session_state.ml_sales_forecast_data = False
        st.session_state.ml_sales_forecast_models = True
        st.rerun()






















































        from jinja2 import Template
import base64
import os

def render_header(logo_path: str) -> str:
    """Рендерит HTML-хедер с логотипом и вкладками (без Streamlit)."""

    # загружаем логотип
    svg_b64 = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            svg_b64 = base64.b64encode(f.read()).decode()

    # шаблон через Jinja2 (работает независимо от Streamlit)
    html_template = Template("""
    <style>
        /* === ХЕДЕР === */
        .header {
            position: fixed;
            top: 60px;
            left: 0;
            right: 0;
            height: 76px;
            border-bottom: 1px solid #000000;
            border-top: 1px solid #000000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 40px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            z-index: 999;
            backdrop-filter: blur(10px); /* эффект размытия */
            -webkit-backdrop-filter: blur(10px);
        }

        /* === ЗАГОЛОВОК === */
        .header-title {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 20px;
            font-weight: 600;
            color: #333;
            font-family: "Segoe UI", sans-serif;
            opacity: 0; /* изначально скрыта */
            transition: opacity 0.5s ease, transform 0.5s ease;
            pointer-events: none; /* чтобы не мешала кликам */
        }

        /* === ЛОГОТИП === */
        .header img {
            height: 45px;
            width: auto;
            transition: height 0.3s ease;
        }

        /* === НАВИГАЦИЯ === */
        .nav {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 28px;
            flex: 1;
        }

        .nav button {
            border: none;
            background: transparent;
            font-size: 22px;
            font-weight: 600;
            color: #FFFFFF;
            cursor: pointer;
            padding: 4px 15px;
            position: relative;
            transition: all 0.25s ease;
            border-radius: 6px;
        }

        .nav button:hover {
            color: #FF0000;
            background: rgba(255,75,75,0.05);
            transform: scale(1.1);
        }

        .nav button.active {
            color: #FF0000;
        }

        /* Чтобы контент не залезал под хедер */
        body, .block-container {
            padding-top: 100px !important;
            margin: 0;
        }
    </style>

    <div class="header">
        <img src="data:image/svg+xml;base64,{{ svg_b64 }}" alt="Logo">
        <div class="nav">
            <button class="active" onclick="selectTab('main')">Главная</button>
            <button onclick="selectTab('ml')">ML-Прогноз</button>
            <button onclick="selectTab('data')">Данные</button>
            <button onclick="selectTab('models')">Модель</button>
        </div>

    </div>

    <script>
    function selectTab(tabName) {
        document.querySelectorAll('.nav button').forEach(btn => btn.classList.remove('active'));
        const btn = document.querySelector(`[onclick="selectTab('${tabName}')"]`);
        if (btn) btn.classList.add('active');
        window.parent.postMessage({ type: 'cybrex_set_page', page: tabName }, '*');
    }
    </script>
    """)

    return html_template.render(svg_b64=svg_b64)
