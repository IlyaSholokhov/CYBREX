import streamlit as st
from streamlit.components.v1 import html
import base64
import os

PC_IMAGE_BACKGROUND_PATH = 'ornament_1366_dark.png'
TABLET_IMAGE_BACKGROUND_PATH = 'ornament_768_dark.png'
MOBILE_IMAGE_BACKGROUND_PATH = 'ornament_360_dark.png'

# Функция для загрузки и кодирования изображения в base64
def load_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Функция для адаптивной установки фона
def set_background(images_path):
    # Кодируем три фона (для ПК, планшета и мобильного)
    bg_pc = load_image_as_base64(os.path.join(images_path, PC_IMAGE_BACKGROUND_PATH))
    bg_tablet = load_image_as_base64(os.path.join(images_path, TABLET_IMAGE_BACKGROUND_PATH))
    bg_mobile = load_image_as_base64(os.path.join(images_path, MOBILE_IMAGE_BACKGROUND_PATH))
    
    # Вставляем JS и CSS для установки фона
    st.markdown(f"""
    <script>
        const width = window.innerWidth;
        let device = "pc";
        if (width <= 768) {{
            device = "mobile";
        }} else if (width <= 1024) {{
            device = "tablet";
        }}
        window.parent.postMessage({{isStreamlitMessage: true, type: "deviceType", device}}, "*");
    </script>

    <style>
    [data-testid="stAppViewContainer"] {{
        transition: background 0.5s ease-in-out;
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True
    )
    
    # Получаем сообщение из JS через компоненты
    device_type = st.session_state.get("device_type", "pc")
    html(f"""
    <script>
    window.addEventListener("message", (event) => {{
        if (event.data.type === "deviceType") {{
            const device = event.data.device;
            window.parent.postMessage({{type: "streamlit:setComponentValue", value: device}}, "*");
        }}
    }});
    </script>
    """
    )

    # Подставляем фон в зависимости от устройства
    bg = {
    "pc": bg_pc,
    "tablet": bg_tablet,
    "mobile": bg_mobile
    }.get(device_type, bg_pc)
    
    st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bg}");
    }}
    </style>
    """,
    unsafe_allow_html=True
    )