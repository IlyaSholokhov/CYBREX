import base64
import streamlit as st



def set_logo(svgs_path):
    with open(svgs_path, "rb") as f:
        svg_b64 = base64.b64encode(f.read()).decode()
    # top: 15px for production
    # z-index: 999;
    st.markdown(f"""
    <style>
        .header {{
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
