# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
from pathlib import Path
import sys
from datetime import datetime

# =====================================================================
# PAGE CONFIGURATION - MUST BE FIRST
# =====================================================================
# Load logo if available
logo_path = Path(__file__).parent / "assets" / "logo.png"
if logo_path.exists():
    page_icon = Image.open(logo_path)
else:
    page_icon = "🧠"

st.set_page_config(
    page_title="PANORAMA",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix path resolution
if str(Path(__file__).parent.resolve()) not in sys.path:
    sys.path.append(str(Path(__file__).parent.resolve()))

from src.config import CHECKPOINT_DIR, TRAIN_PATH

# =====================================================================
# ADVANCED THEME SYSTEM
# =====================================================================
def inject_theme():
    """Injects sophisticated dark theme with glass-morphism effects"""
    st.html("""
    <style>
    
    /* ===== VARIABLES ===== */
    :root {
        --primary: #6C63FF;
        --secondary: #FF6584;
        --accent: #00D4FF;
        --bg-primary: #0A0A0F;
        --bg-secondary: #12121A;
        --bg-card: rgba(255, 255, 255, 0.03);
        --border-glow: rgba(108, 99, 255, 0.15);
        --text-primary: #FFFFFF;
        --text-secondary: #A0A0B8;
        --shadow-glow: 0 8px 32px rgba(108, 99, 255, 0.12);
    }
    
    /* ===== GLOBAL OVERRIDES ===== */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Gill Sans', 'Trebuchet MS', sans-serif !important;
        color: var(--text-primary);
    }
    
    /* Override Streamlit's default background */
    .stApp > header {
        background: transparent !important;
    }
    
    .main > div {
        background: transparent !important;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6, 
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-family: 'Gill Sans', 'Trebuchet MS', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }
    
    h1 {
        font-size: 2.8rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    h2 {
        font-size: 1.8rem !important;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, var(--primary), var(--accent), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
    }
    
    .subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(18, 18, 26, 0.98), rgba(10, 10, 15, 0.98)) !important;
        border-right: 1px solid rgba(108, 99, 255, 0.1) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--text-primary) !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.05em;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-secondary) !important;
    }
    
    /* ===== CARDS / COLUMNS ===== */
    div[data-testid="column"] {
        background: var(--bg-card) !important;
        border: 1px solid rgba(108, 99, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 1.8rem 1.5rem !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: var(--shadow-glow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="column"]:hover {
        border-color: rgba(108, 99, 255, 0.2) !important;
        box-shadow: 0 12px 48px rgba(108, 99, 255, 0.15) !important;
        transform: translateY(-4px);
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(108, 99, 255, 0.03) !important;
        border: 2px dashed rgba(108, 99, 255, 0.2) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease !important;
        color: var(--text-secondary) !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary) !important;
        background: rgba(108, 99, 255, 0.05) !important;
        box-shadow: 0 0 40px rgba(108, 99, 255, 0.05) !important;
    }
    
    [data-testid="stFileUploaderDropzone"] span {
        color: var(--text-secondary) !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.2) !important;
        font-family: 'Gill Sans', 'Trebuchet MS', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.05), rgba(0, 212, 255, 0.05)) !important;
        border: 1px solid rgba(108, 99, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        font-family: 'Gill Sans', 'Trebuchet MS', sans-serif !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 400 !important;
        font-family: 'Gill Sans', 'Trebuchet MS', sans-serif !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: var(--accent) !important;
    }
    
    /* ===== ALERTS ===== */
    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        border-left: 4px solid var(--primary) !important;
        background: rgba(108, 99, 255, 0.05) !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    
    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(108, 99, 255, 0.1) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary) !important;
    }
    
    .stSelectbox select {
        background: transparent !important;
        color: var(--text-primary) !important;
        font-family: 'Gill Sans', 'Trebuchet MS', sans-serif !important;
    }
    
    /* ===== IMAGES ===== */
    [data-testid="stImage"] img {
        border-radius: 16px !important;
        border: 1px solid rgba(108, 99, 255, 0.1) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--primary), transparent) !important;
        margin: 2rem 0 !important;
        opacity: 0.3 !important;
    }
    
    /* ===== PROGRESS / SPINNER ===== */
    .stSpinner > div {
        border-color: var(--primary) !important;
        border-top-color: var(--accent) !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        font-family: 'Gill Sans', 'Trebuchet MS', sans-serif !important;
        background: rgba(108, 99, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(108, 99, 255, 0.1) !important;
    }
    
    /* ===== DATA FRAME ===== */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        border: 1px solid rgba(108, 99, 255, 0.08) !important;
        overflow: hidden !important;
    }
    
    [data-testid="stDataFrame"] table {
        font-family: 'Gill Sans', 'Trebuchet MS', sans-serif !important;
    }
    
    /* ===== RESPONSIVE TWEAKS ===== */
    @media (max-width: 768px) {
        h1 { 
            font-size: 2rem !important; 
        }
        h2 { 
            font-size: 1.4rem !important; 
        }
        [data-testid="stMetricValue"] { 
            font-size: 1.6rem !important; 
        }
        div[data-testid="column"] {
            padding: 1rem !important;
        }
    }
    
    /* ===== SCROLLBAR STYLING ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent);
    }
    
    /* ===== SELECTION COLOR ===== */
    ::selection {
        background: rgba(108, 99, 255, 0.3);
        color: var(--text-primary);
    }
    </style>
    """)
    
    # Optional background image
    bg_img_path = Path(__file__).parent / "assets" / "bg.png"
    if bg_img_path.exists():
        import base64
        with open(bg_img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.html(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(10, 10, 15, 0.82), rgba(10, 10, 15, 0.96)), url("data:image/png;base64,{encoded}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
        }}
        </style>
        """)

inject_theme()

# =====================================================================
# CACHED RESOURCES
# =====================================================================
@st.cache_resource(ttl=3600)
def load_models():
    """Loads and caches trained models with error handling"""
    baseline_path = CHECKPOINT_DIR / "best_baseline_model.keras"
    transfer_path = CHECKPOINT_DIR / "best_transfer_model.keras"
    
    models = {}
    
    if baseline_path.exists():
        try:
            models["Baseline CNN"] = tf.keras.models.load_model(str(baseline_path))
        except Exception as e:
            st.warning(f"Could not load baseline model: {e}")
    
    if transfer_path.exists():
        try:
            models["MobileNetV2 Transfer"] = tf.keras.models.load_model(str(transfer_path))
        except Exception as e:
            st.warning(f"Could not load transfer model: {e}")
    
    # Get class names
    try:
        classes = sorted([d for d in os.listdir(TRAIN_PATH) 
                         if os.path.isdir(os.path.join(TRAIN_PATH, d)) and not d.startswith('.')])
    except:
        classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    
    return models, classes

models_dict, class_names = load_models()

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("## 🎯 Control Panel")
    
    if not models_dict:
        st.error("🚫 No trained models found!")
        st.stop()
    
    if len(models_dict) > 1:
        selected_model = st.selectbox(
            "Select Model",
            options=list(models_dict.keys()),
            help="Choose which trained model to use for inference"
        )
    else:
        selected_model = list(models_dict.keys())[0]
        
    active_model = models_dict[selected_model]
    
    st.markdown("---")
    
    # Model info
    st.markdown("### 📊 Model Info")
    st.caption(f"**Active:** {selected_model}")
    st.caption(f"**Classes:** {len(class_names)} categories")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    
    # Placeholder for additional controls
    if st.button("🔄 Refresh Models", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color: #A0A0B8; padding: 0.5rem;'>
    <strong>⚡ PANORAMA v2.0</strong><br>
    Production-grade vision inference
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# MAIN INTERFACE
# =====================================================================
logo_b64 = ""
if logo_path.exists():
    import base64
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    header_html = f"<h1><img src='data:image/png;base64,{logo_b64}' style='height: 1.5em; vertical-align: text-bottom; margin-right: 0.5rem; border-radius: 8px;'> <span class='gradient-text'>PANORAMA</span></h1>"
else:
    header_html = "<h1>🧠 <span class='gradient-text'>PANORAMA</span></h1>"

st.markdown(header_html, unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 1.5rem;">
        Advanced image classification with real-time confidence analysis
    </p>
    <div style="background: rgba(108, 99, 255, 0.05); border: 1px solid rgba(108, 99, 255, 0.1); border-radius: 12px; padding: 1.2rem; display: inline-block; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);">
        <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">Trained to recognize</p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem 2.5rem; font-weight: 600; font-size: 1.05rem;">
            <span>🏙️ Buildings</span>
            <span>🌲 Forest</span>
            <span>❄️ Glacier</span>
            <span>⛰️ Mountain</span>
            <span>🌊 Sea</span>
            <span>🏙️ Street</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats row
col_stats1, col_stats2, col_stats3 = st.columns(3)
with col_stats1:
    st.metric("Available Models", len(models_dict), delta=None)
with col_stats2:
    st.metric("Class Categories", len(class_names), delta=None)
with col_stats3:
    st.metric("Status", "🟢 Online", delta=None)

st.markdown("---")

# Main layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload Image")
    

    
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", use_container_width=True)
        
        # Image info
        st.caption(f"**Dimensions:** {image.size[0]} × {image.size[1]} px")
        st.caption(f"**Format:** {image.format}")

with col_right:
    st.markdown("### 🎯 Analysis Results")
    
    if not uploaded_file:
        st.info("📸 Upload an image to begin classification")
    else:
        with st.spinner("🧠 Analyzing image..."):
            # Preprocess
            processed = image.resize((150, 150))
            img_array = tf.keras.utils.img_to_array(processed)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = active_model.predict(img_array, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_idx]
            confidence = predictions[predicted_idx] * 100
            
            # Display results
            st.success(f"### 🏷️ {predicted_class.title()}")
            st.metric("Confidence", f"{confidence:.1f}%", delta=None)
            
            # Confidence indicator
            if confidence > 85:
                st.balloons()
                st.caption("✅ High confidence prediction")
            elif confidence > 65:
                st.caption("🔶 Moderate confidence prediction")
            else:
                st.caption("🔴 Low confidence - consider reviewing")
            
            # Probability distribution
            st.markdown("#### 📊 Confidence Distribution")
            chart_data = pd.DataFrame({
                "Category": [c.title() for c in class_names],
                "Confidence": predictions
            }).set_index("Category")
            
            st.bar_chart(chart_data, height=300, use_container_width=True)
            
            # Top 3 predictions
            st.markdown("#### 🏆 Top Predictions")
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_data = pd.DataFrame({
                "Category": [class_names[i].title() for i in top_indices],
                "Score": [f"{predictions[i]*100:.1f}%" for i in top_indices]
            })
            st.dataframe(top_data, hide_index=True, use_container_width=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption(f"🔬 PANORAMA • {datetime.now().strftime('%Y')} • Built with Streamlit & TensorFlow")