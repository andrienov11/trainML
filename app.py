"""
app.py  (Frontend)
Berisi seluruh tampilan Streamlit.
Semua logika ML dan file I/O didelegasikan ke backend.py.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from PIL import Image
from backend import (
    load_dataset,
    build_model,
    prepare_data,
    train_one_epoch,
    evaluate_model,
    predict_single,
    predict_batch,
    build_model_zip,
    load_model_from_zip,
    dataframe_to_excel_bytes,
)


# =========================================================
# PAGE CONFIG
# =========================================================

logo = Image.open("logo.png")
st.set_page_config(
    page_title="Novandri - ML Development Tool",
    page_icon=logo,
    layout="wide"
)


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown("""
<style>

* {
    transition: all 0.3s ease !important;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #0f1942 50%, #0a0e27 100%) !important;
    color: #e0e0e0 !important;
}

.block-container {
    max-width: 1200px;
    padding-top: 3rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b3d 0%, #1a2952 50%, #0d1b3d 100%) !important;
    padding-top: 20px;
    border-right: 2px solid rgba(37, 99, 235, 0.2) !important;
}

.sidebar-title {
    font-size: 36px;
    font-weight: 900;
    background: linear-gradient(135deg, #60A5FA, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 45px;
    letter-spacing: 1px;
}

[data-testid="stSidebar"] div.stButton {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}

[data-testid="stSidebar"] div.stButton > button {
    width: 85% !important;
    min-width: 180px !important;
    height: 56px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-bottom: 12px !important;
    border-radius: 12px !important;
    border: 2px solid transparent !important;
    background: linear-gradient(135deg, #1e40af, #2563eb) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    white-space: nowrap !important;
    box-shadow: 0px 8px 20px rgba(37, 99, 235, 0.25) !important;
    position: relative !important;
    overflow: hidden !important;
}

[data-testid="stSidebar"] div.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
    transition: left 0.5s !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-4px) !important;
    color: white !important;
    box-shadow: 0px 12px 30px rgba(37, 99, 235, 0.4) !important;
    border: 2px solid rgba(96, 165, 250, 0.3) !important;
}

[data-testid="stSidebar"] div.stButton > button:hover::before {
    left: 100% !important;
}

.main-title {
    font-size: 60px !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #60A5FA, #3B82F6, #2563EB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 20px !important;
    margin-bottom: 12px !important;
    line-height: 1.2 !important;
    letter-spacing: -1px;
}

.subtitle {
    font-size: 40px !important;
    color: #bfdbfe !important;
    margin-bottom: 40px !important;
    line-height: 1.6 !important;
    font-weight: 500;
}

.info-card {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.3), rgba(37, 99, 235, 0.1));
    padding: 25px;
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
    border: 1.5px solid rgba(96, 165, 250, 0.3);
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 32px rgba(0, 0, 0, 0.2);
    position: relative;
    overflow: hidden;
}

.info-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(60, 130, 246, 0.1), transparent);
    border-radius: 50%;
}

.info-title {
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(135deg, #93c5fd, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.info-content {
    font-size: 15px;
    color: #e5e7eb;
    line-height: 1.8;
    position: relative;
    z-index: 1;
}

.metric-card {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(59, 130, 246, 0.05));
    padding: 28px 20px;
    border-radius: 16px;
    text-align: center;
    border: 1.5px solid rgba(96, 165, 250, 0.3);
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 32px rgba(0, 0, 0, 0.2);
    transform: scale(1);
    transition: all 0.3s ease;
    position: relative;
}

.metric-card:hover {
    transform: scale(1.05) !important;
    border-color: rgba(96, 165, 250, 0.6) !important;
    box-shadow: 0px 12px 40px rgba(37, 99, 235, 0.25) !important;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(59, 130, 246, 0.15)) !important;
}

.metric-value {
    font-size: 36px;
    font-weight: 900;
    background: linear-gradient(135deg, #93c5fd, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 14px;
    color: #bfdbfe;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.step-title {
    font-size: 24px;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 28px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
}

.step-desc {
    font-size: 16px;
    color: #e5e7eb;
    margin-bottom: 16px;
    line-height: 1.7;
    font-weight: 500;
}

li {
    background-color: rgba(17, 24, 39, 0.6) !important;
    color: #e5e7eb !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    border-left: 3px solid transparent !important;
    transition: all 0.3s ease !important;
    margin-bottom: 8px !important;
    border-radius: 8px !important;
}

li:hover {
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.2), rgba(59, 130, 246, 0.1)) !important;
    color: #bfdbfe !important;
    border-left-color: #60A5FA !important;
    transform: translateX(4px) !important;
}

li[aria-selected="true"] {
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.4), rgba(59, 130, 246, 0.2)) !important;
    color: #93c5fd !important;
    font-weight: 700 !important;
    border-left-color: #60A5FA !important;
    box-shadow: inset 0 0 15px rgba(37, 99, 235, 0.2) !important;
}

/* Streamlit Header Styling */
[data-testid="stHeader"] {
    background: transparent !important;
}

/* Streamlit Expander */
.streamlit-expanderHeader {
    background: rgba(30, 58, 138, 0.2) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(96, 165, 250, 0.2) !important;
}

/* Input Fields */
input, textarea, select {
    background-color: rgba(17, 24, 39, 0.8) !important;
    color: #e5e7eb !important;
    border: 1.5px solid rgba(96, 165, 250, 0.2) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-weight: 500 !important;
}

input:focus, textarea:focus, select:focus {
    background-color: rgba(37, 99, 235, 0.15) !important;
    border-color: rgba(96, 165, 250, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.1) !important;
}

/* Header with logo */
.header-container {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 2px;
    margin-bottom: 2px;
}

.logo-container {
    flex-shrink: 0;
}

.logo-container img {
    width: 30px;
    height: 20px;
    border-radius: 5px;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
    border: 2px solid rgba(96, 165, 250, 0.3);
    
}

.header-text {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.tagline {
    font-size: 20px;
    color: #cbd5e1;
    line-height: 1.65;
    margin-top: 8px;
}

/* How to Use Card Styles */
.how-to-container {
    width: 100%;
}

.section-title {
    font-size: 32px;
    font-weight: 900;
    background: linear-gradient(135deg, #60A5FA, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 30px;
    margin-top: 0px;
    display: flex;
    align-items: center;
    gap: 15px;
}

.step-card {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.32), rgba(37, 99, 235, 0.08));
    border: 1.5px solid rgba(96, 165, 250, 0.3);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18);
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}

.step-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(96, 165, 250, 0.1), transparent);
    border-radius: 50%;
}

.step-card:hover {
    transform: translateY(-8px) scale(1.01);
    border-color: rgba(96, 165, 250, 0.55);
    box-shadow: 0 16px 40px rgba(37, 99, 235, 0.28);
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.55), rgba(37, 99, 235, 0.18));
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #60A5FA, #3B82F6);
    border-radius: 14px;
    font-weight: 900;
    font-size: 24px;
    color: white;
    margin-bottom: 16px;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.28);
}

.step-card-title {
    font-size: 20px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 12px;
    position: relative;
    z-index: 1;
}

.step-card-desc {
    font-size: 15px;
    color: #cbd5e1;
    line-height: 1.8;
    position: relative;
    z-index: 1;
}

.steps-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 24px;
    margin-bottom: 50px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE INIT
# =========================================================

default_states = {
    "menu": "Train Model",
    "dataset_df": None,
    "dataset_name": None,
    "trained_model_ready": False,
    "prediction_result": None,
    "model": None,
    "scaler_x": None,
    "scaler_y": None,
    "input_cols": [],
    "output_cols": [],
    "hidden_layers": [16, 16],
    "activation_function": "relu",
    "epochs": 100,
    "test_size": 0.2,
    "history_df": None,
    "mse": None,
    "mae": None,
    "rmse": None,
    "r2": None,
    "mape": None,
    "y_test_actual": None,
    "y_test_pred": None,
    "zip_data": None,
    "deploy_model_uploaded": False,
    "deploy_prediction_df": None,
    "deploy_result_df": None,
    "deploy_metadata": None,
    "deploy_model": None,
    "deploy_scaler_x": None,
    "deploy_scaler_y": None,
    "deploy_selected_input_cols": []
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">Menu</div>', unsafe_allow_html=True)

    if st.button("Train Model", use_container_width=True):
        st.session_state.menu = "Train Model"
    if st.button("Deploy Model", use_container_width=True):
        st.session_state.menu = "Deploy Model"
    if st.button("How to Use", use_container_width=True):
        st.session_state.menu = "How to Use"

    st.markdown("---")

    if st.button("Reset Application", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

menu = st.session_state.menu


# =========================================================
# HEADER
# =========================================================

col1, col2 = st.columns([0.3, 5])
with col1:
    st.write("")
    st.write("")
    st.image(logo, width=50)
with col2:
    st.markdown('<div class="main-title" style="margin-top: 2px; margin-bottom: 2px;">Novandri</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="margin-bottom: 0px;">ML Development Tool</div>', unsafe_allow_html=True)

st.markdown('<div class="tagline">Build Your Machine Learning Model Based on Artificial Neural Networks</div>', unsafe_allow_html=True)


# =========================================================
# PLOTLY CONFIG
# =========================================================

PLOTLY_STATIC_CONFIG = {
    "scrollZoom": False,
    "doubleClick": False,
    "displayModeBar": False,
    "staticPlot": True,
}


# =========================================================
# UI HELPER FUNCTIONS (murni tampilan, tanpa logika bisnis)
# =========================================================

def show_static_line_chart(df, columns, title=""):
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title, dragmode=False,
        plot_bgcolor="rgba(255, 255, 255, 0.05)", paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="white"),
        margin=dict(l=1, r=1, t=100, b=100),
        legend=dict(orientation="h", yanchor="top", y=-0.25,
                    xanchor="center", x=0.5, font=dict(size=12)),
        xaxis=dict(fixedrange=True, showgrid=False, zeroline=False),
        yaxis=dict(fixedrange=True, showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_STATIC_CONFIG)


def show_static_scatter_chart(df, x_col, y_col, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="markers",
                             name=f"{x_col} vs {y_col}"))
    fig.update_layout(
        title=title, dragmode=False,
        plot_bgcolor="rgba(255, 255, 255, 0.05)", paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="white"),
        margin=dict(l=1, r=1, t=100, b=100),
        xaxis=dict(title=x_col, fixedrange=True, showgrid=False, zeroline=False),
        yaxis=dict(title=y_col, fixedrange=True, showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_STATIC_CONFIG)


def visualize_ann(input_nodes, hidden_layers, output_nodes):
    layers = [input_nodes, *hidden_layers, output_nodes]
    node_x, node_y, edge_x, edge_y = [], [], [], []
    positions = []
    for i, neurons in enumerate(layers):
        x = [i] * neurons
        y = np.linspace(-neurons / 2, neurons / 2, neurons)
        positions.append(list(zip(x, y)))
        node_x.extend(x)
        node_y.extend(y)
    for layer1, layer2 in zip(positions[:-1], positions[1:]):
        for x1, y1 in layer1:
            for x2, y2 in layer2:
                edge_x += [x1, x2, None]
                edge_y += [y1, y2, None]
    fig = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, mode="lines",
                   line=dict(width=1, color="rgba(255,255,255,0.45)"), hoverinfo="none"),
        go.Scatter(x=node_x, y=node_y, mode="markers",
                   marker=dict(size=24, color="rgba(96,165,250,0.9)",
                               line=dict(width=2, color="white")),
                   hoverinfo="none"),
    ])
    fig.update_layout(
        showlegend=False, plot_bgcolor="rgba(0, 0, 0, 0)", paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        dragmode=False, height=500,
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_STATIC_CONFIG)


def show_model_config():
    if not (st.session_state.input_cols and st.session_state.output_cols):
        return
    st.subheader("Model Configuration")
    c1, c2, c3 = st.columns(3)
    for col, val, lbl in [
        (c1, len(st.session_state.input_cols),   "Input Nodes"),
        (c2, len(st.session_state.hidden_layers), "Hidden Layers"),
        (c3, len(st.session_state.output_cols),   "Output Nodes"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Input Variables</div>
            <div class="info-content">{", ".join(st.session_state.input_cols)}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Output Variables</div>
            <div class="info-content">{", ".join(st.session_state.output_cols)}</div>
        </div>""", unsafe_allow_html=True)
    hidden_text = "".join(
        f"Layer {i+1} : {n} neurons<br>"
        for i, n in enumerate(st.session_state.hidden_layers)
    )
    st.markdown(f"""
    <div class="info-card">
        <div class="info-title">Hidden Layer Architecture</div>
        <div class="info-content">
            {hidden_text}
            Activation Function : {st.session_state.activation_function}<br>
            Epoch : {st.session_state.epochs}<br>
            Test Size : {st.session_state.test_size}
        </div>
    </div>""", unsafe_allow_html=True)


def show_evaluation():
    if st.session_state.mse is None:
        return
    st.subheader("Model Evaluation")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("MSE",  f"{st.session_state.mse:.6f}")
    with c2: st.metric("RMSE", f"{st.session_state.rmse:.6f}")
    with c3: st.metric("MAE",  f"{st.session_state.mae:.6f}")
    c4, c5, c6 = st.columns(3)
    with c4: st.metric("R² Score", f"{st.session_state.r2:.6f}")
    with c5: st.metric("MAPE (%)", f"{st.session_state.mape:.2f}%")
    with c6: st.metric("Status", "Trained")


def show_training_graph():
    if st.session_state.history_df is None:
        return
    st.subheader("Training Graph")
    show_static_line_chart(
        st.session_state.history_df,
        ["loss", "val_loss"],
        "Training Loss vs Validation Loss"
    )


def show_deep_analysis():
    if st.session_state.y_test_actual is None or st.session_state.y_test_pred is None:
        return
    st.subheader("Deep Model Performance Analysis")
    y_actual = st.session_state.y_test_actual
    y_pred   = st.session_state.y_test_pred
    analysis_df = pd.DataFrame({
        "Actual":         y_actual.flatten(),
        "Prediction":     y_pred.flatten(),
        "Residual":       (y_actual - y_pred).flatten(),
        "Absolute Error": np.abs(y_actual - y_pred).flatten(),
    })
    show_static_line_chart(analysis_df, ["Actual", "Prediction"], "Actual vs Prediction")
    show_static_scatter_chart(analysis_df, "Actual", "Prediction",
                              "Actual vs Prediction Scatter Plot")
    show_static_line_chart(analysis_df, ["Residual"], "Residual Error")
    show_static_line_chart(analysis_df, ["Absolute Error"], "Absolute Error")
    st.write("Statistical Summary")
    st.dataframe(analysis_df.describe())


def show_trained_model_test():
    if not st.session_state.trained_model_ready:
        return
    st.subheader("Test Trained Model")
    test_inputs = []
    cols = st.columns(len(st.session_state.input_cols))
    for i, var in enumerate(st.session_state.input_cols):
        with cols[i]:
            val = st.number_input(f"{var}", value=0.0, step=0.1, key=f"test_input_{i}")
            test_inputs.append(val)
    if st.button("Predict"):
        result = predict_single(                          # <- backend
            st.session_state.model,
            st.session_state.scaler_x,
            st.session_state.scaler_y,
            test_inputs
        )
        st.session_state.prediction_result = result
    if st.session_state.prediction_result is not None:
        st.subheader("Prediction Result")
        result_cols = st.columns(len(st.session_state.output_cols))
        for i, var in enumerate(st.session_state.output_cols):
            with result_cols[i]:
                st.markdown(f"""
                <div class="info-card" style="text-align:center;">
                    <div class="info-title">{var}</div>
                    <div style="font-size:36px; font-weight:bold; color:#60A5FA;">
                        {st.session_state.prediction_result[0][i]:.6f}
                    </div>
                </div>""", unsafe_allow_html=True)


# =========================================================
# PAGE: TRAIN MODEL
# =========================================================

if menu == "Train Model":
    st.header("Train Your Model")
    just_trained = False

    uploaded_file = st.file_uploader("Upload Dataset Here", type=["csv", "xlsx"])

    if uploaded_file is not None:
        with st.spinner("Loading dataset..."):
            df = load_dataset(uploaded_file, uploaded_file.name)   # <- backend
            st.session_state.dataset_df   = df
            st.session_state.dataset_name = uploaded_file.name

    if st.session_state.dataset_df is not None:
        df      = st.session_state.dataset_df
        columns = df.columns.tolist()

        st.subheader("Dataset Preview")
        if st.session_state.dataset_name:
            st.caption(f"Dataset: {st.session_state.dataset_name}")
        st.dataframe(df.head())

        st.subheader("Select Variables")
        valid_in  = [c for c in st.session_state.input_cols  if c in columns]
        valid_out = [c for c in st.session_state.output_cols if c in columns]

        with st.form("variable_form"):
            input_cols_temp  = st.multiselect("Select Input Variable",  columns, default=valid_in)
            output_cols_temp = st.multiselect("Select Output Variable", columns, default=valid_out)
            apply = st.form_submit_button("Apply Selection")

        if apply:
            st.session_state.input_cols        = input_cols_temp
            st.session_state.output_cols       = output_cols_temp
            st.session_state.prediction_result = None
            st.success("Variable selection applied")

        st.subheader("Configure Hidden Layer")
        num_hidden = st.number_input("Number of Hidden Layer", min_value=1, max_value=10,
                                     value=len(st.session_state.hidden_layers))
        hidden_layers = []
        for i in range(num_hidden):
            default_n = st.session_state.hidden_layers[i] if i < len(st.session_state.hidden_layers) else 16
            neurons   = st.number_input(f"Node Hidden Layer {i+1}", min_value=1,
                                        value=int(default_n), key=f"hidden_{i}")
            hidden_layers.append(neurons)
        st.session_state.hidden_layers = hidden_layers

        st.subheader("Activation Function")
        activation_options = ["relu", "sigmoid", "tanh", "linear", "softmax", "elu", "selu"]
        act_idx = activation_options.index(st.session_state.activation_function) \
                  if st.session_state.activation_function in activation_options else 0
        activation_function = st.selectbox("Select Activation Function", activation_options, index=act_idx)
        st.session_state.activation_function = activation_function

        st.subheader("Training Parameter")
        epochs    = st.number_input("Epoch", min_value=1, value=int(st.session_state.epochs))
        test_size = st.slider("Test Size", 0.1, 0.5, float(st.session_state.test_size))
        st.session_state.epochs    = epochs
        st.session_state.test_size = test_size

        if st.session_state.input_cols and st.session_state.output_cols:
            st.subheader("Architecture Network")
            visualize_ann(len(st.session_state.input_cols), hidden_layers,
                          len(st.session_state.output_cols))

        if st.button("Train Now", use_container_width=True):

            if not st.session_state.input_cols:
                st.error("Select at least 1 input"); st.stop()
            if not st.session_state.output_cols:
                st.error("Select at least 1 output"); st.stop()

            # --- backend calls ---
            X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare_data(
                df, st.session_state.input_cols, st.session_state.output_cols, test_size
            )
            model = build_model(
                input_dim=X_train.shape[1],
                output_dim=y_train.shape[1],
                hidden_layers=hidden_layers,
                activation_function=activation_function
            )

            # --- realtime UI loop ---
            st.subheader("Realtime Training Process")
            progress_bar   = st.progress(0)
            status_text    = st.empty()
            realtime_chart = st.empty()
            history_data   = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

            for epoch in range(int(epochs)):
                metrics = train_one_epoch(model, X_train, y_train, X_test, y_test)  # <- backend
                for k in history_data:
                    history_data[k].append(metrics[k])
                progress_bar.progress((epoch + 1) / int(epochs))
                status_text.write(
                    f"Epoch {epoch+1}/{int(epochs)} | "
                    f"Loss: {metrics['loss']:.6f} | Val Loss: {metrics['val_loss']:.6f} | "
                    f"MAE: {metrics['mae']:.6f} | Val MAE: {metrics['val_mae']:.6f}"
                )
                hdf = pd.DataFrame(history_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hdf["loss"],     mode="lines", name="loss"))
                fig.add_trace(go.Scatter(y=hdf["val_loss"], mode="lines", name="val_loss"))
                fig.update_layout(
                    title="Realtime Training Loss", dragmode=False,
                    plot_bgcolor="rgba(255, 255, 255, 0.05)", paper_bgcolor="rgba(0, 0, 0, 0)",
                    font=dict(color="white", size=12), height=500,
                    margin=dict(l=1, r=1, t=100, b=120),
                    legend=dict(orientation="h", yanchor="top", y=-0.25,
                                xanchor="center", x=0.5),
                    xaxis=dict(fixedrange=True, automargin=True, showgrid=False, zeroline=False),
                    yaxis=dict(fixedrange=True, automargin=True, showgrid=False, zeroline=False),
                )
                realtime_chart.plotly_chart(fig, use_container_width=True,
                                            config=PLOTLY_STATIC_CONFIG)

            history_df  = pd.DataFrame(history_data)
            eval_result = evaluate_model(model, X_test, y_test)          # <- backend
            zip_data    = build_model_zip(                                # <- backend
                model, scaler_x, scaler_y,
                st.session_state.input_cols, st.session_state.output_cols,
                hidden_layers, activation_function
            )

            st.session_state.model               = model
            st.session_state.scaler_x            = scaler_x
            st.session_state.scaler_y            = scaler_y
            st.session_state.hidden_layers       = hidden_layers
            st.session_state.activation_function = activation_function
            st.session_state.epochs              = epochs
            st.session_state.test_size           = test_size
            st.session_state.history_df          = history_df
            st.session_state.mse                 = eval_result["mse"]
            st.session_state.mae                 = eval_result["mae"]
            st.session_state.rmse                = eval_result["rmse"]
            st.session_state.r2                  = eval_result["r2"]
            st.session_state.mape                = eval_result["mape"]
            st.session_state.y_test_actual       = y_test
            st.session_state.y_test_pred         = eval_result["y_pred"]
            st.session_state.zip_data            = zip_data
            st.session_state.trained_model_ready = True
            st.session_state.prediction_result   = None
            just_trained = True

    if st.session_state.trained_model_ready:
        st.success("Model already trained")
        show_model_config()
        if not just_trained:
            show_training_graph()
        show_evaluation()
        show_deep_analysis()
        show_trained_model_test()
        st.subheader("Download Model")
        st.download_button(
            label="Download Model Package ZIP",
            data=st.session_state.zip_data,
            file_name="model_package.zip",
            mime="application/zip"
        )


# =========================================================
# PAGE: DEPLOY MODEL
# =========================================================

if menu == "Deploy Model":
    st.header("Deploy Your Model")

    uploaded_zip  = st.file_uploader("Upload Model Package ZIP",   type=["zip"],         key="deploy_zip_uploader")
    uploaded_data = st.file_uploader("Upload Data for Prediction", type=["csv", "xlsx"], key="deploy_data_uploader")

    if uploaded_zip is not None:
        bundle = load_model_from_zip(uploaded_zip)                   # <- backend
        st.session_state.deploy_model          = bundle["model"]
        st.session_state.deploy_scaler_x       = bundle["scaler_x"]
        st.session_state.deploy_scaler_y       = bundle["scaler_y"]
        st.session_state.deploy_metadata       = bundle["metadata"]
        st.session_state.deploy_model_uploaded = True

    if uploaded_data is not None:
        st.session_state.deploy_prediction_df = load_dataset(uploaded_data, uploaded_data.name)  # <- backend

    if st.session_state.deploy_model_uploaded and st.session_state.deploy_prediction_df is not None:

        model    = st.session_state.deploy_model
        scaler_x = st.session_state.deploy_scaler_x
        scaler_y = st.session_state.deploy_scaler_y
        metadata = st.session_state.deploy_metadata
        pred_df  = st.session_state.deploy_prediction_df

        input_vars_model  = metadata["input_variables"]
        output_vars_model = metadata["output_variables"]
        hidden_layers     = metadata["hidden_layers"]

        st.subheader("Prediction Data Preview")
        st.dataframe(pred_df.head())

        st.subheader("Model Information")
        c1, c2, c3 = st.columns(3)
        for col, val, lbl in [
            (c1, len(input_vars_model),  "Input Nodes"),
            (c2, len(hidden_layers),     "Hidden Layers"),
            (c3, len(output_vars_model), "Output Nodes"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        c4, c5 = st.columns(2)
        with c4:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Model Input Variables</div>
                <div class="info-content">{", ".join(input_vars_model)}</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Model Output Variables</div>
                <div class="info-content">{", ".join(output_vars_model)}</div>
            </div>""", unsafe_allow_html=True)

        st.subheader("Select Input Data Columns")
        data_columns = pred_df.columns.tolist()
        saved_cols   = st.session_state.deploy_selected_input_cols
        default_cols = [c for c in (saved_cols if saved_cols else input_vars_model)
                        if c in data_columns]

        with st.form("deploy_input_selection_form"):
            selected_temp = st.multiselect(
                "Select columns from uploaded data as model input",
                data_columns, default=default_cols
            )
            apply_deploy = st.form_submit_button("Apply Selection")

        if apply_deploy:
            st.session_state.deploy_selected_input_cols = selected_temp
            st.session_state.deploy_result_df = None
            st.success("Input columns applied")

        selected_input_cols = st.session_state.deploy_selected_input_cols

        if len(selected_input_cols) != len(input_vars_model):
            st.warning(
                f"Model has {len(input_vars_model)} inputs. "
                f"Currently you have selected {len(selected_input_cols)} columns."
            )

        if st.button("Run Prediction"):
            if len(selected_input_cols) != len(input_vars_model):
                st.error("The number of input columns selected must equal the number of model inputs.")
                st.stop()
            result_df = predict_batch(                                    # <- backend
                model, scaler_x, scaler_y,
                pred_df, selected_input_cols, output_vars_model
            )
            st.session_state.deploy_result_df = result_df

        if st.session_state.deploy_result_df is not None:
            st.subheader("Prediction Result")
            st.dataframe(st.session_state.deploy_result_df)
            excel_bytes = dataframe_to_excel_bytes(                       # <- backend
                st.session_state.deploy_result_df, sheet_name="Prediction Result"
            )
            st.download_button(
                label="Download Prediction Result",
                data=excel_bytes,
                file_name="prediction_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.info("Upload the Model Package ZIP and prediction data first.")


# =========================================================
# PAGE: HOW TO USE
# =========================================================

if menu == "How to Use":
    st.markdown('<div class="section-title">🎓 How to Train Your Model</div>', unsafe_allow_html=True)
    
    train_steps = [
        ("Upload Dataset",          "📤 Upload the dataset file in CSV or XLSX format."),
        ("Select Variables",        "✅ Select input and output variables from the dataset."),
        ("Configure Hidden Layers", "⚙️ Specify the number of hidden layers and nodes in each layer."),
        ("Train Model",             "🚀 Click the Train Now button to start ANN training."),
        ("Download Model ZIP",      "💾 Download the ZIP file of the trained model."),
    ]
    
    st.markdown('<div class="steps-grid">', unsafe_allow_html=True)
    for i, (title, desc) in enumerate(train_steps, start=1):
        st.markdown(f'''
        <div class="step-card">
            <div class="step-number">{i}</div>
            <div class="step-card-title">{title}</div>
            <div class="step-card-desc">{desc}</div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🔮 How to Deploy Your Model</div>', unsafe_allow_html=True)
    
    deploy_steps = [
        ("Upload Model ZIP",       "📦 Upload the trained model package ZIP."),
        ("Upload Prediction Data", "📊 Upload new data in CSV or XLSX format."),
        ("Select Input Columns",   "🎯 Select input columns that match the model structure."),
        ("Run Prediction",         "⚡ Click Run Prediction to generate predictions."),
        ("Download Result",        "💾 Download prediction results in Excel format."),
    ]
    
    st.markdown('<div class="steps-grid">', unsafe_allow_html=True)
    for i, (title, desc) in enumerate(deploy_steps, start=1):
        st.markdown(f'''
        <div class="step-card">
            <div class="step-number">{i}</div>
            <div class="step-card-title">{title}</div>
            <div class="step-card-desc">{desc}</div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)