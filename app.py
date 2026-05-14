import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import json
import zipfile
import joblib
import plotly.graph_objects as go

from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Novandri - ML Development Tool",
    layout="wide"
)


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #0e1117; !important;
    color: #e0e0e0 !important;
}

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071026, #0B1437);
    padding-top: 20px;
}

.sidebar-title {
    font-size: 34px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-bottom: 40px;
}

[data-testid="stSidebar"] div.stButton {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}

[data-testid="stSidebar"] div.stButton > button {
    width: 85% !important;
    min-width: 180px !important;
    height: 64px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-bottom: 18px !important;
    border-radius: 18px !important;
    border: none !important;
    background: linear-gradient(135deg, #16213E, #1E3A8A) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    white-space: nowrap !important;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3) !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    transform: translateY(-3px);
    color: white !important;
}

.main-title {
    font-size: 44px !important;
    font-weight: 800 !important;
    color: #2563EB !important;
    margin-top: 30px !important;
    margin-bottom: 8px !important;
    line-height: 1.2 !important;
}

.subtitle {
    font-size: 18px !important;
    color: #9CA3AF !important;
    margin-bottom: 35px !important;
    line-height: 1.5 !important;
}

.info-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin-bottom: 20px;
    border: 1px solid #374151;
}

.info-title {
    font-size: 18px;
    font-weight: bold;
    color: #60A5FA;
    margin-bottom: 10px;
}

.info-content {
    font-size: 16px;
    color: #E5E7EB;
    line-height: 1.8;
}

.metric-card {
    background-color: #1F2937;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #374151;
    margin-bottom: 20px;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #60A5FA;
}

.metric-label {
    font-size: 14px;
    color: #D1D5DB;
}

.step-title {
    font-size: 22px;
    font-weight: bold;
    color: #60A5FA;
    margin-top: 20px;
    margin-bottom: 3px;
}

.step-desc {
    font-size: 16px;
    color: #D1D5DB;
    margin-bottom: 3px;
    line-height: 1.7;
}

/* Dropdown popup */


li {
    background-color: #0e1117 !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

li:hover {
    background-color: #2563EB !important;
    color: white !important;
}

li[aria-selected="true"] {
    background-color: #DBEAFE !important;
    color: #1E3A8A !important;
    font-weight: bold !important;
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

    st.markdown(
        '<div class="sidebar-title">Menu</div>',
        unsafe_allow_html=True
    )

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

st.markdown(
    '<div class="main-title">Novandri - ML Development Tool</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Build Your Machine Learning Model Based on Artificial Neural Networks</div>',
    unsafe_allow_html=True
)


# =========================================================
# FUNCTIONS
# =========================================================

PLOTLY_STATIC_CONFIG = {
    "scrollZoom": False,
    "doubleClick": False,
    "displayModeBar": False,
    "staticPlot": True
}


def show_static_line_chart(df, columns, title=""):

    fig = go.Figure()

    for col in columns:

        fig.add_trace(
            go.Scatter(
                y=df[col],
                mode="lines",
                name=col
            )
        )

    fig.update_layout(

        title=title,

        dragmode=False,

        plot_bgcolor="#0E1117",

        paper_bgcolor="#0E1117",

        font=dict(color="white"),

        margin=dict(
            l=20,
            r=20,
            t=80,
            b=60
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),

        xaxis=dict(
            fixedrange=True
        ),

        yaxis=dict(
            fixedrange=True
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_STATIC_CONFIG
    )


def show_static_scatter_chart(df, x_col, y_col, title=""):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            name=f"{x_col} vs {y_col}"
        )
    )

    fig.update_layout(
        title=title,
        dragmode=False,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=x_col,
            fixedrange=True
        ),
        yaxis=dict(
            title=y_col,
            fixedrange=True
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_STATIC_CONFIG
    )

def visualize_ann(input_nodes, hidden_layers, output_nodes):

    layers = [input_nodes, *hidden_layers, output_nodes]

    node_x, node_y = [], []
    edge_x, edge_y = [], []
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

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="gray"),
        hoverinfo="none"
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=24,
            color="royalblue",
            line=dict(width=2, color="white")
        ),
        hoverinfo="none"
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        dragmode=False,
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": False,
            "doubleClick": False,
            "displayModeBar": False,
            "staticPlot": True
        }
    )


def show_model_config():

    if st.session_state.input_cols and st.session_state.output_cols:

        st.subheader("Model Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.input_cols)}</div>
                <div class="metric-label">Input Nodes</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.hidden_layers)}</div>
                <div class="metric-label">Hidden Layers</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.output_cols)}</div>
                <div class="metric-label">Output Nodes</div>
            </div>
            """, unsafe_allow_html=True)

        col4, col5 = st.columns(2)

        with col4:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Input Variables</div>
                <div class="info-content">
                    {", ".join(st.session_state.input_cols)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Output Variables</div>
                <div class="info-content">
                    {", ".join(st.session_state.output_cols)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        hidden_text = ""

        for i, neurons in enumerate(st.session_state.hidden_layers):
            hidden_text += f"Layer {i + 1} : {neurons} neurons<br>"

        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Hidden Layer Architecture</div>
            <div class="info-content">
                {hidden_text}
                Activation Function : {st.session_state.activation_function}<br>
                Epoch : {st.session_state.epochs}<br>
                Test Size : {st.session_state.test_size}
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_evaluation():

    if st.session_state.mse is not None:

        st.subheader("Model Evaluation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean Squared Error (MSE)", f"{st.session_state.mse:.6f}")

        with col2:
            st.metric("Root Mean Squared Error (RMSE)", f"{st.session_state.rmse:.6f}")

        with col3:
            st.metric("Mean Absolute Error (MAE)", f"{st.session_state.mae:.6f}")

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric("R² Score", f"{st.session_state.r2:.6f}")

        with col5:
            st.metric("MAPE (%)", f"{st.session_state.mape:.2f}%")

        with col6:
            st.metric("Status", "Trained")


def show_training_graph():

    if st.session_state.history_df is not None:

        st.subheader("Training Graph")

        show_static_line_chart(
            st.session_state.history_df,
            ["loss", "val_loss"],
            "Training Loss vs Validation Loss"
        )


def show_deep_analysis():

    if (
        st.session_state.y_test_actual is not None
        and st.session_state.y_test_pred is not None
    ):

        st.subheader("Deep Model Performance Analysis")

        y_actual = st.session_state.y_test_actual
        y_pred = st.session_state.y_test_pred

        analysis_df = pd.DataFrame({
            "Actual": y_actual.flatten(),
            "Prediction": y_pred.flatten(),
            "Residual": (y_actual - y_pred).flatten(),
            "Absolute Error": np.abs(y_actual - y_pred).flatten()
        })

        st.write("Actual vs Prediction")

        show_static_line_chart(
            analysis_df,
            ["Actual", "Prediction"],
            "Actual vs Prediction"
        )

        st.write("Scatter Plot: Actual vs Prediction")

        show_static_scatter_chart(
            analysis_df,
            "Actual",
            "Prediction",
            "Actual vs Prediction Scatter Plot"
        )

        st.write("Residual Error")

        show_static_line_chart(
            analysis_df,
            ["Residual"],
            "Residual Error"
        )

        st.write("Absolute Error")

        show_static_line_chart(
            analysis_df,
            ["Absolute Error"],
            "Absolute Error"
        )

        st.write("Statistical Summary")

        st.dataframe(
            analysis_df.describe()
        )


def show_trained_model_test():

    if st.session_state.trained_model_ready:

        st.subheader("Test Trained Model")

        test_inputs = []

        cols = st.columns(len(st.session_state.input_cols))

        for i, var in enumerate(st.session_state.input_cols):

            with cols[i]:

                value = st.number_input(
                    f"{var}",
                    value=0.0,
                    step=0.1,
                    key=f"test_input_{i}"
                )

                test_inputs.append(value)

        if st.button("Predict"):

            X_user = np.array([test_inputs])
            X_scaled_user = st.session_state.scaler_x.transform(X_user)
            y_scaled_user = st.session_state.model.predict(X_scaled_user)
            y_user = st.session_state.scaler_y.inverse_transform(y_scaled_user)

            st.session_state.prediction_result = y_user

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
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================
# TRAIN MODEL
# =========================================================

if menu == "Train Model":

    st.header("Train Your Model")

    just_trained = False

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        with st.spinner("Loading dataset..."):

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.dataset_df = df
            st.session_state.dataset_name = uploaded_file.name

    if st.session_state.dataset_df is not None:

        df = st.session_state.dataset_df

        st.subheader("Dataset Preview")

        if st.session_state.dataset_name is not None:
            st.caption(f"Dataset: {st.session_state.dataset_name}")

        st.dataframe(df.head())

        columns = df.columns.tolist()

        st.subheader("Select Variables")

        valid_input_defaults = [
            col for col in st.session_state.input_cols
            if col in columns
        ]

        valid_output_defaults = [
            col for col in st.session_state.output_cols
            if col in columns
        ]

        with st.form("variable_form"):

            input_cols_temp = st.multiselect(
                "Select Input Variable",
                columns,
                default=valid_input_defaults
            )

            output_cols_temp = st.multiselect(
                "Select Output Variable",
                columns,
                default=valid_output_defaults
            )

            apply_selection = st.form_submit_button("Apply Selection")

        if apply_selection:

            st.session_state.input_cols = input_cols_temp
            st.session_state.output_cols = output_cols_temp
            st.session_state.prediction_result = None

            st.success("Variable selection applied")

        st.subheader("Configure Hidden Layer")

        num_hidden_layers = st.number_input(
            "Number of Hidden Layer",
            min_value=1,
            max_value=10,
            value=len(st.session_state.hidden_layers)
        )

        hidden_layers = []

        for i in range(num_hidden_layers):

            default_neuron = (
                st.session_state.hidden_layers[i]
                if i < len(st.session_state.hidden_layers)
                else 16
            )

            neurons = st.number_input(
                f"Node Hidden Layer {i + 1}",
                min_value=1,
                value=int(default_neuron),
                key=f"hidden_{i}"
            )

            hidden_layers.append(neurons)

        st.session_state.hidden_layers = hidden_layers

        st.subheader("Activation Function")

        activation_options = [
            "relu",
            "sigmoid",
            "tanh",
            "linear",
            "softmax",
            "elu",
            "selu"
        ]

        activation_index = (
            activation_options.index(st.session_state.activation_function)
            if st.session_state.activation_function in activation_options
            else 0
        )

        activation_function = st.selectbox(
            "Select Activation Function",
            activation_options,
            index=activation_index
        )

        st.session_state.activation_function = activation_function

        st.subheader("Training Parameter")

        epochs = st.number_input(
            "Epoch",
            min_value=1,
            value=int(st.session_state.epochs)
        )

        test_size = st.slider(
            "Test Size",
            0.1,
            0.5,
            float(st.session_state.test_size)
        )

        st.session_state.epochs = epochs
        st.session_state.test_size = test_size

        if (
            len(st.session_state.input_cols) > 0
            and len(st.session_state.output_cols) > 0
        ):

            st.subheader("Architecture Network")

            visualize_ann(
                len(st.session_state.input_cols),
                st.session_state.hidden_layers,
                len(st.session_state.output_cols)
            )

        train_clicked = st.button(
            "Train Now",
            use_container_width=True
        )

        if train_clicked:

            if len(st.session_state.input_cols) == 0:
                st.error("Select at least 1 input")
                st.stop()

            if len(st.session_state.output_cols) == 0:
                st.error("Select at least 1 output")
                st.stop()

            X = df[st.session_state.input_cols].values
            y = df[st.session_state.output_cols].values

            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_x.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y_scaled,
                test_size=test_size,
                random_state=42
            )

            model = Sequential()

            model.add(
                Dense(
                    hidden_layers[0],
                    activation=activation_function,
                    input_shape=(X_train.shape[1],)
                )
            )

            for neurons in hidden_layers[1:]:

                model.add(
                    Dense(
                        neurons,
                        activation=activation_function
                    )
                )

            model.add(
                Dense(
                    y_train.shape[1],
                    activation="linear"
                )
            )

            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["mae"]
            )

            st.subheader("Realtime Training Process")

            progress_bar = st.progress(0)
            status_text = st.empty()
            realtime_chart = st.empty()

            history_data = {
                "loss": [],
                "val_loss": [],
                "mae": [],
                "val_mae": []
            }

            for epoch in range(int(epochs)):

                history_epoch = model.fit(
                    X_train,
                    y_train,
                    epochs=1,
                    verbose=0,
                    validation_data=(X_test, y_test)
                )

                loss = history_epoch.history["loss"][0]
                val_loss = history_epoch.history["val_loss"][0]
                mae_train = history_epoch.history["mae"][0]
                val_mae = history_epoch.history["val_mae"][0]

                history_data["loss"].append(loss)
                history_data["val_loss"].append(val_loss)
                history_data["mae"].append(mae_train)
                history_data["val_mae"].append(val_mae)

                progress = (epoch + 1) / int(epochs)

                progress_bar.progress(progress)

                status_text.write(
                    f"Epoch {epoch + 1}/{int(epochs)} | "
                    f"Loss: {loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"MAE: {mae_train:.6f} | "
                    f"Val MAE: {val_mae:.6f}"
                )

                history_df = pd.DataFrame(history_data)

                fig_realtime = go.Figure()

                fig_realtime.add_trace(
                    go.Scatter(
                        y=history_df["loss"],
                        mode="lines",
                        name="loss"
                    )
                )

                fig_realtime.add_trace(
                    go.Scatter(
                        y=history_df["val_loss"],
                        mode="lines",
                        name="val_loss"
                    )
                )

                fig_realtime.update_layout(
                    title="Realtime Training Loss",
                    dragmode=False,
                    plot_bgcolor="#0E1117",
                    paper_bgcolor="#0E1117",
                    font=dict(color="white"),
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True)
                )

                realtime_chart.plotly_chart(
                    fig_realtime,
                    use_container_width=True,
                    config=PLOTLY_STATIC_CONFIG
                )

            history_df = pd.DataFrame(history_data)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            mape = np.mean(
                np.abs((y_test - y_pred) / (y_test + 1e-10))
            ) * 100

            temp_dir = tempfile.mkdtemp()

            model_path = os.path.join(temp_dir, "model.h5")
            metadata_path = os.path.join(temp_dir, "metadata.json")
            scaler_x_path = os.path.join(temp_dir, "scaler_x.pkl")
            scaler_y_path = os.path.join(temp_dir, "scaler_y.pkl")
            zip_path = os.path.join(temp_dir, "model_package.zip")
            predict_path = os.path.join(temp_dir, "predict.py")
            requirements_path = os.path.join(temp_dir, "requirements.txt")
            readme_path = os.path.join(temp_dir, "README.txt")

            model.save(model_path)

            joblib.dump(scaler_x, scaler_x_path)
            joblib.dump(scaler_y, scaler_y_path)

            metadata = {
                "input_variables": st.session_state.input_cols,
                "output_variables": st.session_state.output_cols,
                "input_nodes": len(st.session_state.input_cols),
                "output_nodes": len(st.session_state.output_cols),
                "hidden_layers": hidden_layers,
                "activation_function": activation_function
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            predict_code = '''import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.h5")

scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Change input values below
'''

            for var in st.session_state.input_cols:
                predict_code += f'{var} = 0.0\n'

            predict_code += "\nX = np.array([["

            for i, var in enumerate(st.session_state.input_cols):
                predict_code += var

                if i < len(st.session_state.input_cols) - 1:
                    predict_code += ", "

            predict_code += """]])

X_scaled = scaler_x.transform(X)

y_scaled = model.predict(X_scaled)

y = scaler_y.inverse_transform(y_scaled)

print("\\nPrediction Result:")
"""

            for i, var in enumerate(st.session_state.output_cols):
                predict_code += f'print("{var} =", y[0][{i}])\n'

            with open(predict_path, "w") as f:
                f.write(predict_code)

            with open(requirements_path, "w") as f:
                f.write("""tensorflow
numpy
joblib
scikit-learn
""")

            with open(readme_path, "w") as f:
                f.write("""How to Run:

1. Install dependencies:

pip install -r requirements.txt

2. Run prediction:

python predict.py
""")

            with zipfile.ZipFile(zip_path, "w") as zipf:

                zipf.write(model_path, arcname="model.h5")
                zipf.write(metadata_path, arcname="metadata.json")
                zipf.write(scaler_x_path, arcname="scaler_x.pkl")
                zipf.write(scaler_y_path, arcname="scaler_y.pkl")
                zipf.write(predict_path, arcname="predict.py")
                zipf.write(requirements_path, arcname="requirements.txt")
                zipf.write(readme_path, arcname="README.txt")

            with open(zip_path, "rb") as f:
                zip_data = f.read()

            st.session_state.model = model
            st.session_state.scaler_x = scaler_x
            st.session_state.scaler_y = scaler_y
            st.session_state.hidden_layers = hidden_layers
            st.session_state.activation_function = activation_function
            st.session_state.epochs = epochs
            st.session_state.test_size = test_size
            st.session_state.history_df = history_df
            st.session_state.mse = mse
            st.session_state.mae = mae
            st.session_state.rmse = rmse
            st.session_state.r2 = r2
            st.session_state.mape = mape
            st.session_state.y_test_actual = y_test
            st.session_state.y_test_pred = y_pred
            st.session_state.zip_data = zip_data
            st.session_state.trained_model_ready = True
            st.session_state.prediction_result = None

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
# DEPLOY MODEL
# =========================================================

if menu == "Deploy Model":

    st.header("Deploy Your Model")

    uploaded_zip = st.file_uploader(
        "Upload Model Package ZIP",
        type=["zip"],
        key="deploy_zip_uploader"
    )

    uploaded_data = st.file_uploader(
        "Upload Data for Prediction",
        type=["csv", "xlsx"],
        key="deploy_data_uploader"
    )

    if uploaded_zip is not None:

        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "model_package.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        model_path = os.path.join(temp_dir, "model.h5")
        metadata_path = os.path.join(temp_dir, "metadata.json")
        scaler_x_path = os.path.join(temp_dir, "scaler_x.pkl")
        scaler_y_path = os.path.join(temp_dir, "scaler_y.pkl")

        st.session_state.deploy_model = load_model(model_path)
        st.session_state.deploy_scaler_x = joblib.load(scaler_x_path)
        st.session_state.deploy_scaler_y = joblib.load(scaler_y_path)

        with open(metadata_path, "r") as f:
            st.session_state.deploy_metadata = json.load(f)

        st.session_state.deploy_model_uploaded = True

    if uploaded_data is not None:

        if uploaded_data.name.endswith(".csv"):
            pred_df = pd.read_csv(uploaded_data)
        else:
            pred_df = pd.read_excel(uploaded_data)

        st.session_state.deploy_prediction_df = pred_df

    if (
        st.session_state.deploy_model_uploaded
        and st.session_state.deploy_prediction_df is not None
    ):

        model = st.session_state.deploy_model
        scaler_x = st.session_state.deploy_scaler_x
        scaler_y = st.session_state.deploy_scaler_y
        metadata = st.session_state.deploy_metadata
        pred_df = st.session_state.deploy_prediction_df

        input_vars_model = metadata["input_variables"]
        output_vars_model = metadata["output_variables"]
        hidden_layers = metadata["hidden_layers"]

        st.subheader("Prediction Data Preview")
        st.dataframe(pred_df.head())

        st.subheader("Model Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(input_vars_model)}</div>
                <div class="metric-label">Input Nodes</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(hidden_layers)}</div>
                <div class="metric-label">Hidden Layers</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(output_vars_model)}</div>
                <div class="metric-label">Output Nodes</div>
            </div>
            """, unsafe_allow_html=True)

        col4, col5 = st.columns(2)

        with col4:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Model Input Variables</div>
                <div class="info-content">
                    {", ".join(input_vars_model)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Model Output Variables</div>
                <div class="info-content">
                    {", ".join(output_vars_model)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Select Input Data Columns")

        data_columns = pred_df.columns.tolist()

        default_selected_cols = [
            col for col in input_vars_model
            if col in data_columns
        ]

        if len(st.session_state.deploy_selected_input_cols) > 0:

            default_selected_cols = [
                col for col in st.session_state.deploy_selected_input_cols
                if col in data_columns
            ]

        with st.form("deploy_input_selection_form"):

            selected_input_cols_temp = st.multiselect(
                "Select columns from uploaded data as model input",
                data_columns,
                default=default_selected_cols
            )

            apply_deploy_selection = st.form_submit_button("Apply Selection")

        if apply_deploy_selection:

            st.session_state.deploy_selected_input_cols = selected_input_cols_temp
            st.session_state.deploy_result_df = None

            st.success("Input columns applied")

        selected_input_cols = st.session_state.deploy_selected_input_cols

        if len(selected_input_cols) != len(input_vars_model):

            st.warning(
                f"Model has {len(input_vars_model)} inputs. "
                f"Currently you have selected {len(selected_input_cols)} columns."
            )

        if st.button("Run Prediction"):

            selected_input_cols = st.session_state.deploy_selected_input_cols

            if len(selected_input_cols) != len(input_vars_model):

                st.error(
                    "The number of input columns selected must be equal to the number of model inputs."
                )

                st.stop()

            X_new = pred_df[selected_input_cols].values
            X_new_scaled = scaler_x.transform(X_new)

            y_pred_scaled = model.predict(X_new_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            result_df = pred_df.copy()

            for i, output_name in enumerate(output_vars_model):
                result_df[f"Predicted_{output_name}"] = y_pred[:, i]

            st.session_state.deploy_result_df = result_df

        if st.session_state.deploy_result_df is not None:

            st.subheader("Prediction Result")

            st.dataframe(st.session_state.deploy_result_df)

            excel_buffer = BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                st.session_state.deploy_result_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Prediction Result"
                )

            excel_buffer.seek(0)

            st.download_button(
                label="Download Prediction Result",
                data=excel_buffer,
                file_name="prediction_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:

        st.info("Upload the Model Package ZIP and prediction data first.")


# =========================================================
# HOW TO USE
# =========================================================

if menu == "How to Use":

    st.subheader("How to Train")

    training_steps = [
        ("Upload Dataset", "Upload the dataset file in CSV or XLSX format."),
        ("Select Variables", "Select input and output variables from the dataset."),
        ("Configure Hidden Layers", "Specify the number of hidden layers and nodes in each layer."),
        ("Train Model", "Click the Train Now button to start ANN training."),
        ("Download Model ZIP", "Download the ZIP file of the trained model.")
    ]

    for i, (title, desc) in enumerate(training_steps, start=1):

        st.markdown(
            f'<div class="step-title">Step {i} - {title}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="step-desc">{desc}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("How to Deploy")

    deploy_steps = [
        ("Upload Model ZIP", "Upload the trained model package ZIP."),
        ("Upload Prediction Data", "Upload new data in CSV or XLSX format."),
        ("Select Input Columns", "Select input columns that match the model structure."),
        ("Run Prediction", "Click Run Prediction to generate predictions."),
        ("Download Result", "Download prediction results in Excel format.")
    ]

    for i, (title, desc) in enumerate(deploy_steps, start=1):

        st.markdown(
            f'<div class="step-title">Step {i} - {title}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="step-desc">{desc}</div>',
            unsafe_allow_html=True
        )