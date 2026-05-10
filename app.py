import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import json
import zipfile
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense



# =========================================================
# SIDEBAR MENU MODERN
# =========================================================

st.markdown("""
<style>

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Sidebar title */
.sidebar-title {
    font-size: 28px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

/* Menu button */
div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background-color: #1F2937;
    color: white;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 15px;
    transition: 0.3s;
}

/* Hover effect */
div.stButton > button:hover {
    background-color: #2563EB;
    color: white;
    transform: scale(1.03);
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #2563EB;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #9CA3AF;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR CONTENT
# =========================================================

st.sidebar.markdown(
    '<div class="sidebar-title">Menu</div>',
    unsafe_allow_html=True
)

# =========================================================
# SESSION STATE
# =========================================================

if "menu" not in st.session_state:
    st.session_state.menu = "Training Model"

# =========================================================
# CENTER BUTTONS
# =========================================================

with st.sidebar:

    col1, col2, col3 = st.columns([1,3,1])

    with col2:

        if st.button("Training Model"):
            st.session_state.menu = "Training Model"

        if st.button("Testing Model"):
            st.session_state.menu = "Testing Model"

        if st.button("How to Use"):
            st.session_state.menu = "How to Use"

menu = st.session_state.menu

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">How to Train Your Machine Learning?</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Build Your Neural Network-Based Machine Learning Model</div>',
    unsafe_allow_html=True
)

# =========================================================
# TRAINING MODEL
# =========================================================

if menu == "Training Model":

    st.header("Training Your Model")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        # =================================================
        # READ FILE
        # =================================================

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        columns = df.columns.tolist()

        # =================================================
        # INPUT OUTPUT SELECTION
        # =================================================

        input_cols = st.multiselect(
            "Select Input Variable",
            columns
        )

        output_cols = st.multiselect(
            "Select Output Variable",
            columns
        )

        # =================================================
        # HIDDEN LAYER CONFIGURATION
        # =================================================

        st.subheader("Configure Hidden Layer")

        num_hidden_layers = st.number_input(
            "Number of Hidden Layer",
            min_value=1,
            max_value=10,
            value=2
        )

        hidden_layers = []

        for i in range(num_hidden_layers):

            neurons = st.number_input(
                f"Node Hidden Layer {i+1}",
                min_value=1,
                value=16,
                key=f"hidden_{i}"
            )

            hidden_layers.append(neurons)
            
        # =================================================
        # ACTIVATION FUNCTION
        # =================================================

        st.subheader("Activation Function")

        activation_function = st.selectbox(
        "Pilih Activation Function",
               [
                 "relu",
                 "sigmoid",
                 "tanh",
                 "linear",
                 "softmax",
                 "elu",
                 "selu"
                ]
           )

        # =================================================
        # TRAINING PARAMETERS
        # =================================================

        st.subheader("Training Parameter")

        epochs = st.number_input(
            "Epoch",
            min_value=1,
            value=100
        )

        test_size = st.slider(
            "Test Size",
            0.1,
            0.5,
            0.2
        )
        
        # =========================================================
        # VISUALISASI ANN - PLOTLY SIMPLE
        # =========================================================

        import plotly.graph_objects as go

        st.subheader("Architecture Network")

        # Struktur network
        layers = [
            len(input_cols),
            *hidden_layers,
            len(output_cols)
        ]

        node_x = []
        node_y = []

        edge_x = []
        edge_y = []

        positions = []

        # =============================================
        # NODE POSITION
        # =============================================

        for i, neurons in enumerate(layers):

            x = [i] * neurons

            y = np.linspace(
                -neurons/2,
                neurons/2,
                neurons
            )

            positions.append(list(zip(x, y)))

            node_x.extend(x)
            node_y.extend(y)

        # =============================================
        # CONNECTIONS
        # =============================================

        for layer1, layer2 in zip(
            positions[:-1],
            positions[1:]
        ):

            for x1, y1 in layer1:

                for x2, y2 in layer2:

                    edge_x += [x1, x2, None]
                    edge_y += [y1, y2, None]

        # =============================================
        # EDGE TRACE
        # =============================================

        edge_trace = go.Scatter(

            x=edge_x,
            y=edge_y,

            mode='lines',

            line=dict(
                width=1,
                color='gray'
            ),

            hoverinfo='none'
        )

        # =============================================
        # NODE TRACE
        # =============================================

        node_trace = go.Scatter(

            x=node_x,
            y=node_y,

            mode='markers',

            marker=dict(

                size=24,

                color='royalblue',

                line=dict(
                    width=2,
                    color='white'
                )
            ),

            hoverinfo='none'
        )

        # =============================================
        # FIGURE
        # =============================================

        fig = go.Figure(
            data=[edge_trace, node_trace]
        )

        fig.update_layout(

            showlegend=False,

            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),

            xaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False
            ),

            yaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False
            ),

            height=500
        )

        # =============================================
        # SHOW
        # =============================================

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =================================================
        # TRAIN BUTTON
        # =================================================

        if st.button("Train Model"):

            if len(input_cols) == 0:

                st.error("Select at least 1 input")
                st.stop()

            if len(output_cols) == 0:

                st.error("Select at least 1 output")
                st.stop()

            # =============================================
            # DATASET
            # =============================================

            X = df[input_cols].values
            y = df[output_cols].values

            # =============================================
            # NORMALIZATION
            # =============================================

            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_x.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

            # =============================================
            # SPLIT DATA
            # =============================================

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y_scaled,
                test_size=test_size,
                random_state=42
            )

            
            # =============================================
            # BUILD MODEL
            # =============================================

            model = Sequential()

            # Hidden layer pertama
            model.add(
                Dense(
                    hidden_layers[0],
                    activation=activation_function,
                    input_shape=(X_train.shape[1],)
                )
            )

            # Hidden layer berikutnya
            for neurons in hidden_layers[1:]:

                model.add(
                    Dense(
                        neurons,
                        activation=activation_function
                    )
                )

            # Output layer
            model.add(
                Dense(
                    y_train.shape[1],
                    activation='linear'
                )
            )
            
            # =============================================
            # COMPILE MODEL
            # =============================================

            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )

            # =============================================
            # TRAIN MODEL
            # =============================================

            with st.spinner("Training model..."):

                history = model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(X_test, y_test)
                )

            st.success("Training Done")
            

            # =============================================
            # EVALUATION
            # =============================================

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Mean Absolute Percentage Error
            mape = np.mean(
                np.abs((y_test - y_pred) / (y_test + 1e-10))
            ) * 100

            # =============================================
            # SHOW RESULT
            # =============================================

            st.subheader("Model Evaluation")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Mean Squared Error (MSE)",
                    f"{mse:.6f}"
                )

            with col2:
                st.metric(
                    "Root Mean Squared Error (RMSE)",
                    f"{rmse:.6f}"
                )

            with col3:
                st.metric(
                    "Mean Absolute Error (MAE)",
                    f"{mae:.6f}"
                )

            col4, col5, col6 = st.columns(3)

            with col4:
                st.metric(
                    "R² Score",
                    f"{r2:.6f}"
                )

            with col5:
                st.metric(
                    "MAPE (%)",
                    f"{mape:.2f}%"
                )
            
            with col6:
                st.metric(
                    " ", " "
                )

            # =============================================
            # TRAINING GRAPH
            # =============================================

            st.subheader("Training Graph")

            history_df = pd.DataFrame(history.history)

            st.line_chart(
                history_df[['loss', 'val_loss']]
            )

            
            # =============================================
            # SAVE TEMP DIRECTORY
            # =============================================

            temp_dir = tempfile.mkdtemp()

            model_path = os.path.join(
                temp_dir,
                "model.h5"
            )

            metadata_path = os.path.join(
                temp_dir,
                "metadata.json"
            )

            scaler_x_path = os.path.join(
                temp_dir,
                "scaler_x.pkl"
            )

            scaler_y_path = os.path.join(
                temp_dir,
                "scaler_y.pkl"
            )

            zip_path = os.path.join(
                temp_dir,
                "model_package.zip"
            )

            # =============================================
            # SAVE MODEL
            # =============================================

            model.save(model_path)

            # =============================================
            # SAVE SCALERS
            # =============================================

            joblib.dump(
                scaler_x,
                scaler_x_path
            )

            joblib.dump(
                scaler_y,
                scaler_y_path
            )

            # =============================================
            # SAVE METADATA
            # =============================================

            metadata = {

                "input_variables": input_cols,
                "output_variables": output_cols,
                "input_nodes": len(input_cols),
                "output_nodes": len(output_cols),
                "hidden_layers": hidden_layers,
                "activation_function": activation_function

            }

            with open(metadata_path, "w") as f:

                json.dump(
                    metadata,
                    f,
                    indent=4
                )

            # =============================================
            # CREATE ZIP
            # =============================================

            with zipfile.ZipFile(
                zip_path,
                'w'
            ) as zipf:

                zipf.write(
                    model_path,
                    arcname="model.h5"
                )

                zipf.write(
                    metadata_path,
                    arcname="metadata.json"
                )

                zipf.write(
                    scaler_x_path,
                    arcname="scaler_x.pkl"
                )

                zipf.write(
                    scaler_y_path,
                    arcname="scaler_y.pkl"
                )

            # =============================================
            # DOWNLOAD BUTTON
            # =============================================

            st.subheader("Download Model")

            with open(zip_path, "rb") as f:

                st.download_button(
                    label="Download Model Package ZIP",
                    data=f,
                    file_name="model_package.zip",
                    mime="application/zip"
                )

# =========================================================
# TESTING MODEL
# =========================================================

if menu == "Testing Model":

    st.header("Testing Your Model")

    uploaded_zip = st.file_uploader(
        "Upload Model Package ZIP",
        type=["zip"]
    )

    if uploaded_zip is not None:

        # =============================================
        # TEMP DIRECTORY
        # =============================================

        temp_dir = tempfile.mkdtemp()

        zip_path = os.path.join(
            temp_dir,
            "model_package.zip"
        )

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        # =============================================
        # EXTRACT ZIP
        # =============================================

        with zipfile.ZipFile(
            zip_path,
            'r'
        ) as zip_ref:

            zip_ref.extractall(temp_dir)

        # =============================================
        # FILE PATHS
        # =============================================

        model_path = os.path.join(
            temp_dir,
            "model.h5"
        )

        metadata_path = os.path.join(
            temp_dir,
            "metadata.json"
        )

        scaler_x_path = os.path.join(
            temp_dir,
            "scaler_x.pkl"
        )

        scaler_y_path = os.path.join(
            temp_dir,
            "scaler_y.pkl"
        )

        # =============================================
        # LOAD FILES
        # =============================================

        model = load_model(model_path)

        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        with open(metadata_path, "r") as f:

            metadata = json.load(f)

        input_vars = metadata["input_variables"]
        output_vars = metadata["output_variables"]
        hidden_layers = metadata["hidden_layers"]

        # =========================================================
        # MODEL INFORMATION
        # =========================================================

        st.subheader("Model Information")

        # =============================================
        # CARD STYLE
        # =============================================

        st.markdown("""
        <style>

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
        }

        .metric-card {
            background-color: #1F2937;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid #374151;
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

        </style>
        """, unsafe_allow_html=True)

        # =============================================
        # METRICS
        # =============================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(input_vars)}</div>
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
                <div class="metric-value">{len(output_vars)}</div>
                <div class="metric-label">Output Nodes</div>
            </div>
            """, unsafe_allow_html=True)

        # =============================================
        # DETAIL CARDS
        # =============================================

        col4, col5 = st.columns(2)

        with col4:

            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Input Variables</div>
                <div class="info-content">
                    {", ".join(input_vars)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col5:

            st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Output Variables</div>
                <div class="info-content">
                    {", ".join(output_vars)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =============================================
        # HIDDEN LAYERS DETAIL
        # =============================================

        hidden_text = ""

        for i, neurons in enumerate(hidden_layers):

            hidden_text += f"""
            Layer {i+1} : {neurons} neurons
            """

        st.markdown(f"""
        <div class="info-card">
            <div class="info-title">Hidden Layer Architecture</div>
            <div class="info-content">
                {hidden_text}
            
        
        """, unsafe_allow_html=True)


        # =============================================
        # INPUT FORM
        # =============================================

        st.subheader("Input Data")

        user_inputs = []

        cols = st.columns(len(input_vars))

        for i, var in enumerate(input_vars):

            with cols[i]:

                value = st.number_input(
                    var,
                    value=0.0,
                    step=0.1
                )

                user_inputs.append(value)

        # =============================================
        # PREDICT BUTTON
        # =============================================

        if st.button("Predict"):

            X = np.array([user_inputs])

            X_scaled = scaler_x.transform(X)

            y_scaled = model.predict(X_scaled)

            y = scaler_y.inverse_transform(y_scaled)

            st.subheader("Prediction Result")

            for i, var in enumerate(output_vars):

                st.success(
                    f"{var} : {y[0][i]:.6f}"
                )
# =========================================================
# HOW TO USE
# =========================================================

if menu == "How to Use":

   # st.header("How to Use")

    # =====================================================
    # STYLE
    # =====================================================

    st.markdown("""
    <style>

    .guide-card {
        background-color: #111827;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid #374151;
    }

    .guide-title {
        font-size: 24px;
        font-weight: bold;
        color: #60A5FA;
        margin-bottom: 15px;
    }

    .guide-text {
        font-size: 16px;
        color: #E5E7EB;
        line-height: 1.8;
    }

    .step-box {
        background-color: #1F2937;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        border-left: 5px solid #2563EB;
    }

    .step-title {
        font-size: 18px;
        font-weight: bold;
        color: white;
    }

    .step-desc {
        color: #D1D5DB;
        margin-top: 5px;
        line-height: 1.6;
    }

    </style>
    """, unsafe_allow_html=True)


    # =====================================================
    # TRAINING GUIDE
    # =====================================================

    st.subheader("How to Train")

    training_steps = [

        (
            "1. Upload Dataset",
            "Upload the dataset file in CSV or XLSX format."
        ),

        (
            "2. Select Variables",
            "Select input and output variables from the dataset."
        ),

        (
            "3. Configure Hidden Layers",
            "Specify the number of hidden layers and the number of nodes in each layer."
        ),

        (
            "4. Training Model",
            "Click the Train Model button to start training."
        ),

        (
            "5. Download Model ZIP",
            "Download the ZIP file of the trained model."
        )

    ]

    for title, desc in training_steps:

        st.markdown(f"""
        <div class="step-box">
            
                {title}
                        
                {desc}
            
        
        """, unsafe_allow_html=True)

    # =====================================================
    # TESTING GUIDE
    # =====================================================

    st.subheader("How to Test")

    testing_steps = [

        (
            "1. Upload Model ZIP",
            "Upload the ZIP file of the model training results."
        ),

        (
            "2. Input Data",
            "Enter the input values ​​according to the model variables."
        ),

        (
            "3. Click Predict",
            "The system will generate the prediction output."
        )

    ]

    for title, desc in testing_steps:

        st.markdown(f"""
        <div class="step-box">

                {title}
  
                {desc}
           
        """, unsafe_allow_html=True)

