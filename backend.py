"""
backend.py
Berisi semua logika bisnis: training model ANN, prediksi,
pembuatan file ZIP, dan evaluasi model.
Tidak ada dependensi Streamlit di sini.
"""

import os
import json
import zipfile
import tempfile

import numpy as np
import pandas as pd
import joblib

from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(file_obj, filename: str) -> pd.DataFrame:
    """
    Membaca file CSV atau XLSX dan mengembalikan DataFrame.

    Parameters
    ----------
    file_obj : file-like object
        Objek file yang diunggah.
    filename : str
        Nama file untuk menentukan format (csv/xlsx).

    Returns
    -------
    pd.DataFrame
    """
    if filename.endswith(".csv"):
        return pd.read_csv(file_obj)
    else:
        return pd.read_excel(file_obj)


# =========================================================
# MODEL BUILDING
# =========================================================

def build_model(
    input_dim: int,
    output_dim: int,
    hidden_layers: list[int],
    activation_function: str
) -> Sequential:
    """
    Membangun arsitektur ANN Sequential.

    Parameters
    ----------
    input_dim : int
        Jumlah fitur input.
    output_dim : int
        Jumlah node output.
    hidden_layers : list[int]
        Daftar jumlah neuron per hidden layer.
    activation_function : str
        Fungsi aktivasi hidden layer (relu, sigmoid, tanh, dll).

    Returns
    -------
    model : Sequential (belum di-compile)
    """
    model = Sequential()

    model.add(
        Dense(
            hidden_layers[0],
            activation=activation_function,
            input_shape=(input_dim,)
        )
    )

    for neurons in hidden_layers[1:]:
        model.add(Dense(neurons, activation=activation_function))

    model.add(Dense(output_dim, activation="linear"))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


# =========================================================
# TRAINING
# =========================================================

def train_one_epoch(model, X_train, y_train, X_test, y_test):
    """
    Melatih model selama 1 epoch dan mengembalikan metrik epoch tersebut.

    Parameters
    ----------
    model : Sequential
    X_train, y_train : np.ndarray  — data training (sudah di-scale)
    X_test,  y_test  : np.ndarray  — data validasi (sudah di-scale)

    Returns
    -------
    dict dengan key: loss, val_loss, mae, val_mae
    """
    history = model.fit(
        X_train, y_train,
        epochs=1,
        verbose=0,
        validation_data=(X_test, y_test)
    )

    return {
        "loss":     history.history["loss"][0],
        "val_loss": history.history["val_loss"][0],
        "mae":      history.history["mae"][0],
        "val_mae":  history.history["val_mae"][0],
    }


def prepare_data(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    test_size: float
):
    """
    Melakukan scaling dan split data.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler_x, scaler_y
    """
    X = df[input_cols].values
    y = df[output_cols].values

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=test_size,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y


# =========================================================
# EVALUASI
# =========================================================

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Mengevaluasi model pada data uji dan mengembalikan metrik evaluasi.

    Returns
    -------
    dict dengan key: mse, mae, rmse, r2, mape, y_pred
    """
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)

    return {
        "mse":    float(mse),
        "mae":    float(mae),
        "rmse":   rmse,
        "r2":     float(r2),
        "mape":   mape,
        "y_pred": y_pred,
    }


# =========================================================
# PREDIKSI SINGLE INPUT
# =========================================================

def predict_single(
    model,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    input_values: list[float]
) -> np.ndarray:
    """
    Melakukan prediksi untuk satu baris input (manual input).

    Parameters
    ----------
    input_values : list[float]
        Nilai input sesuai urutan input_cols.

    Returns
    -------
    np.ndarray shape (1, n_output) — nilai asli (inverse-transform)
    """
    X = np.array([input_values])
    X_scaled = scaler_x.transform(X)
    y_scaled = model.predict(X_scaled)
    return scaler_y.inverse_transform(y_scaled)


# =========================================================
# PREDIKSI BATCH (DEPLOY)
# =========================================================

def predict_batch(
    model,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    df: pd.DataFrame,
    selected_input_cols: list[str],
    output_vars: list[str]
) -> pd.DataFrame:
    """
    Melakukan prediksi untuk seluruh baris DataFrame.

    Returns
    -------
    DataFrame asli + kolom Predicted_<output_var>
    """
    X_new = df[selected_input_cols].values
    X_scaled = scaler_x.transform(X_new)
    y_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_scaled)

    result_df = df.copy()
    for i, col in enumerate(output_vars):
        result_df[f"Predicted_{col}"] = y_pred[:, i]

    return result_df


# =========================================================
# PACKAGING MODEL (ZIP)
# =========================================================

def _write_predict_script(input_cols: list[str], output_cols: list[str]) -> str:
    """Membuat isi script predict.py."""
    code = '''import numpy as np
import joblib
from tensorflow.keras.models import load_model

model    = load_model("model.h5")
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Ganti nilai input di bawah ini
'''
    for var in input_cols:
        code += f"{var} = 0.0\n"

    code += "\nX = np.array([[" + ", ".join(input_cols) + "]])\n"
    code += """
X_scaled = scaler_x.transform(X)
y_scaled = model.predict(X_scaled)
y        = scaler_y.inverse_transform(y_scaled)

print("\\nHasil Prediksi:")
"""
    for i, var in enumerate(output_cols):
        code += f'print("{var} =", y[0][{i}])\n'

    return code


def build_model_zip(
    model,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    input_cols: list[str],
    output_cols: list[str],
    hidden_layers: list[int],
    activation_function: str
) -> bytes:
    """
    Menyimpan model, scaler, metadata, dan script ke dalam ZIP.

    Returns
    -------
    bytes — isi file ZIP siap di-download
    """
    temp_dir = tempfile.mkdtemp()

    paths = {
        "model":        os.path.join(temp_dir, "model.h5"),
        "metadata":     os.path.join(temp_dir, "metadata.json"),
        "scaler_x":     os.path.join(temp_dir, "scaler_x.pkl"),
        "scaler_y":     os.path.join(temp_dir, "scaler_y.pkl"),
        "predict":      os.path.join(temp_dir, "predict.py"),
        "requirements": os.path.join(temp_dir, "requirements.txt"),
        "readme":       os.path.join(temp_dir, "README.txt"),
        "zip":          os.path.join(temp_dir, "model_package.zip"),
    }

    # Simpan artefak
    model.save(paths["model"])
    joblib.dump(scaler_x, paths["scaler_x"])
    joblib.dump(scaler_y, paths["scaler_y"])

    metadata = {
        "input_variables":  input_cols,
        "output_variables": output_cols,
        "input_nodes":      len(input_cols),
        "output_nodes":     len(output_cols),
        "hidden_layers":    hidden_layers,
        "activation_function": activation_function,
    }
    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=4)

    with open(paths["predict"], "w") as f:
        f.write(_write_predict_script(input_cols, output_cols))

    with open(paths["requirements"], "w") as f:
        f.write("tensorflow\nnumpy\njoblib\nscikit-learn\n")

    with open(paths["readme"], "w") as f:
        f.write(
            "Cara Menjalankan:\n\n"
            "1. Install dependensi:\n"
            "   pip install -r requirements.txt\n\n"
            "2. Jalankan prediksi:\n"
            "   python predict.py\n"
        )

    with zipfile.ZipFile(paths["zip"], "w") as zf:
        zf.write(paths["model"],        arcname="model.h5")
        zf.write(paths["metadata"],     arcname="metadata.json")
        zf.write(paths["scaler_x"],     arcname="scaler_x.pkl")
        zf.write(paths["scaler_y"],     arcname="scaler_y.pkl")
        zf.write(paths["predict"],      arcname="predict.py")
        zf.write(paths["requirements"], arcname="requirements.txt")
        zf.write(paths["readme"],       arcname="README.txt")

    with open(paths["zip"], "rb") as f:
        return f.read()


# =========================================================
# LOAD MODEL DARI ZIP (DEPLOY)
# =========================================================

def load_model_from_zip(zip_file_obj) -> dict:
    """
    Membaca ZIP model dan mengembalikan dict berisi model, scaler, metadata.

    Parameters
    ----------
    zip_file_obj : file-like object
        File ZIP yang diunggah.

    Returns
    -------
    dict dengan key: model, scaler_x, scaler_y, metadata
    """
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "upload.zip")

    with open(zip_path, "wb") as f:
        f.write(zip_file_obj.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    model    = load_model(os.path.join(temp_dir, "model.h5"))
    scaler_x = joblib.load(os.path.join(temp_dir, "scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(temp_dir, "scaler_y.pkl"))

    with open(os.path.join(temp_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return {
        "model":    model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "metadata": metadata,
    }


# =========================================================
# EXPORT HASIL KE EXCEL
# =========================================================

def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """
    Mengonversi DataFrame ke bytes Excel (.xlsx).

    Returns
    -------
    bytes
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.read()