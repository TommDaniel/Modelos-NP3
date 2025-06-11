# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0=all, 1=info, 2=warning, 3=error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # se quiser desativar as oneDNN custom ops
import time
from datetime import datetime
from math import sqrt

import numpy as np
from keras import Input
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------------------------------------------------------------------
# Directory configuration
# ---------------------------------------------------------------------------
# Usa a pasta onde o script está salvo como BASE_DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(BASE_DIR, "data")      # datasets & raw I/O
LOGS_DIR = os.path.join(BASE_DIR, "logs")      # all log files
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # run‑specific artefacts

# Guarantee required folders exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Run‑specific sub‑folder for plots, model diagrams, etc.
now_ts = datetime.now()
RUN_DIR = os.path.join(RESULTS_DIR, now_ts.strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log_message(filename: str, message: str, mode: str = "a") -> None:
    """Append `message` to <BASE_DIR>/logs/filename."""
    path = os.path.join(LOGS_DIR, filename)
    with open(path, mode, encoding="utf-8") as fh:
        fh.write(message + "\n")

# ---------------------------------------------------------------------------
# 1. Data manipulation
# ---------------------------------------------------------------------------
log_message("ManDados.txt", f"------ Manipulação dos dados iniciada em: {now_ts}")
start_data = time.time()

# Load dataset (now expected under data/)
dataset_path = os.path.join(DATA_DIR, "e1_Treino.csv")
dataset1 = read_csv(dataset_path, header=0, index_col=0, delimiter=";")
values1 = dataset1.values
encoder = LabelEncoder()

# Drop NA values
values1 = values1[~np.isnan(values1).any(axis=1)]

# Ensure all data is float
values1 = values1.astype("float32")
values1[:, -1] = encoder.fit_transform(values1[:, -1])

# Target variable (last column)
real1 = values1[:, -1]

# Normalise features
scaler = MinMaxScaler()
scaled1 = scaler.fit_transform(values1)
values1 = DataFrame(scaled1).values

# Train/test split
n_train = max(1, len(dataset1) - 12)
train1, test1 = values1[:n_train], values1[n_train:]
train_X1, train_y1 = train1[:, :-1], train1[:, -1]
test_X1, test_y1 = test1[:, :-1], test1[:, -1]

# Reshape for LSTM: [samples, timesteps, features]
train_X1 = train_X1.reshape((train_X1.shape[0], 1, train_X1.shape[1]))
test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))

log_message(
    "ManDados.txt",
    f"------ Manipulação dos dados encerrada em: {datetime.now()}\nTempo de execução: {time.time() - start_data}\n",
)

# ---------------------------------------------------------------------------
# 2. Model definition and training
# ---------------------------------------------------------------------------
log_message("DefTreinoMO.txt", f"------ Definição & Treinamento iniciados em: {datetime.now()}")
start_train = time.time()

model = Sequential([
    Input(shape=(train_X1.shape[1], train_X1.shape[2])),
    LSTM(44, kernel_initializer="normal", return_sequences=True),
    LSTM(22, kernel_initializer="normal"),
    Dense(1, kernel_initializer="normal")
])
model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.fit(
    train_X1,
    train_y1,
    epochs=5000,
    batch_size=72,
    validation_data=(test_X1, test_y1),
    verbose=1,
    shuffle=False,
)

log_message(
    "DefTreinoMO.txt",
    f"------ Treinamento encerrado em: {datetime.now()}\nTempo de execução: {time.time() - start_train}\n",
)

# ---------------------------------------------------------------------------
# 3. Post‑training calculations (no Keras Tuner)
# ---------------------------------------------------------------------------
log_message("CalcSKT.txt", f"------ Cálculos iniciados em: {datetime.now()}")
start_calc = time.time()

yhat1 = model.predict(test_X1)

# Flatten test features back to 2‑D for inverse transform
test_X1_flat = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))
inv_yhat1 = scaler.inverse_transform(np.hstack((test_X1_flat, yhat1)))[:, -1]
inv_y1 = scaler.inverse_transform(np.hstack((test_X1_flat, test_y1.reshape(-1, 1))))[:, -1]

rmse = sqrt(mean_squared_error(inv_y1, inv_yhat1))
std_pred, var_pred = np.std(inv_yhat1), inv_yhat1.var()
std_real, var_real = np.std(inv_y1), inv_y1.var()

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y1, inv_yhat1)
coeffs1 = np.polyfit(inv_y1, inv_yhat1, 5)
r1 = np.sum((np.poly1d(coeffs1)(inv_y1) - inv_yhat1.mean()) ** 2) / np.sum((inv_yhat1 - inv_yhat1.mean()) ** 2)

log_message(
    "CalcSKT.txt",
    f"------ Cálculos encerrados em: {datetime.now()}\nTempo de execução: {time.time() - start_calc}\n",
)

# ---------------------------------------------------------------------------
# 4. Visualisations & artefacts (saved to results/)
# ---------------------------------------------------------------------------
try:
    from keras.utils import plot_model
    plot_model(
        model,
        to_file=os.path.join(RUN_DIR, "model_plot.png"),
        show_shapes=True,
        show_layer_names=True,
    )
except (ImportError, OSError) as e:
    log_message("CalcSKT.txt", f"Plot_model skipped: {e}")

erro = inv_yhat1 - inv_y1

fig1, ax1 = plt.subplots()
ax1.boxplot([inv_y1, inv_yhat1, erro], tick_labels=["Real", "Predito", "Erro"])
ax1.set_title("P20 - Infestado")
fig1.savefig(os.path.join(RUN_DIR, "GBoxP_SKT.png"), dpi=fig1.dpi * 2)
plt.close(fig1)

plt.scatter(inv_yhat1, inv_y1)
plt.plot([inv_y1.min(), inv_yhat1.max()], [inv_y1.min(), inv_yhat1.max()], "red")
plt.xlim(0, 110)
plt.ylim(0, 110)
plt.title("P20 Infestado - Real x Predito")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig(os.path.join(RUN_DIR, "GDisp_SKT.png"), dpi=fig1.dpi * 2)
plt.close()

# ---------------------------------------------------------------------------
# 5. Final metrics log
# ---------------------------------------------------------------------------
with open(os.path.join(LOGS_DIR, "Log_SKT.txt"), "w", encoding="utf-8") as f:
    f.write("P20 - Infestado\n")
    f.write(f"Test RMSE: {rmse:.3f}\n")
    f.write(f"R2 linear: {r_value ** 2:.4f}\n")
    f.write(f"R2 Polinomial: {r1:.4f}\n")
    f.write(f"Desvio Real: {std_real:.4f}\n")
    f.write(f"Variancia Real: {var_real:.4f}\n")
    f.write(f"Desvio Predito: {std_pred:.4f}\n")
    f.write(f"Variancia Predito: {var_pred:.4f}\n")
    f.write("Real\n" + ", ".join(f"{v:.2f}" for v in inv_y1) + "\n")
    f.write("Predito\n" + ", ".join(f"{v:.2f}" for v in inv_yhat1) + "\n")

print(f"Run complete. Logs in: {LOGS_DIR}\nArtefacts in: {RUN_DIR}")
