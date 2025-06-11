import sys
import os
import time
from datetime import datetime
from math import sqrt

import numpy as np
from keras import Input
from keras.layers import GRU, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from pandas import DataFrame, read_csv
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

try:
    from keras_tuner.tuners import RandomSearch
    HAS_KERAS_TUNER = True
except ImportError:
    HAS_KERAS_TUNER = False

# ---------------------------------------------------------------------------
# General configuration & helper functions
# ---------------------------------------------------------------------------

# Use the folder where this script lives as BASE_DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(BASE_DIR, "data")      # datasets & raw I/O
LOGS_DIR = os.path.join(BASE_DIR, "logs")      # all log files
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for _d in (DATA_DIR, LOGS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# tiny logging helper (appends to <LOGS_DIR>/<filename>)
# ---------------------------------------------------------------------------

def log_message(filename: str, message: str, mode: str = "a") -> None:
    with open(os.path.join(LOGS_DIR, filename), mode, encoding="utf-8") as fh:
        fh.write(message + "\n")

# ---------------------------------------------------------------------------
# CLI argument – number of repetitions
# ---------------------------------------------------------------------------

if len(sys.argv) != 2:
    sys.exit("Usage: python gru_pipeline_generic_paths.py <num_runs>")
try:
    NUM_RUNS = int(sys.argv[1])
except ValueError:
    sys.exit("The argument must be an integer (number of repetitions)")

# For run‑time averages at the very end
acc_data_prep = acc_def_train = acc_calc_skt = acc_random_search = acc_train_kt = acc_calc_ckt = 0.0

# ---------------------------------------------------------------------------
# Main loop – one full experiment per iteration
# ---------------------------------------------------------------------------
for run_idx in range(1, NUM_RUNS + 1):
    # ---------------------------------------------------------------------
    # Per‑run folder under results/ (timestamp + counter suffix)
    # ---------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join(RESULTS_DIR, f"{ts}_{run_idx:02d}")
    os.makedirs(RUN_DIR, exist_ok=True)

    # Simple helper so we do not repeat datetime.now() everywhere
    def _now_str():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # (1) Data manipulation
    # ------------------------------------------------------------------
    data_start = time.time()
    log_message("ManDados.txt", f"------ Manipulação dos dados iniciada em: {_now_str()}")

    dataset_path = os.path.join(DATA_DIR, "e1_Treino.csv")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = read_csv(dataset_path, header=0, index_col=0, delimiter=";")
    values = dataset.values.astype("float32")

    # Strip rows containing NaNs
    values = values[~np.isnan(values).any(axis=1)]

    # Last column is assumed categorical → label‑encode then treat as float
    encoder = LabelEncoder()
    values[:, -1] = encoder.fit_transform(values[:, -1])

    # Normalisation
    scaler = MinMaxScaler()
    values = DataFrame(scaler.fit_transform(values)).values

    # Train / test split & reshape for 3‑D GRU input
    n_train = max(1, len(dataset1) - 12)
    train, test = values[:N_TRAIN], values[N_TRAIN:]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    log_message(
        "ManDados.txt",
        f"------ Manipulação dos dados encerrada em: {_now_str()}\nTempo de execução: {time.time() - data_start:.2f} s\n",
    )
    acc_data_prep += time.time() - data_start

    # ------------------------------------------------------------------
    # (2) Base GRU model definition + training (no KerasTuner yet)
    # ------------------------------------------------------------------
    train_start = time.time()
    log_message("DefTreinoMO.txt", f"------ Definição & Treinamento (run {run_idx}) iniciados em: {_now_str()}")

    model = Sequential([
        Input(shape=(train_X.shape[1], train_X.shape[2])),
        GRU(30, kernel_initializer="normal", return_sequences=True),
        GRU(15, kernel_initializer="normal"),
        Dense(1, kernel_initializer="normal"),
    ])
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    model.fit(
        train_X,
        train_y,
        epochs=5000,
        batch_size=72,
        validation_data=(test_X, test_y),
        verbose=0,
        shuffle=False,
    )

    log_message(
        "DefTreinoMO.txt",
        f"------ Treinamento (run {run_idx}) encerrado em: {_now_str()}\nTempo de execução: {time.time() - train_start:.2f} s\n",
    )
    acc_def_train += time.time() - train_start

    # ------------------------------------------------------------------
    # (3) Post‑training calculations (no KerasTuner)
    # ------------------------------------------------------------------
    calc_start = time.time()
    log_message("CalcSKT.txt", f"------ Cálculos (run {run_idx}) iniciados em: {_now_str()}")

    y_pred = model.predict(test_X, verbose=0)
    # Bring test features back to 2‑D for inverse transform
    test_X_flat = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_y_pred = scaler.inverse_transform(np.hstack((test_X_flat, y_pred)))[:, -1]
    inv_y_true = scaler.inverse_transform(np.hstack((test_X_flat, test_y.reshape(-1, 1))))[:, -1]

    rmse = sqrt(mean_squared_error(inv_y_true, inv_y_pred))
    std_pred, var_pred = np.std(inv_y_pred), inv_y_pred.var()
    std_true, var_true = np.std(inv_y_true), inv_y_true.var()

    _, _, r_val, _, _ = stats.linregress(inv_y_true, inv_y_pred)
    coeffs = np.polyfit(inv_y_true, inv_y_pred, 5)
    poly = np.poly1d(coeffs)
    r_poly = np.sum((poly(inv_y_true) - inv_y_pred.mean()) ** 2) / np.sum((inv_y_pred - inv_y_pred.mean()) ** 2)

    log_message(
        "CalcSKT.txt",
        f"------ Cálculos (run {run_idx}) encerrados em: {_now_str()}\nTempo de execução: {time.time() - calc_start:.2f} s\n",
    )
    acc_calc_skt += time.time() - calc_start

    # ------------------------------------------------------------------
    #  (4) KerasTuner Random Search (optional – skip if not installed)
    # ------------------------------------------------------------------
    if HAS_KERAS_TUNER:
        kt_start = time.time()
        log_message("RandomSearch.txt", f"------ Random Search (run {run_idx}) iniciada em: {_now_str()}")

        def build_model(hp):
            m = Sequential()
            m.add(
                GRU(
                    units=hp.Int("gru_units", 30, 512, 32),
                    activation=hp.Choice("gru_act", ["relu", "tanh"]),
                    return_sequences=True,
                    input_shape=(1, train_X.shape[2]),
                )
            )
            m.add(
                GRU(
                    units=hp.Int("gru_units2", 15, 512, 32),
                    activation=hp.Choice("gru_act", ["relu", "tanh"]),
                )
            )
            if hp.Boolean("dropout"):
                m.add(Dense(1))  # simple linear head when dropout chosen
            m.add(Dense(1, activation="linear"))
            lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
            m.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
            return m

        tuner_dir = os.path.join(RUN_DIR, "KT")
        tuner = RandomSearch(
            build_model,
            objective="val_mse",
            max_trials=5,
            executions_per_trial=2,
            directory=tuner_dir,
            project_name="p20_infestado",
            overwrite=True,
        )

        tuner.search(train_X, train_y, epochs=5, validation_data=(test_X, test_y), verbose=0)
        best_model = tuner.get_best_models(1)[0]
        model = best_model  # replace baseline with tuned one

        log_message(
            "RandomSearch.txt",
            f"------ Random Search (run {run_idx}) encerrada em: {_now_str()}\nTempo de execução: {time.time() - kt_start:.2f} s\n",
        )
        acc_random_search += time.time() - kt_start

        # ------------------------------------------------------------------
        # (5) Re‑train tuned model & fresh evaluation
        # ------------------------------------------------------------------
        tuned_start = time.time()
        model.fit(train_X, train_y, epochs=5000, validation_data=(test_X, test_y), verbose=0)
        acc_train_kt += time.time() - tuned_start

        # Fresh metrics after KerasTuner
        ckt_start = time.time()
        y_pred = model.predict(test_X, verbose=0)
        inv_y_pred = scaler.inverse_transform(np.hstack((test_X_flat, y_pred)))[:, -1]
        rmse = sqrt(mean_squared_error(inv_y_true, inv_y_pred))
        log_message(
            "CalcCKT.txt",
            f"------ Cálculos com KT (run {run_idx}) encerrados em: {_now_str()}\nTempo de execução: {time.time() - ckt_start:.2f} s\n",
        )
        acc_calc_ckt += time.time() - ckt_start

    # ------------------------------------------------------------------
    # (6) Per‑run artefacts (plots & simple log)
    # ------------------------------------------------------------------
    plt.figure()
    plt.boxplot([inv_y_true, inv_y_pred, inv_y_pred - inv_y_true], labels=["Real", "Predito", "Erro"])
    plt.title("P20 - Infestado")
    plt.savefig(os.path.join(RUN_DIR, "boxplot.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.scatter(inv_y_pred, inv_y_true, s=10)
    plt.plot([inv_y_true.min(), inv_y_pred.max()], [inv_y_true.min(), inv_y_pred.max()], "r--")
    plt.title("Real vs Predito – P20 Infestado")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig(os.path.join(RUN_DIR, "scatter.png"), dpi=150)
    plt.close()

    # Compact per‑run summary
    with open(os.path.join(RUN_DIR, "metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"RMSE: {rmse:.4f}\n")
        fh.write(f"R2 linear: {r_val ** 2:.4f}\n")
        fh.write(f"R2 poly (deg5): {r_poly:.4f}\n")
        fh.write(f"STD real/pred: {std_true:.4f} / {std_pred:.4f}\n")

# ---------------------------------------------------------------------------
# Global averages across runs
# ---------------------------------------------------------------------------
summary_path = os.path.join(LOGS_DIR, f"log_medias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(summary_path, "w", encoding="utf-8") as log:
    log.write(f"A média de tempo de manipulação dos dados é: {acc_data_prep / NUM_RUNS:.2f} s\n")
    log.write(f"A média de tempo de Treino base é: {acc_def_train / NUM_RUNS:.2f} s\n")
    log.write(f"A média de tempo de cálculos sem KT é: {acc_calc_skt / NUM_RUNS:.2f} s\n")
    if HAS_KERAS_TUNER and NUM_RUNS:
        log.write(f"A média de tempo de Random Search é: {acc_random_search / NUM_RUNS:.2f} s\n")
        log.write(f"A média de tempo de Treino com KT é: {acc_train_kt / NUM_RUNS:.2f} s\n")
        log.write(f"A média de tempo de cálculos com KT é: {acc_calc_ckt / NUM_RUNS:.2f} s\n")

print("Run(s) complete.")
print(f"Logs folder : {LOGS_DIR}")
print(f"Results root: {RESULTS_DIR}")
