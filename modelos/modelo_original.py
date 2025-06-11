#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refatoração: centraliza caminhos (BASE_DIR / data | logs | results) e
remove imports não utilizados. A lógica original foi preservada.
"""

import os
import shutil
import time
from math import sqrt
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.utils import plot_model

# ────────────────────────────── PATHS ──────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOGS_DIR     = os.path.join(BASE_DIR, "logs")                       			   # *.txt finais de médias
RESULTS_ROOT = os.path.join(BASE_DIR, "results", "LSTM")            			   # subpastas por execução
DATA_DIR = os.path.join(BASE_DIR, "data")                                          # execuções

for d in (DATA_DIR, LOGS_DIR, RESULTS_ROOT):
    d.mkdir(parents=True, exist_ok=True)

# ────────────────────────── EXECUÇÃO ÚNICA ─────────────────────────
run_dir = RESULTS_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir.mkdir(parents=True, exist_ok=True)

# ─────────────── Manipulação dos dados ───────────────
with open("ManDados.txt", "a", encoding="utf-8") as lg:
    lg.write(f"------ Manipulação dos dados iniciada em: {datetime.now()}\n\n")
inicio = time.time()

dataset_path = DATA_DIR / "e1_leonardo.csv"
dataset1 = read_csv(dataset_path, header=0, index_col=0, delimiter=";")
values1 = dataset1.values
values1 = values1[~np.isnan(values1).any(axis=1)].astype("float32")

scaler = MinMaxScaler()
scaled1 = DataFrame(scaler.fit_transform(values1)).values

n_train = 26
train1, test1 = scaled1[:n_train, :], scaled1[n_train:, :]
train_X1, train_y1 = train1[:, :-1], train1[:, -1]
test_X1, test_y1 = test1[:, :-1], test1[:, -1]

train_X1 = train_X1.reshape((-1, 1, train_X1.shape[1]))
test_X1 = test_X1.reshape((-1, 1, test_X1.shape[1]))

t1 = time.time() - inicio
with open("ManDados.txt", "a", encoding="utf-8") as lg:
    lg.write(f"------ Manipulação dos dados encerrada em: {datetime.now()}\n")
    lg.write(f"Tempo de execução: {t1}\n")
shutil.move("ManDados.txt", run_dir / "ManDados.txt")

# ─────────────── Definição & Treinamento Modelo Original ───────────────
with open("DefTreinoMO.txt", "a", encoding="utf-8") as lg:
    lg.write(f"------ Treinamento ModeloOriginal iniciado em: {datetime.now()}\n\n")
inicio = time.time()

model = Sequential()
model.add(LSTM(44, input_shape=(train_X1.shape[1], train_X1.shape[2]), return_sequences=True))
model.add(LSTM(22))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

history = model.fit(
    train_X1, train_y1,
    epochs=5000,
    batch_size=72,
    validation_data=(test_X1, test_y1),
    verbose=0,
    shuffle=False,
)

t2 = time.time() - inicio
with open("DefTreinoMO.txt", "a", encoding="utf-8") as lg:
    lg.write(f"------ Execução total encerrada em: {datetime.now()}\n")
    lg.write(f"Tempo de execução: {t2}\n")
shutil.move("DefTreinoMO.txt", run_dir / "DefTreinoMO.txt")

# ─────────────── Cálculos (sem Keras Tuner) ───────────────
with open("CalcSKT.txt", "a", encoding="utf-8") as lg:
    lg.write(f"------ Cálculos iniciados em: {datetime.now()}\n\n")
inicio = time.time()

yhat1 = model.predict(test_X1)
test_X1_flat = test_X1.reshape(test_X1.shape[0], test_X1.shape[2])

inv_yhat1 = scaler.inverse_transform(concatenate((test_X1_flat, yhat1), axis=1))[:, -1]
inv_y1 = scaler.inverse_transform(
    concatenate((test_X1_flat, test_y1.reshape(-1, 1)), axis=1)
)[:, -1]

rmse          = sqrt(mean_squared_error(inv_y1, inv_yhat1))
desvio_pred   = np.std(inv_yhat1)
variancia_pred = inv_yhat1.var()
desvio_real   = np.std(inv_y1)
variancia_real = inv_y1.var()
r_value       = stats.linregress(inv_y1, inv_yhat1).rvalue

coeffs1 = np.polyfit(inv_y1, inv_yhat1, 5)
r1 = (
    np.sum((np.poly1d(coeffs1)(inv_y1) - inv_yhat1.mean()) ** 2)
    / np.sum((inv_yhat1 - inv_yhat1.mean()) ** 2)
)

t3 = time.time() - inicio
with open("CalcSKT.txt", "a", encoding="utf-8") as lg:
    lg.write(f"------ Cálculos encerrados em: {datetime.now()}\n")
    lg.write(f"Tempo de execução: {t3}\n")
shutil.move("CalcSKT.txt", run_dir / "CalcSKT.txt")

# ─────────────── Figuras & Logs ───────────────
plot_model(model, to_file=str(run_dir / "model_plot_P20-Infestado.png"),
           show_shapes=True, show_layer_names=True)

erro = inv_yhat1 - inv_y1

fig1, _ = pyplot.subplots()
pyplot.boxplot([inv_y1, inv_yhat1, erro], labels=["Real", "Predito", "Erro"])
pyplot.title("P20 - Infestado")
pyplot.savefig(run_dir / "GBoxP_SKT.png", dpi=fig1.get_dpi() * 2)
pyplot.close()

pyplot.scatter(inv_yhat1, inv_y1)
pyplot.xlim(0, 110)
pyplot.ylim(0, 110)
pyplot.plot([inv_y1.min(), inv_yhat1.max()], [inv_y1.min(), inv_yhat1.max()], "red")
pyplot.title("P20 Infestado - Real x Predito")
pyplot.ylabel("Real")
pyplot.xlabel("Predito")
pyplot.savefig(run_dir / "GDisp_SKT.png", dpi=fig1.get_dpi() * 2)
pyplot.close()

with open("Log_SKT.txt", "w", encoding="utf-8") as file:
    file.write("P20 - Infestado\n")
    file.write(f"Test RMSE: {rmse:.3f}\n")
    file.write(f"R2 linear: {r_value ** 2}\n")
    file.write(f"R2 Polinomial: {r1}\n")
    file.write(f"Desvio Real: {desvio_real}\n")
    file.write(f"Variancia Real: {variancia_real}\n")
    file.write(f"Desvio Predito: {desvio_pred}\n")
    file.write(f"Variancia Predito: {variancia_pred}\n")
    file.write("Real\n" + ", ".join(f"{a:.2f}" for a in inv_y1) + "\n")
    file.write("Predito\n" + ", ".join(f"{b:.2f}" for b in inv_yhat1) + "\n")
shutil.move("Log_SKT.txt", run_dir / "Log_SKT.txt")
