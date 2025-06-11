#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
import random
import time
from math import sqrt
from datetime import datetime

import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy import stats
from tensorflow import keras
from tensorflow.keras import layers

# ────────────────────────────── PATHS ──────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))          # pasta do script
LOGS_DIR     = os.path.join(BASE_DIR, "logs")                       			   # *.txt finais de médias
RESULTS_ROOT = os.path.join(BASE_DIR, "results", "LSTM")            			   # subpastas por execução
DATA_DIR = os.path.join(BASE_DIR, "data")      # datasets & raw I/O

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

# ───────────────────────— FUNÇÃO DE CLASSIFICAÇÃO ─────────────────────────
def AD_classificar_ajuste(dropout, learning_rate, layer1, layer2, activation):
    if dropout == "True":
        if learning_rate <= 0.008:
            if learning_rate <= 0.001:
                return "Ajuste Adequado"
            else:
                if learning_rate <= 0.006:
                    return "Ajuste Inadequado"
                else:
                    if activation == "relu":
                        if layer2 <= 89:
                            return "Ajuste Adequado"
                        else:
                            if learning_rate <= 0.007:
                                return "Ajuste Adequado"
                            else:
                                return "Ajuste Inadequado"
                    else:
                        return "Ajuste Inadequado"
        else:
            return "Ajuste Adequado"
    else:
        if layer2 <= 185:
            if learning_rate <= 0.002:
                if learning_rate <= 0.002:
                    return "Ajuste Adequado"
                else:
                    return "Ajuste Inadequado"
            else:
                if activation == "relu":
                    if learning_rate <= 0.007:
                        if learning_rate <= 0.004:
                            return "Ajuste Adequado"
                        else:
                            if layer1 <= 274:
                                return "Ajuste Adequado"
                            else:
                                return "Ajuste Adequado"
                    else:
                        return "Ajuste Inadequado"
                else:
                    return "Ajuste Adequado"
        else:
            if learning_rate <= 0.005:
                return "Ajuste Adequado"
            else:
                if activation == "relu":
                    if layer2 <= 233:
                        return "Ajuste Adequado"
                    else:
                        return "Ajuste Adequado"
                else:
                    return "Ajuste Adequado"

# ─────────────────────────────── PROGRAMA ─────────────────────────────────
# primeiro argumento = número de repetições
argumento = sys.argv[1]

try:
    valor_inteiro = int(argumento)
except ValueError:
    print("O argumento não é um número inteiro válido.")
    sys.exit(1)

media1 = media4 = media5 = media6 = 0
aux = 0
nomelogmedia = datetime.now()

while aux < valor_inteiro:
    # pasta da execução individual
    run_dir = os.path.join(RESULTS_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    # ─────────────── Manipulação dos dados ───────────────
    now = datetime.now()
    with open("ManDados.txt", "a", encoding="utf-8") as log:
        log.write(f"------ Manipulação dos dados iniciada em: {now}\n\n")
    inicio = time.time()

    dataset1 = read_csv(
        os.path.join(DATA_DIR, "e1_Treino.csv"),
        header=0,
        index_col=0,
        delimiter=";",
    )
    values1 = dataset1.values
    values1 = values1[~np.isnan(values1).any(axis=1)].astype("float32")
    real1 = values1[:, -1]

    scaler = MinMaxScaler()
    scaled1 = scaler.fit_transform(values1)
    scaled1 = DataFrame(scaled1)
    values1 = scaled1.values

    n_train = 36
    train1, test1 = values1[:n_train, :], values1[n_train:, :]
    train_X1, train_y1 = train1[:, :-1], train1[:, -1]
    test_X1, test_y1 = test1[:, :-1], test1[:, -1]

    train_X1 = train_X1.reshape((train_X1.shape[0], 1, train_X1.shape[1]))
    test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))

    t1 = time.time() - inicio
    with open("ManDados.txt", "a", encoding="utf-8") as log:
        log.write(f"------ Manipulação dos dados encerrada em: {datetime.now()}\n\n")
        log.write(f"Tempo de execução: {t1}\n")
    shutil.move(os.path.join(BASE_DIR, "ManDados.txt"), run_dir)

    # ─────────────── Sintonização (Random Search manual) ───────────────
    with open("Sintonização.txt", "a", encoding="utf-8") as log:
        log.write(f"------ Sintonização iniciada em: {datetime.now()}\n\n")
    inicio = time.time()

    qualidade = "Ajuste Inadequado"
    while qualidade == "Ajuste Inadequado":
        learning_rate = random.uniform(0.0, 0.01)
        dropout = random.choice(["True", "False"])
        layer1 = random.randint(1, 512)
        layer2 = random.randint(1, 512)
        activation = random.choice(["relu", "tanh"])
        qualidade = AD_classificar_ajuste(dropout, learning_rate, layer1, layer2, activation)

    with open("hiperparâmetros.txt", "w", encoding="utf-8") as f:
        f.write(f"Layer 1: {layer1}\nLayer 2: {layer2}\nDropout: {dropout}\n")
        f.write(f"Activation: {activation}\nLearning Rate: {learning_rate}\n")

    shutil.move(os.path.join(BASE_DIR, "hiperparâmetros.txt"), run_dir)

    t4 = time.time() - inicio
    with open("Sintonização.txt", "a", encoding="utf-8") as log:
        log.write(f"------ Execução encerrada em: {datetime.now()}\n\n")
        log.write(f"Tempo de execução: {t4}\n")
    shutil.move(os.path.join(BASE_DIR, "Sintonização.txt"), run_dir)

    # ─────────────── Treinamento do modelo ───────────────
    with open("TreinoMA.txt", "a", encoding="utf-8") as log:
        log.write(f"------ Treinamento iniciado em: {datetime.now()}\n\n")
    inicio = time.time()

    def build_model():
        model = keras.Sequential()
        model.add(
            layers.LSTM(
                units=layer1,
                activation=activation,
                return_sequences=True,
                input_shape=(1, 33),
            )
        )
        model.add(layers.LSTM(units=layer2, activation=activation))
        if dropout == "True":
            model.add(layers.Dropout(rate=0.25))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    model = build_model()
    history = model.fit(train_X1, train_y1, epochs=5000, validation_data=(test_X1, test_y1))

    t5 = time.time() - inicio
    with open("TreinoMA.txt", "a", encoding="utf-8") as log:
        log.write(f"------ Execução total encerrada em: {datetime.now()}\n\n")
        log.write(f"Tempo de execução: {t5}\n")
    shutil.move(os.path.join(BASE_DIR, "TreinoMA.txt"), run_dir)

    # ─────────────── Cálculos de métricas ───────────────
    with open("CalcCKT.txt", "a", encoding="utf-8") as log:
        log.write(f"------ Cálculos iniciados em: {datetime.now()}\n\n")
    inicio = time.time()

    yhat1 = model.predict(test_X1)
    test_X1_flat = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))

    inv_yhat1 = scaler.inverse_transform(concatenate((test_X1_flat, yhat1), axis=1))[:, -1]
    inv_y1 = scaler.inverse_transform(
        concatenate((test_X1_flat, test_y1.reshape((len(test_y1), 1))), axis=1)
    )[:, -1]

    rmse = sqrt(mean_squared_error(inv_y1, inv_yhat1))
    desvioAmostralpred1 = np.std(inv_yhat1)
    varianciaAmostralpred1 = inv_yhat1.var()
    desvioAmostralreal1 = np.std(inv_y1)
    varianciaAmostralreal1 = inv_y1.var()
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y1, inv_yhat1)

    coeffs1 = np.polyfit(inv_y1, inv_yhat1, 5)
    p1 = np.poly1d(coeffs1)
    yhat_poly = p1(inv_y1)
    ybar1 = np.sum(inv_yhat1) / len(inv_yhat1)
    ssreg1 = np.sum((yhat_poly - ybar1) ** 2)
    sstot1 = np.sum((inv_yhat1 - ybar1) ** 2)
    r1 = ssreg1 / sstot1

    t6 = time.time() - inicio
    with open("CalcCKT.txt", "a", encoding="utf-8") as log:
        log.write(f"------ Cálculos encerrados em: {datetime.now()}\n\n")
        log.write(f"Tempo de execução: {t6}\n")
    shutil.move(os.path.join(BASE_DIR, "CalcCKT.txt"), run_dir)

    # ─────────────── Gráficos ───────────────
    erro = inv_yhat1 - inv_y1

    fig1, ax1 = pyplot.subplots()
    pyplot.boxplot([inv_y1, inv_yhat1, erro], labels=["Real", "Predito", "Erro"])
    pyplot.title("P20 - Infestado")
    fig1_path = os.path.join(run_dir, "GBoxP_CKT.png")
    pyplot.savefig(fig1_path, dpi=fig1.get_dpi() * 2)
    pyplot.close()

    pyplot.scatter(inv_yhat1, inv_y1)
    pyplot.xlim(0, 110)
    pyplot.ylim(0, 110)
    pyplot.plot([inv_y1.min(), inv_yhat1.max()], [inv_y1.min(), inv_yhat1.max()], "red")
    pyplot.title("P20 Infestado - Real x Predito")
    pyplot.ylabel("Real")
    pyplot.xlabel("Predito")
    disp_path = os.path.join(run_dir, "GDisp_CKT.png")
    pyplot.savefig(disp_path, dpi=fig1.get_dpi() * 2)
    pyplot.close()

    # ─────────────── Log detalhado da execução ───────────────
    with open("Log_CKT.txt", "w", encoding="utf-8") as file:
        file.write("P20 - Infestado\n")
        file.write(f"Test RMSE:{rmse:.3f}\n")
        file.write(f"R2 linear:{r_value ** 2}\n")
        file.write(f"R2 Polinomial:{r1}\n")
        file.write(f"Desvio Real:{desvioAmostralreal1}\n")
        file.write(f"Variancia Real:{varianciaAmostralreal1}\n")
        file.write(f"Desvio Predito:{desvioAmostralpred1}\n")
        file.write(f"Variancia Predito:{varianciaAmostralpred1}\n")
        file.write("Real\n")
        file.write(", ".join(f"{a:.2f}" for a in inv_y1) + "\n")
        file.write("Predito\n")
        file.write(", ".join(f"{b:.2f}" for b in inv_yhat1) + "\n")
    shutil.move(os.path.join(BASE_DIR, "Log_CKT.txt"), run_dir)

    media1 += t1
    media4 += t4
    media5 += t5
    media6 += t6
    aux += 1

# ─────────────────────── Métricas Médias Gerais ───────────────────────
media1 /= valor_inteiro
media4 /= valor_inteiro
media5 /= valor_inteiro
media6 /= valor_inteiro

with open(
    os.path.join(LOGS_DIR, f"log_medias_{nomelogmedia.strftime('%Y-%m-%d_%H-%M-%S')}.txt"),
    "a",
    encoding="utf-8",
) as log:
    log.write(f"A média de tempo de manipulação dos dados é: {media1}s\n\n")
    log.write(f"A média de tempo de execução da sintonização é: {media4}s\n\n")
    log.write(f"A média de tempo de execução do treinamento é: {media5}s\n\n")
    log.write(f"A média de tempo dos cálculos é: {media6}s\n\n")
