import sys

# Obtém o primeiro argumento da linha de comando
argumento = sys.argv[1] 

# Converte o argumento para um inteiro
try:
    valor_inteiro = int(argumento)
except ValueError:
    print("O argumento não é um número inteiro válido.")

import shutil
import os
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import GRU #ADICIONEI
import numpy as np
from datetime import date
import csv
#from folium.plugins import HeatMap
import os
#import folium
import time
import webbrowser
#import branca.colormap as cmp
#from folium.features import DivIcon
#from folium.plugins import FloatImage
from scipy import stats
from sklearn.metrics import r2_score
from datetime import datetime

from keras.layers import Conv1D

#KERASTUNER

from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch

from keras.optimizers import Adam

import random

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

    else: #dropout principal
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

#LSTM

media1=0
media2=0
media3=0
media4=0
media5=0
media6=0
aux=0
now = datetime.now()
nomelogmedia = now
while(aux<valor_inteiro):
	now = datetime.now()
	name4 = now
	os.mkdir(str(name4))
	caminho_destino = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/' + str(name4))
	log = open('ManDados.txt', 'a')
	log.write("------Manipulação dos dados iniciada em: " + str(now) + "\n\n")
	log.close()
	inicio = time.time()


	caminho = r'/home/davilemos/Área de Trabalho/modelos'
	cmh = os.path.join(caminho, 'P20Inf')
	################################################Potreiro 20 Infestado#############################################################
	# load dataset
	dataset1 = read_csv('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/e1_leo_2019.csv', header=0, index_col=0, delimiter=';')
	values1 = dataset1.values
	# integer encode direction
	encoder = LabelEncoder()

	#Drop NA Values
	values1=values1[~np.isnan(values1).any(axis=1)]


	# ensure all data is float
	values1 = values1.astype('float32')


	real1 = values1[:, -1]

	# normalize features
	scaler = MinMaxScaler()
	scaled1 = scaler.fit_transform(values1)

	scaled1 = DataFrame(scaled1)

	#print(scaled)
	#print(scaled2)
	# split into train and test sets
	values1 = scaled1.values

	#n_train = 24
	n_train = 36
	train1 = values1[:n_train, :]
	test1 = values1[n_train:, :]
	#print(train)
	#print(test)
	# split into input and outputs
	train_X1, train_y1 = train1[:, :-1], train1[:, -1]
	test_X1, test_y1 = test1[:, :-1], test1[:, -1]

	#print(train_X)
	#print(train_y)
	#print(test_X1)
	#print(test_y1)

	# reshape input to be 3D [samples, timesteps, features]
	train_X1 = train_X1.reshape((train_X1.shape[0], 1, train_X1.shape[1]))
	test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))

	#print(train_X)
	#print(train_y)
	#print(test_X)
	#print(test_y)

	#print(train_X1.shape, train_y1.shape, test_X1.shape, test_y1.shape)
	#print(train_X1.shape[1])

	now = datetime.now()
	fim = time.time()
	t1 = fim - inicio
	log = open('ManDados.txt', 'a')
	log.write("------ Manipulação dos dados encerrada em: " + str(now) + "\n\nTempo de execução: " + str(t1) )
	log.close()

	caminho_origem = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/ManDados.txt')
	shutil.move(caminho_origem, caminho_destino)

	now = datetime.now()
	log = open('Sintonização.txt', 'a')
	log.write("------Definição KT e Random Search iniciada em: " + str(now) + "\n\n")
	log.close()
	inicio = time.time()

	qualidade = "Ajuste Inadequado"

	while qualidade == "Ajuste Inadequado":
	    learning_rate = random.uniform(0.0, 0.01)
	    dropout = random.choice(["True", "False"])
	    layer1 = random.randint(1, 512)
	    layer2 = random.randint(1, 512)
	    activation = random.choice(["relu", "tanh"])

	    qualidade = AD_classificar_ajuste(dropout, learning_rate, layer1, layer2, activation)
	    print("\n______________________________________\n")
	    print("Layer 1:", layer1)
	    print("Layer 2:", layer2)
	    print("Dropout:", dropout)
	    print("Activation:", activation)
	    print("Learning Rate:", learning_rate)
	    print(qualidade)
	    print("______________________________________\n")

	arquivo = open("hiperparâmetros.txt", "w")
	arquivo.write("Layer 1: " + str(layer1) + "\n")
	arquivo.write("Layer 2: " + str(layer2) + "\n")
	arquivo.write("Dropout: " + str(dropout) + "\n")
	arquivo.write("Activation: " + str(activation) + "\n")
	arquivo.write("Learning Rate: " + str(learning_rate) + "\n")
	arquivo.close()
	
	caminho_origem = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/hiperparâmetros.txt')
	shutil.move(caminho_origem, caminho_destino)

	
	now = datetime.now()
	fim = time.time()
	t4 = fim - inicio
	log = open('Sintonização.txt', 'a')
	log.write("------ Excecução encerrada em: " + str(now) + "\n\nTempo de execução: " + str(t4) )
	log.close()

	caminho_origem = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/Sintonização.txt')
	shutil.move(caminho_origem, caminho_destino)

	from contextlib import redirect_stdout

	now = datetime.now()
	log = open('TreinoMA.txt', 'a')
	log.write("------Modelo Adaptado Excecução iniciada em: " + str(now) + "\n\n")
	log.close()
	inicio = time.time()

	def build_model():
		model = keras.Sequential()
		model.add(layers.LSTM(
		    units=layer1,
		    activation=activation,
		    return_sequences=True,
		    input_shape=(1, 33)
		))
		model.add(layers.LSTM(
		    units=layer2,
		    activation=activation
		))
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

	now = datetime.now()
	fim = time.time()
	t5 = fim - inicio
	log = open('TreinoMA.txt', 'a')
	log.write("------ Excecução Total encerrada em: " + str(now) + "\n\nTempo de execução: " + str(t5) )
	log.close()

	caminho_origem = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/TreinoMA.txt')
	shutil.move(caminho_origem, caminho_destino)

	now = datetime.now()
	log = open('CalcCKT.txt', 'a')
	log.write("------Cálculos com KT iniciados em: " + str(now) + "\n\n")
	log.close()
	inicio = time.time()


	# make a prediction
	yhat1 = model.predict(test_X1)
	test_X1 = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))
	# invert scaling for forecast
	inv_yhat1 = concatenate((test_X1, yhat1), axis=1)
	inv_yhat1 = scaler.inverse_transform(inv_yhat1)
	inv_yhat1 = inv_yhat1[:,-1]
	# invert scaling for actual
	test_y1 = test_y1.reshape((len(test_y1), 1))
	inv_y1 = concatenate((test_X1, test_y1), axis=1)
	inv_y1 = scaler.inverse_transform(inv_y1)
	inv_y1 = inv_y1[:,-1]

	#print(model.summary())

	#from ann_visualizer.visualize import ann_viz

	#ann_viz(model, title="My first neural network")

	# calculate RMSE
	rmse = sqrt(mean_squared_error(inv_y1, inv_yhat1))
	desvioAmostralpred1 = np.std(inv_yhat1) #desvio padrão populacional
	varianciaAmostralpred1 = inv_yhat1.var() #variancia populacional

	desvioAmostralreal1 = np.std(inv_y1) #desvio padrão populacional
	varianciaAmostralreal1 = inv_y1.var() #variancia populacional

	slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y1, inv_yhat1)


	coeffs1 = np.polyfit(inv_y1, inv_yhat1, 5)
	p1 = np.poly1d(coeffs1)
	# fit values, and mean
	yhat1 = p1(inv_y1)							  # or [p(z) for z in x]
	ybar1 = np.sum(inv_yhat1)/len(inv_yhat1)		   # or sum(y)/len(y)
	ssreg1 = np.sum((yhat1-ybar1)**2)			   # or sum([ (yihat - ybar)**2 for yihat in yhat])
	sstot1 = np.sum((inv_yhat1 - ybar1)**2)			 # or sum([ (yi - ybar)**2 for yi in y])
	r1 = ssreg1 / sstot1

	now = datetime.now()
	fim = time.time()
	t6 = fim - inicio
	log = open('CalcCKT.txt', 'a')
	log.write("------ Cálculos encerrados em: " + str(now) + "\n\nTempo de execução: " + str(t6) )
	log.close()

	caminho_origem = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/CalcCKT.txt')
	shutil.move(caminho_origem, caminho_destino)

	erro = []
	for x in inv_yhat1:
		erro = inv_yhat1 - inv_y1


	#Boxplot
	fig1, ax1 = pyplot.subplots()
	pyplot.boxplot([inv_y1, inv_yhat1, erro], labels=['Real', 'Predito', 'Erro'])
	pyplot.title('P20 - Infestado')
	dpi = fig1.get_dpi()

	pyplot.savefig(f'GBoxP_CKT.png', dpi=dpi*2)
	pyplot.close()

	#Gráfico de Dispersão
	pyplot.scatter(inv_yhat1, inv_y1)
	range = [inv_y1.min(), inv_yhat1.max()]
	pyplot.xlim(left=0)
	pyplot.xlim(right=110
	)
	pyplot.ylim(bottom=0)
	pyplot.ylim(top=110)
	pyplot.plot(range, range, 'red')
	pyplot.title('P20 Infestado - Real x Predito')
	pyplot.ylabel('Real')
	pyplot.xlabel('Predito')
	pyplot.savefig(f'GDisp_CKT.png', dpi=dpi*2)


	pyplot.close()

	caminho_origem = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/GBoxP_CKT.png')
	shutil.move(caminho_origem, caminho_destino)
	caminho_origem = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/GDisp_CKT.png')
	shutil.move(caminho_origem, caminho_destino)
	
	#print('P20 - Infestado')
	#print("R2 linear", r_value ** 2)
	#print("R2 Polinomial:", r1)
	#print('Test RMSE: %.3f' % rmse)
	#print("Desvio Real", desvioAmostralreal1)
	#print("Variancia Real", varianciaAmostralreal1)
	#print("Desvio Predito", desvioAmostralpred1)
	#print("Variancia Predito", varianciaAmostralpred1)
	#print('Real')
	#print(inv_y1)
	#print('Predito')
	#print(inv_yhat1)

	file = open('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/Log_CKT.txt', "w")
	file.write('P20 - Infestado' + '\n')
	file.write('Test RMSE:' + '%.3f' % rmse + '\n')
	file.write('R2 linear:' + str(r_value ** 2) + '\n')
	file.write("R2 Polinomial:" + str(r1) + '\r')
	file.write("Desvio Real:" + str(desvioAmostralreal1) + '\n')
	file.write("Variancia Real:" + str(varianciaAmostralreal1) + '\n')
	file.write("Desvio Predito:" + str(desvioAmostralpred1) + '\n')
	file.write("Variancia Predito:" + str(varianciaAmostralpred1) + '\n')
	file.write('Real' + '\n')
	for a in inv_y1:
		file.write(str('%.2f' % a) + ',' + ' ')
	file.write('\n' + 'Predito' + '\n')
	for b in inv_yhat1:
		file.write(str('%.2f' % b) + ',' + ' ')
	file.close()

	caminho_origem = str('/home/davilemos/Área de Trabalho/modelos/modelo-original (ÂNDERSON)/LSTM/Log_CKT.txt')
	shutil.move(caminho_origem, caminho_destino)

	media1=t1+media1
	media4=t4+media4
	media5=t5+media5
	media6=t6+media6

	aux = aux+1
	
media1=media1/valor_inteiro
media4=media4/valor_inteiro
media5=media5/valor_inteiro
media6=media6/valor_inteiro
	
log = open(f'log_medias_{str(nomelogmedia)}.txt', 'a')

log.write("A média de tempo de manipulação dos dados é: " + str(media1) + "s\n\n")
log.write("A média de tempo de execução da sintonização é: " + str(media4) + "s\n\n")
log.write("A média de tempo de execução do Treinamento é: " + str(media5) + "s\n\n")
log.write("A média de tempo dos cálculos é: " + str(media6) + "s\n\n")

log.close()


