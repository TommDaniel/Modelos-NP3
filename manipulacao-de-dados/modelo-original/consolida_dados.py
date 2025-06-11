import psycopg2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, 'log.txt')
now = datetime.now()

with open(log_path, 'a', encoding='utf-8') as log:
	log.write("------ Consolida dados ModeloOriginal Excecução iniciada em: " + str(now) + "\n\n")

try:
#Conecta com o BD
	con = psycopg2.connect("host='localhost' port='5432' dbname='db_2020' user='postgres' password='030902'")
	#con = psycopg2.connect("host='localhost' port='5432' dbname='pastagemOutlier' user='postgres' password='123456'")
	cur = con.cursor()
		
except:
	log.write("\nFalha na conexão com o BD!")

file = open("selectent.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	ant=None
except:
	print("Erro - Nao foi possivel acessar os dados no BD")

file1 = open("selectmstotalD.sql", 'r')
sql1 = " ".join(file1.readlines())

cur.execute(sql1)
con.commit()

try:
	dados1= cur.fetchall()
except:
	print("Erro - Nao foi possivel acessar os dados no BD")

file2 = open("selectmstotalF.sql", 'r')
sql2 = " ".join(file2.readlines())

cur.execute(sql2)
con.commit()

try:
	dados2= cur.fetchall()
except:
	print("Erro - Nao foi possivel acessar os dados no BD")


mstotalD = [float(d[0]) for d in dados1]
quartilD = mquantiles(mstotalD, prob=[0.75])
print(quartilD)

mstotalF = [float(d[0]) for d in dados2]
quartilF = mquantiles(mstotalF, prob=[0.75])
print(quartilF)


for x in tdo:

	if (ant == None): 
		ant=x
		continue
	#print(x)
	if (ant[1] != x[1]) and x[0]=='D':
		c=x[1]-ant[1]

		file = open("selectclima.sql", 'r')
		sqlclima = " ".join(file.readlines())

		data = (ant[1].isoformat(), x[1].isoformat())

		cur.execute(sqlclima, data)
		con.commit()
		
		clima= cur.fetchall()
		
		if x[3] > quartilD:
			if ant[3] > quartilF:
				#presente
				#ent.append([x[1], (x[1]-ant[1]).days, x[2], ant[2], x[4], ant[3], clima[0][0], clima[0][7],
				#		   clima[0][14], clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], 
				#		   clima[0][16], clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], 
				#		   clima[0][18], clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], 
				#		   clima[0][20], clima[0][21], clima[0][22], clima[0][23],clima[0][24], clima[0][25], 
				#		   clima[0][26],clima[0][27], clima[0][28], clima[0][29], x[3]])
				#futuro
				ent.append([x[1], (x[1]-ant[1]).days, ant[2], '%.2f' % quartilF[0], clima[0][0], clima[0][7], clima[0][14],
							clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], clima[0][16], 
							clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], clima[0][18], 
							clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], clima[0][20], 
							clima[0][21], clima[0][22], clima[0][23], clima[0][24], clima[0][25], clima[0][26],
							clima[0][27], clima[0][28], clima[0][29], '%.2f' % quartilD[0]])
			else:
				#presente
				#ent.append([x[1], (x[1]-ant[1]).days, x[2], ant[2], x[4], ant[3], clima[0][0], clima[0][7],
				#		   clima[0][14], clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], 
				#		   clima[0][16], clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], 
				#		   clima[0][18], clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], 
				#		   clima[0][20], clima[0][21], clima[0][22], clima[0][23],clima[0][24], clima[0][25], 
				#		   clima[0][26],clima[0][27], clima[0][28], clima[0][29], x[3]])
				#futuro
				ent.append([x[1], (x[1]-ant[1]).days, ant[2], ant[3], clima[0][0], clima[0][7], clima[0][14],
							clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], clima[0][16], 
							clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], clima[0][18], 
							clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], clima[0][20], 
							clima[0][21], clima[0][22], clima[0][23], clima[0][24], clima[0][25], clima[0][26],
							clima[0][27], clima[0][28], clima[0][29], '%.2f' % quartilD[0]])				
		else:
			if ant[3] > quartilF:
				#presente
				#ent.append([x[1], (x[1]-ant[1]).days, x[2], ant[2], x[4], ant[3], clima[0][0], clima[0][7],
				#		   clima[0][14], clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], 
				#		   clima[0][16], clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], 
				#		   clima[0][18], clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], 
				#		   clima[0][20], clima[0][21], clima[0][22], clima[0][23],clima[0][24], clima[0][25], 
				#		   clima[0][26],clima[0][27], clima[0][28], clima[0][29], x[3]])
				#futuro
				ent.append([x[1], (x[1]-ant[1]).days, ant[2], '%.2f' % quartilF[0], clima[0][0], clima[0][7], clima[0][14],
							clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], clima[0][16], 
							clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], clima[0][18], 
							clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], clima[0][20], 
							clima[0][21], clima[0][22], clima[0][23], clima[0][24], clima[0][25], clima[0][26],
							clima[0][27], clima[0][28], clima[0][29], x[3]])
			else:
				#presente
				#ent.append([x[1], (x[1]-ant[1]).days, x[2], ant[2], x[4], ant[3], clima[0][0], clima[0][7],
				#		   clima[0][14], clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], 
				#		   clima[0][16], clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], 
				#		   clima[0][18], clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], 
				#		   clima[0][20], clima[0][21], clima[0][22], clima[0][23],clima[0][24], clima[0][25], 
				#		   clima[0][26],clima[0][27], clima[0][28], clima[0][29], x[3]])
				#futuro
				ent.append([x[1], (x[1]-ant[1]).days, ant[2], ant[3], clima[0][0], clima[0][7], clima[0][14],
							clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], clima[0][16], 
							clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], clima[0][18], 
							clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], clima[0][20], 
							clima[0][21], clima[0][22], clima[0][23], clima[0][24], clima[0][25], clima[0][26],
							clima[0][27], clima[0][28], clima[0][29], x[3]])		  
	ant=x
	

import csv

#presente
#header = ['data', 'numerodias', 'alturamedia', 'alturamediaanterior', 'pcanonni', 'mstotalanterior', 
#		  'tmin', 'dptmin', 'vartmin', 'tmed', 'dptmed', 'vartmed', 'tmax', 'dptmax', 'vartmax', 
#		  'umidade', 'dpumidade', 'varumidade', 'velocidadevento', 'dpvelocidadevento', 'varvelovidadevento', 
#		  'radiacaosolar', 'dpradiacaosolar', 'varradiacaosolar', 'chuva', 'dpchuva', 'varchuva', 'somatermica', 
#		  'dpsomatermica', 'varsomatermica', 'def', 'dpdef', 'vardef', 'exc', 'dpexc', 'varexc', 'taxaacumulo']

#futuro
header = ['data', 'numerodias', 'alturamediaanterior', 'mstotalanterior', 'tmin', 'dptmin', 'vartmin', 'tmed', 
		   'dptmed', 'vartmed', 'tmax', 'dptmax', 'vartmax', 'umidade', 'dpumidade', 'varumidade', 'velocidadevento', 
		   'dpvelocidadevento', 'varvelovidadevento', 'radiacaosolar', 'dpradiacaosolar', 'varradiacaosolar', 'chuva', 
		   'dpchuva', 'varchuva', 'somatermica', 'dpsomatermica', 'varsomatermica', 'def', 'dpdef', 'vardef', 'exc', 
		   'dpexc', 'varexc', 'taxaacumulo']


#REVERSE
#ent.reverse()

csv_path = os.path.join(DATA_DIR, 'e1_Treino.csv')
with open(csv_path, 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)

now = datetime.now()

with open(log_path, 'a', encoding='utf-8') as log:
	log.write("------Consolida dados ModeloOriginal Excecução terminada em: " + str(now) + "\n\n")

