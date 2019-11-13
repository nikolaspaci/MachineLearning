import math
import random
import numpy
import matplotlib.pyplot as plt
import pandas as pd


""" Calcul de la moyenne d'une liste"""
def moyenne(x):
    moyX1=0
    moyX2=0
    moy=[]
    for i in range(0,len(x)):
        moyX1+=x[i][0]
        moyX2+=x[i][1]
    moy.append([moyX1/len(x)])
    moy.append([moyX2/len(x)])
    print(moy)
    return moy

""" Calcul de la covariance"""
def covariance(X,Moy):
    cov=0
    matcov=[]
    for i in range(0,len(X)):
        cov+=(x[i]-moyX)*(y[i]-moyY)
    return cov/len(x)

""" Calcul somme covariances des classes"""
#covarianceSomme
def Sommecovariance(X,Y):
    z=(((len(X)*X)+(len(Y)*Y))/(len(X)+len(Y)))
    return z

#1 Chargement des données et Nuage de points
WS = pd.read_csv('dataset1.csv',',')
x1=numpy.array(WS['X1'])
x2=numpy.array(WS['X2'])
y=numpy.array(WS['y'])

plt.title("Nuage de points des données extraites")
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1,x2)
#plt.show()

#2 ADL

"""Rassemblement des des observations de meme classe"""
obs0=[]
obs1=[]
for i in range(0,len(x1)):
    if y[i]==0.0:
        obs0.append([x1[i],x2[i]])
    else:
        obs1.append([x1[i],x2[i]])
#print('moyenne obs0',moyenne(obs0))
#print('moyenne obs1',moyenne(obs1))

#print((numpy.array(obs0)))
#print((numpy.array(obs1)))

covObs0=numpy.cov(numpy.array(obs0))
covObs1=numpy.cov(numpy.array(obs1))
#print(covObs0)
#print(covObs1)

covSomme=Sommecovariance(covObs0,covObs1)
#print('sommeCov',covSomme)

""" Calcul de la covariance inversée"""
covarianceInv=numpy.linalg.inv(covSomme)
#print("matrice cov inversé",covarianceInv)


