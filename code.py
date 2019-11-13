import math
import random
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

"""Adl"""
def RegleDeciAdl(point,covarianceInv,moy,pi):
    mulcovmoy=numpy.dot(covarianceInv,moy)
    Rd=(numpy.dot(numpy.transpose(point),mulcovmoy)-(1/2)*numpy.dot(numpy.transpose(moy),mulcovmoy)+numpy.log(pi))
    return Rd

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


obs0=numpy.array((obs0))
obs1=numpy.array((obs1))
moyobs0=numpy.array(moyenne(obs0))
moyobs1=numpy.array(moyenne(obs1))
pi0=len(obs0)/(len(obs0)+len(obs1))
pi1=len(obs1)/(len(obs0)+len(obs1))
print('moyenne obs0',moyobs0)
print('moyenne obs1',moyobs1)

#print((numpy.array(obs0)))
#print((numpy.array(obs1)))

covObs0=numpy.cov(obs0.T)
covObs1=numpy.cov(obs1.T)
print(covObs0)
print(covObs1)

covSomme=Sommecovariance(covObs0,covObs1)
print('sommeCov',covSomme)

""" Calcul de la covariance inversée"""
covarianceInv=numpy.linalg.inv(covSomme)
print("matrice cov inversé",covarianceInv)
#print(covarianceInv.dot(covSomme))

point=numpy.array([-10,10])
Rd0=RegleDeciAdl(point,covarianceInv,moyobs0,pi0)
Rd1=RegleDeciAdl(point,covarianceInv,moyobs1,pi1)
print("R0 pour ce point: ",Rd0)
print("R1 pour ce point: ",Rd1)
#Fd=((numpy.transpose(point)*covarianceInv*(moyobs0-moyobs1))-(1/2)*numpy.transpose(moyobs0-moyobs1)*covarianceInv(moyobs0+moyobs1))


z=numpy.concatenate(obs0,obs1)
clf = LinearDiscriminantAnalysis()
clf.fit(z,numpy.array(y))
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd',
  store_covariance=False, tol=0.0001)
print(clf.predict([[-10, 10]]))