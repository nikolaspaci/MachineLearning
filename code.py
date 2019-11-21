import math
import random
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

""" Calcul de la moyenne d'une liste"""
def moyenne(x):
    moy=[]
    moyX1=0
    moyX2=0
    for index,rows in x.iterrows():
        moyX1+=rows['X1']
        moyX2+=rows['X2']
    moy.append([moyX1/len(x)])
    moy.append([moyX2/len(x)])
    return numpy.array(moy)


""" Calcul somme covariances des classes"""
def Sommecovariance(X,Y):
    z=(((len(X)*X)+(len(Y)*Y))/(len(X)+len(Y)))
    return z

"""Adl"""
def RegleDeciAdl(point,covarianceInv,moy,pi):
    mulcovmoy=numpy.dot(covarianceInv,moy)
    Rd=(numpy.dot(numpy.transpose(point),mulcovmoy)-(1/2)*numpy.dot(numpy.transpose(moy),mulcovmoy)+numpy.log(pi))
    return Rd

"""Tracer de la frontière de décision"""
def tracerFrontiereDecision(moyobs0,moyobs1,pi0,pi1,covarianceInv):
    b=numpy.transpose(moyobs0-moyobs1)
    b=numpy.dot(b,covarianceInv)
    b=numpy.dot(b,moyobs0+moyobs1)
    b=b+math.log(pi0/pi1)
    b=numpy.dot(b,-0.5)

    w=numpy.dot(covarianceInv,moyobs0-moyobs1)
    x1fd=-b/w[0]
    x2fd=-b/w[1]

    p=[]
    p.append(x2fd[0][0])
    p.append(0)
    z=[]
    z.append(0)
    z.append(x1fd[0][0])
    plt.plot(z,p,label='Frontiere décision ADL réalisé')

##1 Chargement des données et Nuage de points
WS = pd.read_csv('dataset1.csv',',')

plt.title("Nuage de points des données extraites")
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(WS['X1'],WS['X2'],c=WS['y'])

##2 ADL
obs0=WS.loc[WS['y']==0.0,['X1','X2']]
obs1=WS.loc[WS['y']==1.0,['X1','X2']]

moyobs0=moyenne(obs0)
moyobs1=moyenne(obs1)
pi0=len(obs0)/(len(WS))
pi1=len(obs1)/(len(WS))
#print('moyenne obs0',moyobs0)
#print('moyenne obs1',moyobs1)

covObs0=numpy.cov(obs0.T)
covObs1=numpy.cov(obs1.T)
#print(covObs0)
#print(covObs1)

covSomme=Sommecovariance(covObs0,covObs1)
#print('sommeCov',covSomme)

""" Calcul de la covariance inversée"""
covarianceInv=numpy.linalg.pinv(covSomme)
#print("matrice cov inversé",covarianceInv)
#print(covarianceInv.dot(covSomme))

##3
"""Frontiere decision"""
tracerFrontiereDecision(moyobs0,moyobs1,pi0,pi1,covarianceInv)

"""Prediction du point"""
point=numpy.array([-10,10])
plt.scatter(-10,10,c='r')
Rd0=RegleDeciAdl(point,covarianceInv,moyobs0,pi0)
Rd1=RegleDeciAdl(point,covarianceInv,moyobs1,pi1)
print("R0 pour ce point: ",Rd0)
print("R1 pour ce point: ",Rd1)

if Rd0>Rd1:
    print("On prédit la classe 0")
else:
    print("on prédit la classe 1")

##4
#LDA
X=WS[['X1','X2']]
yTrain=WS['y']
clf = LinearDiscriminantAnalysis()
clf.fit(X,yTrain)
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd',
store_covariance=False, tol=0.0001)
w = clf.coef_[0]
a = -w[0] / w[1]
yy = a * X - (clf.intercept_[0]) / w[1]
plt.plot(X, yy, label='Frontiere décision ADL sklearn',color='g')
print("Lda de sklearn predit la classe : ",clf.predict([[-10, 10]]))
scoreldaskl=cross_val_score(clf,X,yTrain,cv=LeaveOneOut())
print('Mean Accuracy lda skl',sum(scoreldaskl)/len(scoreldaskl))
#print(scoreldaskl)

#Logistique
clflogis = LogisticRegression(random_state=0,solver='liblinear')
clflogis.fit(X,yTrain)
w = clflogis.coef_[0]
a = -w[0] / w[1]
yy = a * X - (clflogis.intercept_[0]) / w[1]
plt.plot(X, yy,label='Frontiere décision logistique sklearn',color= 'y')
print("logistique de sklearn predit la classe : ",clflogis.predict([[-10, 10]]))
scorelogisskl=cross_val_score(clflogis,X,yTrain,cv=LeaveOneOut())
print('Mean accuracy logistique skl',sum(scorelogisskl)/len(scorelogisskl))
plt.show()
