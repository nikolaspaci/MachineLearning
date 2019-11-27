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

"""ADL"""
def estimerParamAdl(obs0,obs1):
    moyobs0=moyenne(obs0)
    moyobs1=moyenne(obs1)
    pi0=len(obs0)/(len(WS))
    pi1=len(obs1)/(len(WS))

    covObs0=numpy.cov(obs0.T)
    covObs1=numpy.cov(obs1.T)

    covSomme=Sommecovariance(covObs0,covObs1)
    covarianceInv=numpy.linalg.pinv(covSomme)
    return moyobs0,moyobs1,pi0,pi1,covarianceInv

def sepObs(WS):
    obs0=WS.loc[WS['y']==0.0,['X1','X2']]
    obs1=WS.loc[WS['y']==1.0,['X1','X2']]
    return obs0,obs1

def predictLDA(point):
    plt.scatter(point[0],point[1],c='r')
    Rd0=RegleDeciAdl(point,covarianceInv,moyobs0,pi0)
    Rd1=RegleDeciAdl(point,covarianceInv,moyobs1,pi1)
    print("R0 pour ce point: ",Rd0)
    print("R1 pour ce point: ",Rd1)
    if Rd0>Rd1:
        print("On prédit la classe 0")
    else:
        print("on prédit la classe 1")

"""Validation croisée LOO"""
def validationCroisé(obs0,obs1):
    nbObservation=len(obs0)+len(obs1)
    tn=0
    tp=0

    for i in range(0,len(obs0)) :
        obsMoinsUn=obs0.drop(obs0.index[i])
        moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obsMoinsUn,obs1)
        point=numpy.array([obs0.iat[i,0],obs0.iat[i,1]])
        if RegleDeciAdl(point,covarianceInv,moyobs1,pi1)<RegleDeciAdl(point,covarianceInv,moyobs0,pi0) :
            tn+=1
    for i in range(0,len(obs1)) :
        obsMoinsUn=obs1.drop(obs1.index[i])
        moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obs0,obsMoinsUn)
        point=numpy.array([obs1.iat[i,0],obs1.iat[i,1]])
        if RegleDeciAdl(point,covarianceInv,moyobs0,pi0)<RegleDeciAdl(point,covarianceInv,moyobs1,pi1) :
            tp+=1

    return (tp+tn)/nbObservation

def sklLDAPredict(WS,point):
    X=WS[['X1','X2']]
    Y=WS['y']

    """prediction"""
    clf = LinearDiscriminantAnalysis()
    clf.fit(X,Y)
    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd',
    store_covariance=False, tol=0.0001)
    w = clf.coef_[0]
    a = -w[0] / w[1]
    yy = a * X - (clf.intercept_[0]) / w[1]
    plt.plot(X, yy, label='Frontiere décision ADL sklearn',color='g')
    print("Lda de sklearn predit la classe : ",clf.predict(point))

    """Validation croisée"""
    scoreldaskl=cross_val_score(clf,X,Y,cv=LeaveOneOut())
    print('Mean Accuracy lda skl',sum(scoreldaskl)/len(scoreldaskl))
    #print(scoreldaskl)

def sklLogisticPredict(WS,point):
    X=WS[['X1','X2']]
    Y=WS['y']
    clflogis = LogisticRegression(random_state=0,solver='liblinear')
    clflogis.fit(X,Y)
    w = clflogis.coef_[0]
    a = -w[0] / w[1]
    yy = a * X - (clflogis.intercept_[0]) / w[1]
    plt.plot(X, yy,label='Frontiere décision logistique sklearn',color= 'y')
    print("logistique de sklearn predit la classe : ",clflogis.predict(point))
    scorelogisskl=cross_val_score(clflogis,X,Y,cv=LeaveOneOut())
    print('Mean accuracy logistique skl',sum(scorelogisskl)/len(scorelogisskl))

##1 Chargement des données et Nuage de points
WS = pd.read_csv('dataset3''.csv',',')

plt.title("Nuage de points des données extraites")
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(WS['X1'],WS['X2'],c=WS['y'])

##2 ADL
obs0,obs1=sepObs(WS)
moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obs0,obs1)

##3
"""Frontiere decision"""
tracerFrontiereDecision(moyobs0,moyobs1,pi0,pi1,covarianceInv)

"""Prediction du point"""
point=numpy.array([-10,10])
predictLDA(point)

##4
#LDA CODE
print('Mean Accuracy lda codé', validationCroisé(obs0,obs1))

#LDA SKLEARN
point=[[-10, 10]]
sklLDAPredict(WS,point)
#Logistique SKLEARN
sklLogisticPredict(WS,point)

plt.show()
