import math
import random
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from math import *

""" Calcul de la moyenne"""
def moyenne(x):
    moy=[]
    xmeans=x.mean()
    for index,mean in xmeans.iteritems():
        moy.append([mean])
    return numpy.array(moy)

""" Calcul de la somme des covariances des classes"""
def Sommecovariance(X,Y):
    z=(((len(X)*X)+(len(Y)*Y))/(len(X)+len(Y)))
    return z

"""Calcul de la règle de décision de l'Adl"""
def RegleDeciAdl(point,covarianceInv,moy,pi):
    mulcovmoy=numpy.dot(covarianceInv,moy)
    Rd=(numpy.dot(numpy.transpose(point),mulcovmoy)-(1/2)*numpy.dot(numpy.transpose(moy),mulcovmoy)+numpy.log(pi))
    return Rd

"""Tracer de la frontière de décision de l'ADL"""
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
    plt.plot(z,p,label='Frontiere décision ADL réalisé',color='green')

"""Estimation des parametrès pour appliquer l'ADL"""
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

"""Séparation des observations du dataframe"""
def sepObs(WS):
    obs0=WS.loc[WS['y']==0.0,['X1','X2']]
    obs1=WS.loc[WS['y']==1.0,['X1','X2']]
    return obs0,obs1

"""Fonction de prédiction pour un modèle ADL"""
def predictLDA(point):
    Rd0=RegleDeciAdl(point,covarianceInv,moyobs0,pi0)
    Rd1=RegleDeciAdl(point,covarianceInv,moyobs1,pi1)
    print("S0 de notre ADL pour Xtest: ",Rd0)
    print("S1 de notre ADL pour Xtest: ",Rd1)
    if Rd0>Rd1:
        print("Notre ADL prédit la classe 0")
    else:
        print("Notre ADL prédit la classe 1")

"""Validation croisée Leave One Out"""
def validationCroiseeLOO(obs0,obs1):
    nbObservation=len(obs0)+len(obs1)
    tn=0
    tp=0

    for i in range(0,len(obs0)) :
        obsMoinsUn=obs0.drop(obs0.index[i])
        moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obsMoinsUn,obs1)
        point=numpy.array([obs0.iloc[i,:]]).T
        if RegleDeciAdl(point,covarianceInv,moyobs1,pi1)<RegleDeciAdl(point,covarianceInv,moyobs0,pi0) :
            tn+=1
    for i in range(0,len(obs1)) :
        obsMoinsUn=obs1.drop(obs1.index[i])
        moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obs0,obsMoinsUn)
        point=numpy.array([obs1.iloc[i,:]]).T
        if RegleDeciAdl(point,covarianceInv,moyobs0,pi0)<RegleDeciAdl(point,covarianceInv,moyobs1,pi1) :
            tp+=1

    return ((tp+tn)/nbObservation)*100

def faireBlockTailleSemblable(nbBlock,nbObservation) :
    tailleBlock=floor(nbObservation/nbBlock)
    reste=nbObservation-(tailleBlock*nbBlock)
    tailleParBlock=[]
    
    for i in range(0,nbBlock):
        tailleParBlock.append(tailleBlock)
    
    cpt=0
    while(reste>0):
        tailleParBlock[cpt]+=1
        reste-=1
        cpt=(cpt+1)%nbBlock
    
    return tailleParBlock

def validationCroisee(WS,nbBlock) :
    nbObservation=len(WS.index)
    tailleChaqueBlock=faireBlockTailleSemblable(nbBlock,nbObservation)
    nbDonneesPasseEnTest=0
    tn=0
    tp=0
    fn=0
    fp=0
    
    #Chaque block passe 1 fois en test et (nbBlock-1) fois en apprentissage
    for i in range(0,nbBlock) :
        #Séparation données apprentissage/test
        WSApprentissage=WS.copy()
        WSTest=WS.copy()
        WSApprentissage.drop(WSApprentissage.index.values[nbDonneesPasseEnTest:nbDonneesPasseEnTest+tailleChaqueBlock[i]],inplace=True)
        WSTest.drop(WSApprentissage.index.values,inplace=True)
        
        #Séparation par classe des données d'apprentissage
        obs0=WSApprentissage.loc[WSApprentissage['Revenue']==True,WSApprentissage.columns!= 'Revenue']
        obs1=WSApprentissage.loc[WSApprentissage['Revenue']==False,WSApprentissage.columns!= 'Revenue']
        moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obs0,obs1)
        
        #Séparation par classe des données de test
        ensemblePoint0=WSTest.loc[WSTest['Revenue']==True,WSTest.columns!= 'Revenue']
        ensemblePoint1=WSTest.loc[WSTest['Revenue']==False,WSTest.columns!= 'Revenue']
        
        #Calcule des positifs
        for j in range(0,len(ensemblePoint0)):
            point=numpy.array([ensemblePoint0.iloc[j,:]]).T
            if(RegleDeciAdl(point,covarianceInv,moyobs0,pi0)>RegleDeciAdl(point,covarianceInv,moyobs1,pi1)):
                tp+=1
            else :
                fn+=1
        
        #Calcule des négatifs
        for j in range(0,len(ensemblePoint1)):
            point=numpy.array([ensemblePoint1.iloc[j,:]]).T
            if(RegleDeciAdl(point,covarianceInv,moyobs0,pi0)<RegleDeciAdl(point,covarianceInv,moyobs1,pi1)):
                tn+=1
            else :
                fp+=1
            
        nbDonneesPasseEnTest+=tailleChaqueBlock[i]
    print("Nombre de True positif : ",tp)
    print("Nombre de True negatif : ",tn)
    print("Nombre de False positif : ",fp)
    print("Nombre de False negatif : ",fn)

"""Fonction en charge de la prédiction d'un point pour ADL de SkLearn"""
def sklLDAPredict(X,Y,point):
    """Prédiction du point"""
    clf = LinearDiscriminantAnalysis()
    clf.fit(X,Y)
    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd',
    store_covariance=False, tol=0.0001)
    w = clf.coef_[0]
    a = -w[0] / w[1]
    yy = a * X - (clf.intercept_[0]) / w[1]
    print("Lda de sklearn prédit la classe : ",clf.predict(point),' pour Xtest')
    plt.plot(X, yy, label='Frontiere décision ADL sklearn',color='orange')

"""Fonction en charge de la prédiction d'un point pour la classifcation Logistique de SkLearn"""
def sklLogisticPredict(X,Y,point):
    """Prédiction du point"""
    clflogis = LogisticRegression(random_state=0,solver='liblinear')
    clflogis.fit(X,Y)
    w = clflogis.coef_[0]
    a = -w[0] / w[1]
    yy = a * X - (clflogis.intercept_[0]) / w[1]
    plt.plot(X, yy,label='Frontiere de décision de classification logistique sklearn',color='blue')
    print("Classification logistique de sklearn prédit la classe : ",clflogis.predict(point),' pour Xtest')

"""Fonction en charge de la validation croiséé pour la classifcation ADL de SkLearn"""
def validationCroiséeSklLDA(X,Y):
    clf = LinearDiscriminantAnalysis()
    scoreldaskl=cross_val_score(clf,X,Y,cv=LeaveOneOut())
    print('Accuracy de ADL de Sklearn : ',(sum(scoreldaskl)/len(scoreldaskl))*100,'%')

"""Fonction en charge de la validation croiséé pour la classifcation Logistique de SkLearn"""
def validationCroiséeSklLOG(X,Y):
    clflogis = LogisticRegression(random_state=0,solver='liblinear')
    scorelogisskl=cross_val_score(clflogis,X,Y,cv=LeaveOneOut())
    print('Accuracy de la classification logistique de SkLearn : ',(sum(scorelogisskl)/len(scorelogisskl))*100,'%')

####DATASET1
##1 Chargement des données et Nuage de points
print("#######DATASET1#######")
WS = pd.read_csv('dataset1.csv',',')
plt.figure(1)
plt.title("Nuage de points des données du DataSet1")
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(WS['X1'],WS['X2'],c=WS['y'])
##2 Implémentation de l'ADL
obs0,obs1=sepObs(WS)
moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obs0,obs1)
##3 Frontière de décision de l'ADL et prédiction
"""Frontière de décision"""
tracerFrontiereDecision(moyobs0,moyobs1,pi0,pi1,covarianceInv)
"""Prédiction du point"""
pointpredictarray=numpy.array([-10,10])
plt.scatter(pointpredictarray[0],pointpredictarray[1],c='r',label='point à prédire')
predictLDA(pointpredictarray)
##4 Comparaison  des résultats avec les modèles de Sklearn
X=WS[['X1','X2']]
Y=WS['y']
point=[[-10, 10]]
#LDA SKLEARN
sklLDAPredict(X,Y,point)
#Logistique SKLEARN
sklLogisticPredict(X,Y,point)
###5 Comparaison des erreurs de classification
#LDA CODE
print('Accuracy de ADL codée', validationCroiseeLOO(obs0,obs1),'%')
#LDA SKLEARN
validationCroiséeSklLDA(X,Y)
#Logistique SKLEARN
validationCroiséeSklLOG(X,Y)
plt.draw()

####DATASET2
##1 Chargement des données et Nuage de points
print("#######DATASET2#######")
WS = pd.read_csv('dataset2.csv',',')
plt.figure(2)
plt.title("Nuage de points des données du DataSet2")
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(WS['X1'],WS['X2'],c=WS['y'])
##2 Implémentation de l'ADL
obs0,obs1=sepObs(WS)
moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obs0,obs1)
##3 Frontière de décision de l'ADL et prédiction
"""Frontière de décision"""
tracerFrontiereDecision(moyobs0,moyobs1,pi0,pi1,covarianceInv)
"""Prédiction du point"""
pointpredictarray=numpy.array([-10,10])
plt.scatter(pointpredictarray[0],pointpredictarray[1],c='r',label='point à prédire')
predictLDA(pointpredictarray)
##4 Comparaison  des résultats avec les modèles de Sklearn
X=WS[['X1','X2']]
Y=WS['y']
point=[[-10, 10]]
#LDA SKLEARN
sklLDAPredict(X,Y,point)
#Logistique SKLEARN
sklLogisticPredict(X,Y,point)
###5 Comparaison des erreurs de classification
#LDA CODE
print('Accuracy de ADL codée', validationCroiseeLOO(obs0,obs1),'%')
#LDA SKLEARN
validationCroiséeSklLDA(X,Y)
#Logistique SKLEARN
validationCroiséeSklLOG(X,Y)
plt.draw()

####DATASET3
##1 Chargement des données et Nuage de points
print("#######DATASET3#######")
WS = pd.read_csv('dataset3.csv',',')
plt.figure(3)
plt.title("Nuage de points des données du DataSet3")
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(WS['X1'],WS['X2'],c=WS['y'])
##2 Implémentation de l'ADL
obs0,obs1=sepObs(WS)
moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obs0,obs1)
##3 Frontière de décision de l'ADL et prédiction
"""Frontière de décision"""
tracerFrontiereDecision(moyobs0,moyobs1,pi0,pi1,covarianceInv)
"""Prédiction du point"""
pointpredictarray=numpy.array([-10,10])
plt.scatter(pointpredictarray[0],pointpredictarray[1],c='r',label='point à prédire')
predictLDA(pointpredictarray)
##4 Comparaison  des résultats avec les modèles de Sklearn
X=WS[['X1','X2']]
Y=WS['y']
point=[[-10, 10]]
#LDA SKLEARN
sklLDAPredict(X,Y,point)
#Logistique SKLEARN
sklLogisticPredict(X,Y,point)
###5 Comparaison des erreurs de classification
#LDA CODE
print('Accuracy de ADL codée', validationCroiseeLOO(obs0,obs1),'%')
#LDA SKLEARN
validationCroiséeSklLDA(X,Y)
#Logistique SKLEARN
validationCroiséeSklLOG(X,Y)
plt.draw()

####Analyse d'un jeu de données
print("#######DataSet choisi#######")
WS = pd.read_csv('online_shoppers_intention.csv',',')

#Transformation des variables
onehot = pd.get_dummies(WS['Month'])
WS = WS.drop('Month',axis = 1)
WS = WS.join(onehot)

onehot = pd.get_dummies(WS['VisitorType'])
WS = WS.drop('VisitorType',axis = 1)
WS = WS.join(onehot)
WS['Weekend']=WS['Weekend'].map({True: 1, False: 0})

#Nettoyage du jeu de données
WS=WS.dropna()

obs0=WS.loc[WS['Revenue']==True,WS.columns!= 'Revenue']
obs1=WS.loc[WS['Revenue']==False,WS.columns!= 'Revenue']

#Application de l'ADL sur le jeu de données
moyobs0,moyobs1,pi0,pi1,covarianceInv=estimerParamAdl(obs0,obs1)
#print('Accuracy du Jeu de donnée choisi', validationCroiseeLOO(obs0,obs1))
validationCroisee(WS,5)

plt.show()
