# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy  # pour deepcopy()

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 



def crossval(X, Y, n_iterations, iteration):
   
    born_inf = int(iteration*(len(X)/n_iterations))
    born_sup = int((iteration+1)*(len(X)/n_iterations)-1)

    Xtest = X[born_inf : born_sup+1]
    Ytest = Y[born_inf : born_sup+1]
    
    Xapp = np.concatenate(( X[: born_inf], X[born_sup+1 :]  )) 
    Yapp = np.concatenate(( Y[: born_inf], Y[born_sup+1 :]  )) 
    
    return Xapp, Yapp, Xtest, Ytest     




# code de la validation croisée (version qui respecte la distribution des classes)
def crossval_strat(X, Y, n_iterations, iteration):
        
    classe = np.unique(Y) #valeur de chaque classe
    n = len(classe) #nombre d'éléments de classe
    #print("classe=",classe)
    
    liste_X = [ 0 for i in range(len(classe)) ]
    liste_Y = [ 0 for i in range(len(classe)) ]
    
    for i in range(len(classe)): # repartition des description selon leurs labels
        liste_indice_classe_cour = np.where(Y == classe[i])
        #print(liste_indice_classe_cour)
        liste_X[i] = X[liste_indice_classe_cour[0]]
        liste_Y[i] = Y[liste_indice_classe_cour[0]]
    
    
    Xapp = np.array([])
    Xtest = np.array([])
    Yapp = np.array([])
    Ytest = np.array([])
    
    for i in range(len(classe)):

        born_inf = iteration*(len(liste_X[i])//n_iterations)
        born_sup = (iteration+1)*(len(liste_X[i])//n_iterations) 
        
   
        if i != 0: #Pour ne pas avoir de probleme de dimension
            Xtest= np.concatenate(( Xtest, liste_X[i][(born_inf) : (born_sup)] ))    
            Xapp = np.concatenate(( Xapp, liste_X[i][: (born_inf)], liste_X[i][(born_sup) :]  )) 

        else:
            
            Xtest= liste_X[i][(born_inf) : (born_sup)]
            Xapp = np.concatenate((liste_X[i][: (born_inf)], liste_X[i][(born_sup) :]  )) 
    
        Ytest= np.concatenate(( Ytest, liste_Y[i][(born_inf): (born_sup)] ))
        Yapp = np.concatenate((Yapp, liste_Y[i][: (born_inf)], liste_Y[i][(born_sup) :] )) 
            
    
    #print("Cross:","Xapp:", Xapp[:3],"Yapp:", Yapp[:3],"Xtest:", Xtest[:3], "Ytest:",Ytest[:3])
    #return Xapp.astype(int), Yapp.astype(int), Xtest.astype(int), Ytest.astype(int)
    return Xapp, Yapp, Xtest, Ytest





def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return  np.mean(L), np.std(L)



def validation_croisee(C, DS, nb_iter,verbose=True):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    
    perf = []
    X = DS[0]
    Y = DS[1]    

    for i in range(nb_iter):
        
        Xapp, Yapp, Xtest, Ytest = crossval_strat(X, Y, nb_iter, i)
        C_cpy = copy.deepcopy(C)
        #print("VC:",Xtest)
        #print("VC:",Xapp[:2])
        C_cpy.train(Xapp, Yapp)
        perf.append(C_cpy.accuracy(Xtest, Ytest))
        if verbose :
            print("Itération", i, ": taile base app.= ", len(Yapp), "taille base test= ", len(Xtest), "Taux de bonne classif: ",perf[i])

        taux_moyen, taux_ecart = analyse_perfs(perf)
    return perf, taux_moyen, taux_ecart
        
    







