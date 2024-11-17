# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2024

# Import de packages externes
import numpy as np
import pandas as pd
import graphviz as gv #Pour visualisation arbre classifier

# ---------------------------


# Recopier ici la classe Classifier (complète) du TME 2
# ------------------------ A COMPLETER :
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        # COMPLETER CETTE FONCTION ICI : 
        prediction = np.array([])
        i = 0
        for description in desc_set: # Utilisation possible de numpy.vectorize pour éviter la boucle
            pred = self.predict(description)
            prediction = np.append ( prediction, pred ) #prediction du classifier
            #print("Valeur réelle :", label_set[i], "Prédiction :", pred)
            i += 1
            
        #----> Ancienne version qui considerait seulement les classes +1 / -1   
        #nombre_bonne_reponse = np.sum(prediction == label_set)
        #precision = nombre_bonne_reponse / len(label_set)
        #return precision
        #print("tableau de prediction :", prediction)
        nombre_bonne_reponse = np.where(label_set == prediction,1.,0.)
        

        return nombre_bonne_reponse.mean()

        # ............
        
        # ------------------------------
        

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist = np.linalg.norm(x - self.desc_set, axis=1) #On calcule la distance euclidienne de x à tous les autres points de la base de donnée
        dist_croissante = np.argsort(dist) #on tris les distances selon leurs indices
        kpp = dist_croissante[:self.k] #On recupere l'indice des k plus proches
        self.label_set[kpp] #On recupere les labels des k plus proches voisins
        #print("label:",self.label_set[kpp])
        score = np.sum(self.label_set[kpp] == 1) / len(kpp)
        score = 2*(score-0.5)
        return score
        
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        score = self.score(x)
        if score < 0:
            return -1
        else:
            return 1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set        
        
        
     
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.w = np.random.uniform(-1,1,(1,self.input_dimension))[0]
        self.w = self.w / np.linalg.norm(self.w)
        #print("norme=",np.linalg.norm(self.w))
        #print("len(w)=",len(self.w))
        #print("w=",self.w)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        #self.desct_set = desc_set
        #self.label_set = label_set
        #print(len(desc_set),len(label_set))
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        #print(np.dot(x,self.w.T))
        return np.dot(x,self.w)
    
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        else:
            return 1
        
        
        

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.init = init
        if init :
            self.w = np.zeros(self.input_dimension)
        else:
            self.w = np.random.uniform(0,1,(self.input_dimension))
            self.w = (2 * self.w) -1 
            self.w *= 0.001
            
        self.allw = [self.w.copy()]
        #print("w=",self.w)
        #print("learning_rate=",self.learning_rate)
        print("len(w)=",len(self.w))
            
                    
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """     
        liste_indice = [i for i in range (0,len(desc_set))]#On recupere les indices de desc
        
        liste_indice_melange = np.random.shuffle(liste_indice)
        
        for indice in liste_indice:
        
            score_etoile = self.predict(desc_set[indice])
            
            if score_etoile-label_set[indice] != 0:
                self.w=self.w+(self.learning_rate*desc_set[indice]*label_set[indice])
                self.allw.append(self.w.copy())

                
                     
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        liste_valeurs_norme_differences = []
        
        for nb_iteration in range (nb_max):
            
            w_avant = self.w.copy()
            self.train_step(desc_set, label_set)
            difference = np.linalg.norm( np.abs( w_avant - self.w ) )
            
            liste_valeurs_norme_differences.append(difference)
            
            
            if difference < seuil:
                return liste_valeurs_norme_differences
            
        print("nb_it:",nb_iteration+1 ,"=max itérations")
        return liste_valeurs_norme_differences
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        #print("predict->",x)
        return 1 if self.score(x)>=0 else -1
    
    def get_allw(self):
    
    	return self.allw

    

# Remarque : quand vous transférerez cette classe dans le fichier classifieur.py 
# de votre librairie, il faudra enlever "classif." en préfixe de la classe ClassifierPerceptron:

class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        liste_indice = [i for i in range (0,len(desc_set))]#On recupere les indices de desc
        
        liste_indice_melange = np.random.shuffle(liste_indice)
        
        for indice in liste_indice:
        
            score_etoile = self.predict(desc_set[indice])
            
            if self.score(desc_set[indice])*label_set[indice] < 1:
                self.w=self.w+(self.learning_rate * (label_set[indice] - self.score(desc_set[indice])) * desc_set[indice])
                self.allw.append(self.w.copy())

                
        
# ------------------------ 

# Donner la définition de la classe ClassifierMultiOAA

# Vous pouvez avoir besoin d'utiliser la fonction deepcopy de la librairie standard copy:
import copy 


# ------------------------ A COMPLETER :

class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """
        self.cl_bin = cl_bin
        self.liste_classifiers = []
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        nb_classe = len(np.unique(label_set))
        self.liste_classifiers  = [ copy.deepcopy(self.cl_bin) for i in range(nb_classe) ] #clonage du classifieur de reference autant de fois que de classe differente dans la base d'apprentissage
        for classe in range(nb_classe):
            
            label_set_tmp = np.where(label_set==classe, +1, -1) #np.where(label_set==classe, +1, -1) # la classe courante sera représenté par le label +1, et toute les autres classes par le label -1
            #print("classe=",classe,"label_set_tmp=",label_set_tmp,"unique:",np.unique(label_set_tmp))
            self.liste_classifiers[classe].train(desc_set, label_set_tmp)
                    
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        liste_score = []
        for i in range(len(self.liste_classifiers)):
            liste_score.append(self.liste_classifiers[i].score(x))
            #print("classifier",i,"w=",self.liste_classifiers[i].w)
            
        #liste_score = [ self.liste_classifiers[i].score(x) for i in range(len(self.liste_classifiers)) ]
        #print("liste score:",liste_score)

        return liste_score
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        liste_score = self.score(x)
        #print("maxarg:",liste_score.index(max(liste_score)))
        
        return liste_score.index(max(liste_score)) #retourne l'indice de la valeur maximal




# code de la classe pour le classifieur ADALINE


# ATTENTION: contrairement à la classe ClassifierPerceptron, on n'utilise pas de méthode train_step()
# dans ce classifier, tout se fera dans train()


#TODO: Classe à Compléter

class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max
        
        #self.w = np.zeros(self.input_dimension)
        self.w = np.random.uniform(0,1,(self.input_dimension))
        self.w = (2 * self.w) -1 
        self.w *= 0.001

        if self.history:
            self.allw = [self.w.copy()]
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """  
        liste_valeurs_norme_differences = []
            
        for nb_iteration in range(self.niter_max):
            
            w_avant = self.w.copy()
            seuil = 5e-3 #seuil de convergence

            #Tirage d'un point i
            nb_desc = len(desc_set)
            i = np.random.randint(nb_desc)

            #Calcul du gradient
            x_i = desc_set[i].reshape(1, self.input_dimension) # Pour resize le x_i choisit avec le bon format pour les multiplocation matricielles
            #print("desc_set[i]=",x_i,"shape=", x_i.shape )
            #print("desc_set[i].T=",x_i.T,"shape=", x_i.T.shape)
            
            gradient = x_i.T@((x_i@self.w) - label_set[i])
            self.w = self.w - self.learning_rate*gradient
            if self.history:
                self.allw.append(self.w.copy()) 
            
            #Calcul pour vérifier convergence
            difference = np.linalg.norm( np.abs( w_avant - self.w ) )
            liste_valeurs_norme_differences.append(difference)
             
            if difference < seuil: #Test convergence
                return liste_valeurs_norme_differences 
            
        return liste_valeurs_norme_differences
                                        

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x)>=0 else -1
    

class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE analytique
    """
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension

        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            utilise le fait que l'on peut montrer que l'annulation du gradient 
            correspond au problème suivant(X.T @ X) @ w = X.T @ Y 
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """  
        self.w = np.linalg.solve( (desc_set.T@desc_set), desc_set.T@label_set )
          

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x)>=0 else -1
    
    

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    ########################## COMPLETER ICI
    valeurs, nb_fois = np.unique(Y,return_counts=True)

    indice_max=np.argmax(nb_fois)
    
    return valeurs[indice_max]
    ##########################
        

import math
def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    ########################## COMPLETER ICI 
    somme=0
    if len(P)==1:
            return 0.0
    else:
        for p in P:
            if p!=0:
                somme += p*math.log(p,len(P))
        return -somme
        
    ##########################
    
def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    ########################## COMPLETER ICI 
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    nb_fois=nb_fois/len(Y)
    return shannon(nb_fois)

    ##########################



###################################################
#
# K plus proche voisins MC
#
###################################################



# ------------------------ A COMPLETER :


class ClassifierKNN_MC(ClassifierKNN):
    """ Classe pour représenter un classifieur par K plus proches voisins. Pour un probleme multiclasse
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension, k, nb_label):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        self.nb_label = nb_label

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist = np.linalg.norm(x - self.desc_set, axis=1) # On calcule la distance euclidienne de x à tous les autres points de la base de donnée
        dist_croissante = np.argsort(dist) # On tris les distances selon leurs indices
        kpp = dist_croissante[:self.k] # On recupere l'indice des k plus proches
        self.label_set[kpp] # On recupere les labels des k plus proches voisins

        # Compter les occurrences de chaque label dans les k plus proches voisins
        #print(type(self.label_set[kpp].astype(int))) # Mise en int car float de base et ne marche pas avec np.bincount
        #print(type(self.label_set[kpp][0]))

        counts = np.bincount(self.label_set[kpp].astype(int))
        label = np.argmax(counts) # Label le plus présent
        score = label

        return score
            
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        score = self.score(x)
        return score

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        
        self.desc_set = desc_set
        self.label_set = label_set

        
        





###################################################
#
# Arbre de decision numérique
#
###################################################


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    		
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)





def partitionne(m_desc,m_class,n,s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    return ((m_desc[m_desc[:,n]<=s], m_class[m_desc[:,n]<=s]), \
            (m_desc[m_desc[:,n]>s], m_class[m_desc[:,n]>s]))






class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        #############
        # COMPLETER CETTE PARTIE 
        if self.est_feuille():
            #print("cl:",self.classe)
            #print("att:",self.nom_attribut)
            return self.classe
        
        #print("Les_fils:")
        #print(self.Les_fils)

        #print("exemple[self.attribut]=", exemple[self.attribut])
        #print("self.attribut", self.attribut)
        #print("self.seuil=", self.seuil)
        
        if exemple[self.attribut] <= self.seuil :
            return self.Les_fils["inf"].classifie(exemple)
        
        if exemple[self.attribut] > self.seuil :
            return self.Les_fils["sup"].classifie(exemple)
        
        else:
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])     
            return 0
        #############    

    
    def compte_feuilles(self):
        """ rend le nombre de feuilles sous ce noeud
        """
        #############
        # COMPLETER CETTE PARTIE AUSSI
        tmp = 0
        if self.est_feuille():
            return 1
        else :
            #print("fils:")
            #print( "len(self.Les_fils)=",len(self.Les_fils) )
            for i in self.Les_fils:
                #print(self.Les_fils[i].nom_attribut)
                tmp += self.Les_fils[i].compte_feuilles()
            return tmp
        
        #############
        
     
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g





def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = 0.0  # meilleur gain trouvé (initalisé à 0.0 => aucun gain)
        i_best = -1     # numéro du meilleur attribut (init à -1 (aucun))
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))
        
        #print("----------------------------------------------------")
        #print(X)
        for indice_attr in range (len(X[0])): # Pour chaque colonne de la BD
            #print("\n\ncolonne:",indice_attr,"nom:",LNoms[indice_attr])
            entropie_col = 0
            resultat, _ = discretise(X, Y, indice_attr)
            coupure = resultat[0]
            if coupure != None :# Si plus de deux valeurs differentes pour l'attribut courant
                #print("coupure=", coupure)


                #On considère 2 sorte de label. Les exemples avec label <= resultat et ceux > resultat
                val_inf = np.where(X[:,indice_attr:indice_attr+1] <= coupure)[0] #Permet de prendre la valeur que l'on cherche uniquement dans la colonne courante.
                label_associe = Y[val_inf]
                entropie_col += entropie(label_associe) * (len(val_inf)/len(Y))


                val_sup = np.where(X[:,indice_attr:indice_attr+1] > coupure)[0] #Permet de prendre la valeur que l'on cherche uniquement dans la colonne courante.
                label_associe = Y[val_sup]
                entropie_col += entropie(label_associe) * (len(val_sup)/len(Y))
                #print("\nentropie:",entropie_col)
            else:
                entropie_col = entropie(Y) 


            
            if(entropie_classe > entropie_col):
                
                entropie_classe = entropie_col
                gain_max = entropie_col
                
                i_best = indice_attr
                if resultat != None:
                    Xbest_tuple = partitionne(X, Y, indice_attr, coupure) 
                else:
                    Xbest_tuple = ((X,Y),(None,None))
                Xbest_seuil = coupure
        
        
        ############
        if (i_best != -1): # Un attribut qui amène un gain d'information >0 a été trouvé
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_num(left_data,left_class, epsilon, LNoms), \
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud





class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()
    
    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ---------------------------

