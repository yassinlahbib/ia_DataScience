# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt

# ------------------------ 

def normalisation(df):
	return (df - np.min(df)) / (np.max(df) - np.min(df))

def dist_euclidienne(x1,x2):
	return np.sqrt(np.sum((x2-x1)**2))

def centroide(df):
	return np.mean(df, axis=0)

def dist_centroides(v1,v2):
	return dist_euclidienne(centroide(v1), centroide(v2))

def initialise_CHA(df):
	partition = dict()
	for index in df.index.to_list(): #Donne les index des ligne du dataframe
		partition[index] = [index]
	return partition





########################################
# CHA_centroid
########################################

def fusionne(df, partition, verbose=False):
	""" NB : ceci modifie la partition en entrée"""
	if len(partition) == 1 :
		return

	#print(partition)

	deja_vue = set() #Pour ne pas calculer une distance plusieurs fois
	dist_min = float('inf') #Pour trouvr la distance min 
	cle_fusion = 0,0 #Les deux clés qui vont etre fusionnées 
	cle_max = 0 #La clé du nouveaux cluster (Ne pas oublier d'ajouter +1)


	for exemple in partition :
		#print("ex1:",exemple)
		deja_vue.add(exemple)
		#print("dv:",deja_vue)
		
		if exemple > cle_max: #Permet de rajouter le cluster avec la clée max +1
			cle_max = exemple
			
		for exemple2 in partition:
			#print("ex2:",exemple2)
			if exemple != exemple2 and (exemple2 not in deja_vue): #Pas distance du meme point et avec un point deja vu                                                             
				#print("diff et ex2 pas deja vu")
				dist_cour = dist_centroides(df.iloc[partition[exemple]],df.iloc[partition[exemple2]])

				if dist_cour < dist_min :
					dist_min = dist_cour
					cle_fusion = exemple, exemple2
					#print("\t\t\tla->",cle_fusion)

	#print(cle_fusion, partition)
	
	
	partition[cle_fusion[0]].extend(partition.pop(cle_fusion[1])) #On fusionne les deux clusters
	partition[cle_max+1] = partition.pop(cle_fusion[0]) # On les met dans un nouveaux clusters
	
	
	if verbose:
		print("fusionne: distance minimale trouvée entre [",cle_fusion[0],",",cle_fusion[1],"] =", dist_min)
		print("fusionne:les 2 clusters dont les clés sont [",cle_fusion[0],",",cle_fusion[1],"] sont fusionnées")
		print("fusionne: on crée la nouvelle clé ",cle_max+1,"dans le dictionnaire")
		print("fusionne: les clés de [",cle_fusion[0],",",cle_fusion[1],"] sont supprimées car leurs clusters on été fusionnées")
	
	return partition, cle_fusion[0], cle_fusion[1], dist_min      


def CHA_centroid(df, verbose=False, dendrogramme=False):
	
	depart = initialise_CHA(df) #Partition de base 
	L = []
	n = len(depart)


	for i in range(n-1):
		if verbose :
			part,c1,c2,dist = fusionne(df, depart, verbose = True)
		else :
			part,c1,c2,dist = fusionne(df, depart)
			
		new_key = max(list(part.keys()))
		L.append([c1,c2,dist,+len(part[new_key])])
		
		if verbose :
			print("CHA_centroid: une fusion réalisée de",c1,"avec",c2,"de distance",dist)
			print("CHA_centroid: le nouveau cluster contient ",len(part[new_key]),"exemples")
	
	if verbose:
		print("CHA_centroid: plus de fusion possible, il ne reste qu'un cluster unique.")
	
	if dendrogramme:
		# Paramètre de la fenêtre d'affichage: 
		plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
		plt.title('Dendrogramme (Approche Centroid linkage)', fontsize=25)    
		plt.xlabel("Indice d'exemple", fontsize=25)
		plt.ylabel('Distance', fontsize=25)

		# Construction du dendrogramme pour notre clustering :
		scipy.cluster.hierarchy.dendrogram(
			L, 
			leaf_font_size=24.,  # taille des caractères de l'axe des X
		)

		# Affichage du résultat obtenu:
		plt.grid(True)
		plt.show()
		
	
	return L
	




########################################
# CHA_complete
########################################

def dist_complete(v1,v2):
		
	a = v1.to_numpy()
	b = v2.to_numpy()
	
	if a.ndim == 1: #Si le vecteur possede une seul point
		a = np.array([a])
	if b.ndim == 1:
		b = np.array([b])
	
	
	dist_max = -1
	memo_indice = 0,0
		
	for i in range(len(a)) :
		for j in range(len(b)) :
					   
			dist_cour = dist_euclidienne(a[i],b[j])
			if dist_cour > dist_max:
				dist_max = dist_cour
				memo_indice = j,i
				
	return dist_max, memo_indice


def fusionne_complete_linkage(df, partition, verbose=False):
	""" NB : ceci modifie la partition en entrée"""
	if len(partition) == 1 :
		return
	
	deja_vue = set() #Pour ne pas calculer une distance plusieurs fois
	dist_min = 100 #Pour trouvr la distance min 
	cle_fusion = 0,0 #Les deux clés qui vont etre fusionnées 
	cle_max = 0 #La clé du nouveaux cluster (Ne pas oublier d'ajouter +1)
	
	for exemple in partition :
		deja_vue.add(exemple)
		
		if exemple > cle_max: #Permet de rajouter le cluster avec la clée max +1
			cle_max = exemple
			
		
		for exemple2 in partition:
			
			if exemple != exemple2 and (exemple2 not in deja_vue): #Pas distance du meme point et avec un point deja vu      
				
				dist_cour,_ = dist_complete(df.iloc[partition[exemple]],df.iloc[partition[exemple2]])

				if dist_cour < dist_min :
					dist_min = dist_cour
					cle_fusion = exemple, exemple2
	
	
	partition[cle_fusion[0]].extend(partition.pop(cle_fusion[1])) #On fusionne les deux clusters
	partition[cle_max+1] = partition.pop(cle_fusion[0]) # On les met dans un nouveaux clusters
	
	
	if verbose:
		print("fusionne: distance minimale trouvée entre [",cle_fusion[0],",",cle_fusion[1],"] =", dist_min)
		print("fusionne:les 2 clusters dont les clés sont [",cle_fusion[0],",",cle_fusion[1],"] sont fusionnées")
		print("fusionne: on crée la nouvelle clé ",cle_max+1,"dans le dictionnaire")
		print("fusionne: les clés de [",cle_fusion[0],",",cle_fusion[1],"] sont supprimées car leurs clusters on été fusionnées")
	
	return partition, cle_fusion[0], cle_fusion[1], dist_min


def CHA_complete(df, verbose=False, dendrogramme=False):
	if verbose:
		print(f"Clustering hiérarchique ascendant, version Complete Linkage")
	
	depart = initialise_CHA(df) #Partition de base 
	L = []
	n = len(depart)

	for i in range(n-1):
		if verbose :
			part,c1,c2,dist = fusionne_complete_linkage(df, depart, verbose=True)
		else :
			part,c1,c2,dist = fusionne_complete_linkage(df, depart)
			
		new_key = max(list(part.keys()))
		L.append([c1,c2,dist,+len(part[new_key])])
		
		if verbose :
			print("CHA_complete: une fusion réalisée de",c1,"avec",c2,"de distance",dist)
			print("CHA_complete: le nouveau cluster contient ",len(part[new_key]),"exemples")
	
	if verbose:
		print("CHA_complete: plus de fusion possible, il ne reste qu'un cluster unique.")
	
	if dendrogramme:
		# Paramètre de la fenêtre d'affichage: 
		plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
		plt.title('Dendrogramme (Approche Complete linkage)', fontsize=25)    
		plt.xlabel("Indice d'exemple", fontsize=25)
		plt.ylabel('Distance', fontsize=25)

		# Construction du dendrogramme pour notre clustering :
		scipy.cluster.hierarchy.dendrogram(
			L, 
			leaf_font_size=24.,  # taille des caractères de l'axe des X
		)

		# Affichage du résultat obtenu:
		plt.grid(True)
		plt.show()
		
	return L
	



########################################
# CHA_simple
########################################

def dist_simple(v1,v2):
		
	a = v1.to_numpy()
	b = v2.to_numpy()
	
	if a.ndim == 1: #Si le vecteur possede une seul point
		a = np.array([a])
	if b.ndim == 1:
		b = np.array([b])
	
	dist_min = float('inf')
	memo_indice = 0,0
		
	for i in range(len(a)) :
		for j in range(len(b)) :
					   
			dist_cour = dist_euclidienne(a[i],b[j])
			if dist_cour < dist_min:
				dist_min = dist_cour
				memo_indice = j,i
				
	return dist_min, memo_indice






def fusionne_simple_linkage(df, partition, verbose=False):
	""" NB : ceci modifie la partition en entrée"""
	if len(partition) == 1 :
		return
	
	deja_vue = set() #Pour ne pas calculer une distance plusieurs fois
	dist_min = 100 #Pour trouvr la distance min 
	cle_fusion = 0,0 #Les deux clés qui vont etre fusionnées 
	cle_max = 0 #La clé du nouveaux cluster (Ne pas oublier d'ajouter +1)
	
	for exemple in partition :
		deja_vue.add(exemple)
		
		if exemple > cle_max: #Permet de rajouter le cluster avec la clée max +1
			cle_max = exemple
			
		
		for exemple2 in partition:
			
			if exemple != exemple2 and (exemple2 not in deja_vue): #Pas distance du meme point et avec un point deja vu      
				
				dist_cour,_ = dist_simple(df.iloc[partition[exemple]],df.iloc[partition[exemple2]])

				if dist_cour < dist_min :
					dist_min = dist_cour
					cle_fusion = exemple, exemple2
	
	
	partition[cle_fusion[0]].extend(partition.pop(cle_fusion[1])) #On fusionne les deux clusters
	partition[cle_max+1] = partition.pop(cle_fusion[0]) # On les met dans un nouveaux clusters
	
	
	if verbose:
		print("fusionne: distance minimale trouvée entre [",cle_fusion[0],",",cle_fusion[1],"] =", dist_min)
		#print("fusionne:les 2 clusters dont les clés sont [",cle_fusion[0],",",cle_fusion[1],"] sont fusionnées")
		#print("fusionne: on crée la nouvelle clé ",cle_max+1,"dans le dictionnaire")
		#print("fusionne: les clés de [",cle_fusion[0],",",cle_fusion[1],"] sont supprimées car leurs clusters on été fusionnées")
	
	return partition, cle_fusion[0], cle_fusion[1], dist_min


def CHA_simple(df, verbose=False, dendrogramme=False):
	
	if verbose:
		print(f"Clustering hiérarchique ascendant, version Simple Linkage")
	
	depart = initialise_CHA(df) #Partition de base 
	L = []
	n = len(depart)

	for i in range(n-1):
		if verbose :
			part,c1,c2,dist = fusionne_simple_linkage(df, depart, verbose=True)
		else :
			part,c1,c2,dist = fusionne_simple_linkage(df, depart)
			
		new_key = max(list(part.keys()))
		L.append([c1,c2,dist,+len(part[new_key])])
		
		if verbose :
			print("CHA_simple: une fusion réalisée de",c1,"avec",c2,"de distance",dist)
			print("CHA_simple: le nouveau cluster contient ",len(part[new_key]),"exemples")
	
	if verbose:
		print("CHA_simple: plus de fusion possible, il ne reste qu'un cluster unique.")
	
	if dendrogramme:
		# Paramètre de la fenêtre d'affichage: 
		plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
		plt.title('Dendrogramme (Approche Simple linkage)', fontsize=25)    
		plt.xlabel("Indice d'exemple", fontsize=25)
		plt.ylabel('Distance', fontsize=25)

		# Construction du dendrogramme pour notre clustering :
		scipy.cluster.hierarchy.dendrogram(
			L, 
			leaf_font_size=24.,  # taille des caractères de l'axe des X
		)

		# Affichage du résultat obtenu:
		plt.grid(True)
		plt.show()
		
	return L
	




########################################
# CHA_average
########################################

def dist_average(v1,v2):
		
	a = v1.to_numpy()
	b = v2.to_numpy()
	
	if a.ndim == 1: #Si le vecteur possede une seul point
		a = np.array([a])
	if b.ndim == 1:
		b = np.array([b])
		
		
	Liste_all_dist = []
	for i in range(len(a)) :
		for j in range(len(b)) :             
			Liste_all_dist.append(dist_euclidienne(a[i],b[j]))
				
	return np.mean(Liste_all_dist), len(Liste_all_dist)



def fusionne_average_linkage(df, partition, verbose=False):
	""" NB : ceci modifie la partition en entrée"""
	if len(partition) == 1 :
		return
	
	deja_vue = set() #Pour ne pas calculer une distance plusieurs fois
	dist_min = 100 #Pour trouvr la distance min 
	cle_fusion = 0,0 #Les deux clés qui vont etre fusionnées 
	cle_max = 0 #La clé du nouveaux cluster (Ne pas oublier d'ajouter +1)
	
	for exemple in partition :
		deja_vue.add(exemple)
		
		if exemple > cle_max: #Permet de rajouter le cluster avec la clée max +1
			cle_max = exemple
			
		
		for exemple2 in partition:
			
			if exemple != exemple2 and (exemple2 not in deja_vue): #Pas distance du meme point et avec un point deja vu      
				
				dist_cour,_ = dist_average(df.iloc[partition[exemple]],df.iloc[partition[exemple2]])

				if dist_cour < dist_min :
					dist_min = dist_cour
					cle_fusion = exemple, exemple2
	
	
	partition[cle_fusion[0]].extend(partition.pop(cle_fusion[1])) #On fusionne les deux clusters
	partition[cle_max+1] = partition.pop(cle_fusion[0]) # On les met dans un nouveaux clusters
	
	
	if verbose:
		print("fusionne: distance minimale trouvée entre [",cle_fusion[0],",",cle_fusion[1],"] =", dist_min)
		print("fusionne:les 2 clusters dont les clés sont [",cle_fusion[0],",",cle_fusion[1],"] sont fusionnées")
		print("fusionne: on crée la nouvelle clé ",cle_max+1,"dans le dictionnaire")
		print("fusionne: les clés de [",cle_fusion[0],",",cle_fusion[1],"] sont supprimées car leurs clusters on été fusionnées")
	
	return partition, cle_fusion[0], cle_fusion[1], dist_min


def CHA_average(df, verbose=False, dendrogramme=False):
	
	if verbose:
		print(f"Clustering hiérarchique ascendant, version Average Linkage")
	
	depart = initialise_CHA(df) #Partition de base 
	L = []
	n = len(depart)

	for i in range(n-1):
		if verbose :
			part,c1,c2,dist = fusionne_average_linkage(df, depart, verbose=True)
		else :
			part,c1,c2,dist = fusionne_average_linkage(df, depart)
			
		new_key = max(list(part.keys()))
		L.append([c1,c2,dist,+len(part[new_key])])
		
		if verbose :
			print("CHA_average: une fusion réalisée de",c1,"avec",c2,"de distance",dist)
			print("CHA_average: le nouveau cluster contient ",len(part[new_key]),"exemples")
	
	if verbose:
		print("CHA_average: plus de fusion possible, il ne reste qu'un cluster unique.")
	
	if dendrogramme:
		# Paramètre de la fenêtre d'affichage: 
		plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
		plt.title('Dendrogramme (Approche Average linkage)', fontsize=25)    
		plt.xlabel("Indice d'exemple", fontsize=25)
		plt.ylabel('Distance', fontsize=25)

		# Construction du dendrogramme pour notre clustering :
		scipy.cluster.hierarchy.dendrogram(
			L, 
			leaf_font_size=24.,  # taille des caractères de l'axe des X
		)

		# Affichage du résultat obtenu:
		plt.grid(True)
		plt.show()
		
	return L
	



def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
	""" Params : 
			DF: dataframe des descriptions
			linkage: str Choix du linkage entre les clusters.
				Default=centroid sinon choisir parmis ('simple' ou 'average' ou 'complete' ou 'centroid') 
			
	"""

	if verbose :
		print(f"Clustering Hiérarchique Ascendant : approche {linkage}")
		
	if linkage == "complete":
		return CHA_complete(DF,verbose,dendrogramme)
	if linkage == "average":
		return CHA_average(DF,verbose,dendrogramme)
	if linkage == "simple":
		return CHA_simple(DF,verbose,dendrogramme)
	
		
	return CHA_centroid(DF,verbose,dendrogramme)
	


########################################
# k - Means
########################################




def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    if len(Ens) < 2 :
        #print("Hypothèse: len(Ens) >= 2")
        return 0
    
    return dist_euclidienne(Ens.to_numpy(), centroide(Ens).to_numpy())**2
    
    



from random import sample

def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
        étant donné un entier  𝐾>1  et une base d'apprentissage de  𝑛  exemples rend un np.array composés de  𝐾 exemples tirés aléatoirement dans la base. On fait l'hypothèse que  𝐾<=𝑛 .
    """
    n = len(Ens)
    indice = sample(list(np.arange(n)),K) #Choisi K element dans la liste comprenant les entier de 0 à n-1
    #print(indice)
    
    return Ens.to_numpy()[indice]
    



def plus_proche(Exe,Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
        étant donné un exemple et un array contenant un ensemble de centroides, rend l'indice du centroide dont l'exemple est le plus proche. En cas d'égalité de distance, le centroide de plus petit indice est choisi.
    """

    #dist_min = float('inf')
    #indice_dist_min = -1
    
    #for i in range(len(Centres)):
    #    dist = dist_euclidienne(Exe ,Centres[i])
    #    if  dist < dist_min :
    #        dist_min = dist
    #        indice_dist_min = i
            
    #if indice_dist_min == -1 :
    #    print("Probleme fonction plus_proche: N'a pas trouvé l'indice du centre le plus proche")
    #    return
     
    ############## Mise en commentaire de la partie supérieur car trop lente

    differences = Centres - Exe.to_numpy()
    distances = np.linalg.norm(differences, axis=1)
    
    indice_dist_min = np.argmin(distances)
            
    return indice_dist_min






def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
        étant donné une base d'apprentissage et un ensemble de  𝐾  centroïdes, rend la matrice d'affectation des exemples de la base aux clusters représentés par chaque centroïde.
    """

    dict_affect = dict()
    for i in range(len(Centres)):
        dict_affect[i] = []

    for i in range(len(Base)):
        indice_pp = plus_proche(Base.iloc[i],Centres)
        dict_affect[indice_pp].append(Base.index[i]) #Récupere l'indice dans la base de donnée de l'élément
    
    return dict_affect



def nouveaux_centroides(Base,U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
        étant donné une base d'apprentissage et une matrice d'affectation, rend l'ensemble des nouveaux centroides obtenus
    """
    List_new_centroides = []
    
    for i in range(len(U)) :
        centroide_new = centroide(Base.iloc[U[i]]) #Calcule du centroides avec les exemples du cluster d'indice i
        List_new_centroides.append(list(centroide_new))
    

    return np.array(List_new_centroides)


def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
        étant donné une base d'apprentissage et une matrice d'affectation, rend la valeur de l'inertie globale du partitionnement correspondant
    """
    somme_inertie_cluster = 0
    for cluster in U:
        inerie_cluster_cour = inertie_cluster(Base.iloc[U[cluster]])
        somme_inertie_cluster += inerie_cluster_cour
    return somme_inertie_cluster
    




def kmoyennes(K, Base, epsilon, iter_max, verbose=True):
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : Array pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    
    # Etape 0: Séléctionner K exemples de Base
    
    centres = init_kmeans(K,Base)
    #print(centres)
    historique_affectation = []
    
    for i in range(iter_max):
        
        #plt.scatter(Base['X1'],Base['X2'],color='b')
        #plt.scatter(centres[:,0],centres[:,1],color='r',marker='x')
        #plt.show()
        
        
        # Affecter chaque x au groupe dont il est le plus proche
        dict_affect = affecte_cluster(Base,centres)
        #print(dict_affect)


        # Calcules des nouveaux centroides de chaque clusters
        centres = nouveaux_centroides(Base,dict_affect)
        dict_affect = affecte_cluster(Base,centres)
        
        historique_affectation.append(dict_affect)
        #Critere de convergence
        
        if i > 0:
            if verbose :
                print(f"itération {i} Inertie: {inertie_globale(Base, dict_affect):.4f} Différence : {np.abs(inertie_globale(Base, dict_affect) - inertie_globale_prec):.4f}")


        if i>0 and np.abs(inertie_globale(Base, dict_affect) - inertie_globale_prec) < epsilon:
            break
    
        
        inertie_globale_prec = inertie_globale(Base, dict_affect)
        

    #print(historique_affectation)
    return centres, dict_affect
    
    



import matplotlib.cm as cm

def affiche_resultat(Base,Centres,Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    	Marche pour un DataFrame en 2Dim ['X1','X2']
    """
    couleurs = cm.tab20(np.linspace(0, 1, len(Centres)))
    couleurs_affecte = [ [] for i in range(len(Base)) ]
    
    for cluster in Affect:
        for exemple in Affect[cluster]:
            couleurs_affecte[exemple].append(couleurs[cluster]) #Pour chaque exemple on lui associe la couleur associer au numero de son cluster

    x = Base['X1']
    y = Base['X2']        
    
    for (x,y,c) in zip(x,y,couleurs_affecte):
        plt.scatter(x, y, color=c)
    
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
    plt.show()
    












###########################################################
###########################################################
#
# Evaluation du resultat d'un clustering
#
###########################################################
###########################################################


##########################
# Index de Xie-Beni
##########################



def index_Xie_Beni(centres, dict_affect, Base):
    """ Params:
            numpy.ndarray : centres des clusters
            dict[int,list[int]] : association de chaque exemples a son clusters
            Base : Array pour la base d'apprentissage
        Return:
            float: Co-Inertie [(Compacité d'un cluster) ] divisée par la mesure Inter-Cluster [(Séparabilité des clusters) -> Distance entre les centroïdes] 
            
    """
    intra_cluster = inertie_globale(Base, dict_affect)
    
    dist_min = float('inf')
    
    
    for i in range(len(centres)):
        for j in range(i+1, len(centres)): #On calcule la distance entre tous les centres entre eux
            #print(i,j)
            dist_cour = dist_euclidienne(centres[i], centres[j])
            if dist_cour < dist_min : #On garde la distance la plus petite entre deux centres de cluster
                dist_min = dist_cour
            #print(dist_cour)



        
    inter_cluster = dist_min
    print("dist_min=",inter_cluster)
    
    res = intra_cluster / inter_cluster
    #print(res)
    return res


    
##########################
# Index de Dunn
##########################






def index_Dunn(centres, dict_affect, Base):
    """ 
    Calcule l'indice de Dunn, une mesure de la séparabilité des clusters.

    Params:
        centres (numpy.ndarray) : Les centres des clusters
        dict_affect (dict[int, list[int]]) : L'association de chaque exemple à son cluster
        Base : Array pour la base d'apprentissage

    Return:
        float: Co-Distance divisée par la mesure Inter-Cluster [(Séparabilité des clusters) -> Distance entre les centroïdes] 
    """
    Co_distance = 0
    for i in range(len(centres)):
        desciption_etendue = Base.iloc[dict_affect[i]].to_numpy()[:, np.newaxis, :]
        distances = np.sqrt(np.sum((desciption_etendue - desciption_etendue.transpose(1, 0, 2)) ** 2, axis=2))
        Co_distance += np.max(distances)
		
    dist_min = float('inf')
    
    for i in range(len(centres)):
        for j in range(i+1, len(centres)): #On calcule la distance entre tous les centres entre eux
            #print(i,j)
            dist_cour = dist_euclidienne(centres[i], centres[j])
            if dist_cour < dist_min : #On garde la distance la plus petite entre deux centres de cluster
                dist_min = dist_cour
            #print(dist_cour)
	
    distance_etendue = centres[:, np.newaxis, :]
    distances = np.sqrt(np.sum((distance_etendue - distance_etendue.transpose(1, 0, 2)) ** 2, axis=2))
    dist_min  += np.min(np.triu(distances)) #Utiliastion  de np.triu()pour extraire les valeurs differentes de zero

	
	
        
    inter_cluster = dist_min
    #print("dist_min=",inter_cluster)
    
    res = Co_distance / inter_cluster
    #print("Co_distance=",Co_distance)
    #print(res)
    return res

    




