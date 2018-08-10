#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Mon Apr 10 17:41:24 2017

# DEPENDENCIES:
import numpy as np
import random


# FUNCTION THAT CREATES GAUSSIAN MULTIVARIATE 2D DATASETS, D = features, N = observations
def create_multivariate_Gauss_2D_dataset(mean, sigma, N_observations):
    np.random.seed(444445)              #   Seeding for consistency and reproducibility   seed>100000 prefereably,                 
    MEAN_2D       = np.array([mean,mean])  
    I_2D          = np.matrix(np.eye(2))                                    # Creating m1,aka MEAN1 as an np.array  
    COV_MATRIX_2D = sigma*I_2D              # Could use np.array as well instead of eye, np.array([[1,0,0],[0,1,0],[0,0,1]])
    SAMPLE_SET    = np.random.multivariate_normal(MEAN_2D,COV_MATRIX_2D , N_observations).T    
    #print("MEAN_2D:\n", MEAN_2D); print("\nCOV_MATRIX_2D:\n", COV_MATRIX_2D);    print("\nI_2D:\n", I_2D) ;    print("\nSAMPLE_SET.shape:", SAMPLE_SET.shape) 
    return(SAMPLE_SET)


#%%
# Calling create_multivariate_Gauss_2D_dataset function with desired parameters:
SAMPLE_SET_220 = (create_multivariate_Gauss_2D_dataset(1,0.5,220))
SAMPLE_SET_280 = (create_multivariate_Gauss_2D_dataset(-1,0.75,280))

# Merge into one unified unlabeled dataset:
DATASET = np.concatenate((SAMPLE_SET_220, SAMPLE_SET_280), axis=1)
#%%

# CODE BLOCK FOR PLOTTING UNIFIED DATASET, NO LABELS:

from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
from matplotlib import style
             
style.use('bmh')

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
#plt.rcParams['legend.fontsize'] = 7 


ax.plot(SAMPLE_SET_220 [0,:], SAMPLE_SET_220 [1,:], '.', markersize=8, color='yellow', alpha=0.567, label='SUBSET 220')
ax.plot(SAMPLE_SET_280 [0,:], SAMPLE_SET_280 [1,:], '.', markersize=8, color='teal', alpha=0.567, label='SUBSET 280')


plt.title('DATA POINTS OF THE TWO SUBSETS')
ax.legend(loc='lower left')

plt.show()

## for the maxiters_counter, upon loop completion do: maxiters_counter -=1

#def K_MEANS(X, k, maxiters):#maxiters_counter = maxiters

# Foolproofing iteration through dataset; for i in x_vectors take sample, observation (D,) array AND NOT  feature (N,) array! 

#%%
# Temporarily dumped here:
def K_means(DATASET, k, maxiters):

    X_vectors = [j for j in DATASET.T]            #x_vector.shape = (1,2) ; type(x_vector) = matrix

    
    # Generate a list with k random samples from the DATASET as first centroids:
    random_k_centroids_list = [random.choice(X_vectors) for k in range(0,k)]
    
    #for i in range reps:
    iter_counter = 0
    # Init just once and outside while
    centroids_list = random_k_centroids_list 
    
    SSSE = 0    # Sum of Sum Standard Errors of k clusters
    while iter_counter != maxiters: # or maxiters_counter!=0:    #Converge or stop it!
        
        # A list that denotes the label has an obeservation (D,) of the dataset e.g. [0, 0, 1, 2 , 0 ..]
        # label is the cluster  number, 1,2 etc
        
        y = []
        
        # Initalizing a dict with as many keys as the number of clusters, k 
        clusters_dict = {}    
        
        # Looping through k number of centroids to create k keys of the dictionary:
        # each key is a cluster label    
        for i in range(0,len(centroids_list)):
            
            # Initializing each dictionary key's values, setting it as an empty list
            # Key values will be populated with the samples allocated to the cluster        
            clusters_dict[i] = []
        
        
        # Looping through observations to calculate distance from centroids & allocate to centroid with minimum distance
        for j in X_vectors:
            distances  = [np.linalg.norm(j - c) for c in centroids_list]  # calculating at once distances from all centroids
            label = distances.index(min(distances))                       # the index of the min distance is the label of the cluster
            clusters_dict[label].append(j)                                # append the observation of this loop, to the values of the dict key with the respective label
            y.append(label)                                               # keep a list that holds in which cluster the observations have been allocated; 
            SSSE+= distances[label]     #distortion  measure ,  Bishop 9.1 ?
        for i in range(0,k):
            print("centroid_"+str(i),": ", (centroids_list)[i].T)         # temporary, just for checking the random centroids
        
        centroids_from_mean = []                                          # initialize a list that will hold the new centroids, as calculated by the mean of all observations that made it in the cluster
        for u in range(0,k):
            try:
                centroids_from_mean.append(sum(clusters_dict[u])/len(clusters_dict[u]))  # mean calculation for each key-value pair
            except:
                centroids_from_mean.append(0*clusters_dict[u][0])  #handling zero div error, if no sample has been allocated to a cluster
            
            print("cluster_"+str(u),": ", len(clusters_dict[u]))
            print("cluster_"+str(u),"mean: ", sum(clusters_dict[u])/len(clusters_dict[u]))
        
    
        #centroids_list = centroids_list
        
        
        print("\n\ncentroids_from_mean:", centroids_from_mean)
        print("\n\ncentroids_list:", centroids_list)
        
        print("len(y)", len(y))
        #print(centroids_from_mean)
        
        
        
        # Check for convergence or keep them centroids dancing around: 
        # np.allclose found here: http://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
        # np.allclse official docum page: 
        if np.allclose(np.matrix(centroids_list),np.matrix(centroids_from_mean)) == False: # if this was True it would mean that the centroids only slightly change, tolerance = 0.001, very low
            
            centroids_list = centroids_from_mean                # assign centroids_from_mean to the centroids_list, for the following iter               
            iter_counter += 1                               # substract 1, like a stopwatch, when counter==0 , break bc enough is enough
            print("iteration:" ,iter_counter)
        else:
            from matplotlib import style
             
            style.use('bmh')
            
            colors = [ "teal","coral",  "yellow", "#37BC61", "pink","#CC99CC","teal", 'coral']
            
            for cluster in clusters_dict:
                color = colors[cluster]
                for vector in np.asarray(clusters_dict[cluster]):
                    plt.scatter(vector[0], vector[1], marker="o", color=color, s=2, linewidths=4, alpha=0.876)
            
            for centroid in range(0,len(centroids_from_mean)):
                plt.scatter(centroids_from_mean[centroid][0], centroids_from_mean[centroid][1], marker="x", color="black", s=100, linewidths=4)
                    
            
                
            plt.title("Clustering (K-means) with k = "+str(k)+" and SSSE = "+str(int(SSSE)) )
            plt.savefig("clustering_Kmeans_with_k_eq_"+str(k)+"_cristina_"+str(int(SSSE))+".png", dpi=300)
            return(SSSE, y, centroids_from_mean, plt.show())
            break
                
            
            
    #==============================================================================
    #     #%%  
    #==============================================================================
    # print("\n\ntype(SAMPLE_SET_220)", type(SAMPLE_SET_220))
    # print("\n\nSAMPLE_SET_220.shape:", SAMPLE_SET_220.shape)
    # print("type(clusters_dict[0])",type(clusters_dict[0]))
    # print("\n\ntype(np.asarray(clusters_dict[0]))", type(np.asarray(clusters_dict[0])))
    # print("\n\nnp.asarray(clusters_dict[0])", np.asarray(clusters_dict[0]).shape)
    #==============================================================================

#==============================================================================
# RUN FOR REPS:
# clusterings =  []
# for k in range(1,10):
#     clusterings.append(K_means(DATASET,5, 100))
# # 
#==============================================================================
#==============================================================================

    
#clustering_0 = K_means(DATASET,4, 100)
    
    
#%%
#    CAUTION!!    BUILT-INS KICK IN :
#%%  elbow plot: Distortion  - Number of Clusters
#==============================================================================
# FIND OUT HOW MANY k YOU SHOULD USE FOR THE CLUSTERING, "Elbow Method" 
#==============================================================================
#==============================================================================
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# distortions = []                          # Distortion, the Sum of Squared errors within a cluster.
# for i in range(1, 11):                    # Let's test the performance of clusterings with different k, kE[1,11] 
#     km = KMeans(n_clusters=i,
#     init='k-means++',
#     n_init=10,
#     max_iter=300,
#     random_state=0)
#     km.fit(DATASET.T)     # sklearn wants the data .T if you have them Features x Observations
#     distortions.append(km.inertia_)
# plt.plot(range(1,11), distortions, marker='o', color = "coral")
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.title("Elbow Curve Method: Choose Optimal Number of Centroids", fontsize = 10) # color = "teal")
# 
# plt.show()
#==============================================================================
#==============================================================================
# #%%
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=3, 
#             init='k-means++',
#             n_init=10,
#             max_iter=300,
#             tol=1e-04,
#             random_state=0)
# y_km = km.fit_predict(DATASET.T)
# 
# 
#     
# import numpy as np
# from matplotlib import cm
# from sklearn.metrics import silhouette_samples
# cluster_labels = np.unique(y_km)
# n_clusters = cluster_labels.shape[0]
# silhouette_vals = silhouette_samples(DATASET.T,  y_km,  metric='euclidean')
# 
# y_ax_lower, y_ax_upper = 0, 0
# yticks = []
# 
# 
# colors = [ "teal","coral",  "yellow", "#37BC61", "pink","#CC99CC","teal", 'coral']
# for i, c in enumerate(cluster_labels):
#     c_silhouette_vals = silhouette_vals[y_km == c]
#     c_silhouette_vals.sort()
#     y_ax_upper += len(c_silhouette_vals)
#     color = colors[i]
# 
#     plt.barh(range(y_ax_lower, y_ax_upper), 
#         c_silhouette_vals,
#         height=1.0,
#         edgecolor='none',
#         color=color)
#     
#     yticks.append((y_ax_lower + y_ax_upper) / 2)
#     y_ax_lower += len(c_silhouette_vals)
# 
# silhouette_avg = np.mean(silhouette_vals)
# plt.axvline(silhouette_avg, color="red", linestyle="--")
# 
# plt.yticks(yticks, cluster_labels + 1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.title("Silhouette coefficient plot for k = 3")
# plt.savefig("silh_coeff_k_eq3"+".png", dpi=300)
# plt.show()
#==============================================================================

#%%



#%%
#==============================================================================
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=2, 
#             init='k-means++',
#             n_init=10,
#             max_iter=300,
#             tol=1e-04,
#             random_state=0)
# y_km = km.fit_predict(DATASET.T)
# 
#==============================================================================

#==============================================================================
#     
# import numpy as np
# from matplotlib import cm
# from sklearn.metrics import silhouette_samples
# cluster_labels = np.unique(y_km)
# n_clusters = cluster_labels.shape[0]
# silhouette_vals = silhouette_samples(DATASET.T,  y_km,  metric='euclidean')
# 
# y_ax_lower, y_ax_upper = 0, 0
# yticks = []
# 
# 
# colors = [ "teal","coral",  "yellow", "#37BC61", "pink","#CC99CC","teal", 'coral']
# for i, c in enumerate(cluster_labels):
#     c_silhouette_vals = silhouette_vals[y_km == c]
#     c_silhouette_vals.sort()
#     y_ax_upper += len(c_silhouette_vals)
#     color = colors[i]
# 
#     plt.barh(range(y_ax_lower, y_ax_upper), 
#         c_silhouette_vals,
#         height=1.0,
#         edgecolor='none',
#         color=color)
#     
#     yticks.append((y_ax_lower + y_ax_upper) / 2)
#     y_ax_lower += len(c_silhouette_vals)
# 
# silhouette_avg = np.mean(silhouette_vals)
# plt.axvline(silhouette_avg, color="red", linestyle="--")
# 
# plt.yticks(yticks, cluster_labels + 1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette coefficient')
# plt.title("Silhouette coefficient plot for k = 2")
# plt.savefig("silh_coeff_k_eq2"+".png", dpi=300)
# plt.show()
# 
#==============================================================================
