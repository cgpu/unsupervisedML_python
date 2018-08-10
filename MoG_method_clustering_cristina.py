#!/usr/bin/env python3

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

#from scipy.stats import multivariate_normal    #release me , in case of built-in emergency




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
#==============================================================================
# END OF "GENERATE ARTIFICIAL DATASET"
#==============================================================================

#----------------------------------------------------------------------------------------------------------

#==============================================================================
# START OF MIXTURE of GAUSSIANS BLOCK:
#==============================================================================
#%%
    
def generate_k_random_parameters(DATASET,k):
    

    # Store your observation-arrays with shape (D,) as elements of a list:
    X_vectors = [j for j in DATASET.T]            #x_vector.shape = (1,2) ; type(x_vector) = matrix
       
 
    # > Initialize k sets of random pi, mu_vector, SIGMA_matrix.    

    # Should be: len(mu_list_old) = len(pi_list_old) = len(SIGMA_list_old) = k , for k Gaussians  
    mu_list_old      = []               
    pi_list_old      = []                
    SIGMA_list_old   = []

    # Let's populate them with a "for-loop", now that we know the lists should contain k elements:       
    for i in range(0,k):
        mu = random.uniform(0,1) + X_vectors[0]*0      # the X_vector*0, as a no-brainer trick to get the shape right
        mu_list_old.append(mu)

        pi = random.uniform(0,1) 
        pi_list_old.append(pi)

        SIGMA = random.uniform(0,1) * np.matrix(np.eye((DATASET.shape[0])))
        SIGMA_list_old.append(SIGMA)
                
    return(mu_list_old, pi_list_old, SIGMA_list_old)


#==============================================================================
# CALCULATING k PDFs FOR EACH Xn vector: 
#==============================================================================


def MoG_method_clustering(DATASET, k, maxiters):
    
    start = timeit.default_timer()

    mu_old_list, pi_old_list, SIGMA_old_list = generate_k_random_parameters(DATASET,2)

    L_old = - float("inf")
    
    X_vectors = [j for j in DATASET.T]            #x_vector.shape = (1,2) ; type(x_vector) = matrix
    mus    = mu_old_list    
    pis    = pi_old_list    
    SIGMAS = SIGMA_old_list

    #for i in range reps:
    iter_counter = 0
    # Init just once and outside while
    sth_very_little = np.eye((SIGMAS[0].shape[0])) * 0.0000000001  # Add this tiny bit to SIGMA matrix to handle (aka: tackle) possible singular incidents, dunno if it works yet "REMOVEMELATER"
   
    
    
    #SSSE = 0    # Sum of Sum Standard Errors of k clusters
    while iter_counter != maxiters: # or maxiters_counter!=0:    #Converge or stop it!
    
        N_kalligraf_LIST_all_x_vectors = []          #list of lists; each list contains k N_kalligraphs -pdfs- for each x_vector of the  dataset
        for j in range(0,len(X_vectors)):    
            
            N_kalligraf_list_of_one_x_vector = []    #a list of length = k, for one x_vector
            for i in range(0,len(mus)):
                D = DATASET.shape[0]
                A = np.asmatrix(X_vectors[j] - mus[i]).T
                N_kalligraf = (1/(2*math.pi)**D) * (1/ (np.linalg.det(sth_very_little + SIGMAS[i]))**0.5) *  math.exp(-0.5* A.T * (sth_very_little + SIGMAS[i]).I * A)
                N_kalligraf_list_of_one_x_vector.append(N_kalligraf)
        
            N_kalligraf_LIST_all_x_vectors.append(N_kalligraf_list_of_one_x_vector)
            
        ##%%
        
        #==============================================================================
        #  CALCULATING N sets of k gammas FOR EACH Xn vector: 
        #==============================================================================
        
        
        # ONLY ARI8MHTES FIRST:
        gamma_ari8m_LIST = []                            # len(gamma_ari8m_LIST)    = N
        for j in range(0,len(X_vectors)):   
            gamma_ari8m_sublist = []                     # len(gamma_ari8m_sublist) = k 
            for i in range(0,len(mus)):
                gamma_ari8m = pis[i]*N_kalligraf_LIST_all_x_vectors[j][i]
                gamma_ari8m_sublist.append(gamma_ari8m)
            gamma_ari8m_LIST.append(gamma_ari8m_sublist)
        
#==============================================================================
#         print(len(gamma_ari8m_LIST))    
#         print(gamma_ari8m_LIST[0])    
#         print(sum(gamma_ari8m_LIST[2])) 
#         
#==============================================================================
        ##%%
        # EXOUME ARI8MHTES, PAME NA FTIAKSOUME TO KLASMA:
        y = []
        gamma_LIST = []
        
        for j in range(0,len(X_vectors)):   
            gamma_sublist = [gamma_ari8m_LIST[j][i]/sum(gamma_ari8m_LIST[j]) for i in range(0,len(mus))]
            label = gamma_sublist.index(max(gamma_sublist))                       # the index of the max gamma is the label of the gaussian that the x_vector belongs to
            y.append(label)
            gamma_LIST.append(gamma_sublist)
#==============================================================================
#         


#         print("len(gamma_LIST[0]: )", len(gamma_LIST[0]))    
#         print("gamma_LIST[0]: ", gamma_LIST[0])  
#         print("gamma_LIST: ", gamma_LIST)  
#         print("len(y)",len(y))
#         
#==============================================================================
        print("y[:50]",y[:50])

        
        # KAI NAI! TA FTIAKSAME! WOOHOOOO! KAI A8ROIZOUN KAI STI MONADA! Yesssss!!
        #= =============================================================================
        #for j in range(0,len(X_vectors)):
        #   print(sum(gamma_LIST[j])) 
        #==============================================================================
        #..............................................................................
        
        #==============================================================================
        # START OF M-Step
        #==============================================================================
        
        # RE-CALCULATE PARAMETERS pi,mu, SIGMAS:   #
        #(i) first create required accessory data structs: k sum(gammas), k in number Nk, "[..]as the effective number of points assigned to cluster k."

        
        gamma_k_LIST = []
        for i in range(0,len(mus)):
            gamma_k = [x[i] for x in gamma_LIST]
            gamma_k_LIST.append(gamma_k)
#==============================================================================
#         print("gamma_k_LIST:\n\n", gamma_k_LIST)
#         
#         print("len(gamma_k_LIST[0])", len(gamma_k_LIST[0]))
#         print("len(gamma_k_LIST)", len(gamma_k_LIST))
#         
#==============================================================================

        Nk_LIST = []
        for i in range(0,len(mus)):
            Nk  = sum(gamma_k_LIST[i])
            Nk_LIST.append(Nk)
#==============================================================================
#             
#         print("len(Nk_LIST)", len(Nk_LIST)) # should be len(Nk_LIST) = k
#         print("Nk_LIST", Nk_LIST) # should be len(Nk_LIST) = k
#         
#==============================================================================

        # First part of mu_new: 1/Nk
        
        ena_pros_Nk_list = []
        for i in range(0,len(Nk_LIST)):
            ena_pros_Nk  = 1/Nk_LIST[i]
            ena_pros_Nk_list.append(ena_pros_Nk)
#==============================================================================
#         print("ena_pros_Nk_list:\n\n", ena_pros_Nk_list)
#==============================================================================

        pi_new_list      = []                
        for i in range(0,len(Nk_LIST)):
            pi_new = Nk_LIST[i]/len(X_vectors)
            pi_new_list.append(pi_new)


        gamma_k_LIST = []
        for i in range(0,len(mus)):
            gamma_k = [x[i] for x in gamma_LIST]
            gamma_k_LIST.append(gamma_k)


                                       # creting this, to then use sum(gamma_epi_xn_LIST)[i] to create mu_new vector
        gamma_epi_xn_LIST = []         # len(gamma_epi_xn_LIST) = k , len(gamma_epi_xn_LIST)[0] = N
        for i in range(0,len(Nk_LIST)):
            gamma_epi_xn_sublist_for_each_k = [gamma_k_LIST[i][j]*X_vectors[j] for j in range(0,len(X_vectors))]
            gamma_epi_xn_LIST.append(gamma_epi_xn_sublist_for_each_k)
#==============================================================================
#         print("len(gamma_epi_xn_LIST)",len(gamma_epi_xn_LIST))
#         print("len(gamma_epi_xn_LIST[0]):", len(gamma_epi_xn_LIST[0]))
#         
#==============================================================================

        
        # Getting closer to mu_new, this would be 
        
        SUM_gamma_epi_xn_list = []
        for i in range(0,len(gamma_k_LIST)):
            SUM_gamma_epi_xn = sum(gamma_epi_xn_LIST[i])
            SUM_gamma_epi_xn_list.append(SUM_gamma_epi_xn)
##==============================================================================
#         print( "len(SUM_gamma_epi_xn_list): ",len(SUM_gamma_epi_xn_list)) 
#         for i in range(0,len(SUM_gamma_epi_xn_list)):
#             print( "SUM_gamma_epi_xn_list_"+str(i)+":",(SUM_gamma_epi_xn_list[i])) 
#         
##==============================================================================

        ena_pros_Nk_list = []
        for i in range(0,len(Nk_LIST)):
            ena_pros_Nk  = 1/Nk_LIST[i]
            ena_pros_Nk_list.append(ena_pros_Nk)
        
        # FINALLY! mu_new_list calculated
        mu_new_list = []
        for i in range(len(SUM_gamma_epi_xn_list)):
            mu_new = ena_pros_Nk_list[i]*SUM_gamma_epi_xn_list[i]
            mu_new_list.append(mu_new)
            
            
        print((mu_new_list))
            

        
        #PREPING SIGMA_new:
        xn_minus_mu_epi_self_transpose_LIST = []
        
        for i in range(0,len(mu_new_list)):
            
            xn_minus_mu_new_sublist_for_each_k  = [(np.asmatrix(X_vectors[j] - mu_new_list[i]).T) for j in range(0,len(X_vectors))]
            xn_minus_mu_epi_self_transpose_list = [xn_minus_mu_new_sublist_for_each_k[j]*xn_minus_mu_new_sublist_for_each_k[j].T for j in range(0,len(xn_minus_mu_new_sublist_for_each_k))]
            xn_minus_mu_epi_self_transpose_LIST.append(xn_minus_mu_epi_self_transpose_list)
            
#==============================================================================
#         
#         #print("(xn_minus_mu_epi_self_transpose_LIST[0][0].shape)", (xn_minus_mu_epi_self_transpose_LIST[0][0].shape))
#         #print("(xn_minus_mu_epi_self_transpose_LIST[0])", (xn_minus_mu_epi_self_transpose_LIST[0]))
#         print("len(xn_minus_mu_epi_self_transpose_LIST)", len(xn_minus_mu_epi_self_transpose_LIST))
#         print("len(xn_minus_mu_epi_self_transpose_LIST[0])", len(xn_minus_mu_epi_self_transpose_LIST[0]))
#         
#         
#==============================================================================
        


        gamma_k_LIST = []
        for i in range(0,len(mus)):
            gamma_k = [x[i] for x in gamma_LIST]
            gamma_k_LIST.append(gamma_k)
#==============================================================================
#         print("len(gamma_k_LIST)", len(gamma_k_LIST))
#         print("len(gamma_k_LIST[0])", len(gamma_k_LIST[0]))
#         
#==============================================================================

        sum_gamma_epi_x_minus_mu_LIST = []
        gamma_epi_x_minus_mu_LIST     = []
        for i in range(0,len(mus)):
            gamma_epi_x_minus_mu_sublist = [gamma_k_LIST[i][j] * xn_minus_mu_epi_self_transpose_LIST[i][j] for j in range(0,len(X_vectors))]
            gamma_epi_x_minus_mu_LIST.append(gamma_epi_x_minus_mu_sublist)
        
        sum_gamma_epi_x_minus_mu_LIST = [sum(gamma_epi_x_minus_mu_LIST[i]) for i in range(0,len(mus))]
#==============================================================================
#         
#         print("len(gamma_epi_x_minus_mu_LIST)", len(gamma_epi_x_minus_mu_LIST))
#         print("len(gamma_epi_x_minus_mu_LIST[0])", len(gamma_epi_x_minus_mu_LIST[0]))
#         print("len(sum_gamma_epi_x_minus_mu_LIST)", len(sum_gamma_epi_x_minus_mu_LIST))
#==============================================================================
        

        
        SIGMA_new_list = [ena_pros_Nk_list[i] * sum_gamma_epi_x_minus_mu_LIST[i] for i in range(0,len(sum_gamma_epi_x_minus_mu_LIST)) ]
#==============================================================================
#         print("len(SIGMA_new_list)", len(SIGMA_new_list))
#         
#         for i in range(0,len(sum_gamma_epi_x_minus_mu_LIST)) :
#             print("SIGMA_new_list_"+str(i)+":", SIGMA_new_list[i])
#         
#         
#==============================================================================
        
        #==============================================================================
        # END OF CALCULATION SHENANIGANS! NEW PARAMETERS HAVE BEEN ESTIMATED
        # We'VE HOPEFULLY CREATED ALL WE NEED TO CALCULATE L_new
        #==============================================================================
     #   #%%
#==============================================================================
#         print("len(pi_)", len(pi_new_list))                 # should be k
#         print("len(mu_new_list)", len(mu_new_list))         # should be k
#         print("len(SIGMA_new_list)", len(SIGMA_new_list))   # should be k
#         
#==============================================================================

        
        # BUILDING THE L_new little by little:
        
        N_kalligraf_LIST_all_x_vectors_new = []          #list of lists; each list contains k N_kalligraphs for each x_vector of the  dataset
        for j in range(0,len(X_vectors)):    
            
            N_kalligraf_list_of_one_x_vector_new = []    #a list of length = k, for one x_vector
            for i in range(0,len(mus)):
                D = DATASET.shape[0]
                sth_very_little = np.eye((SIGMAS[0].shape[0])) * 0.0000000001  # Add this tiny bit to SIGMA matrix to handle (aka: tackle) possible singular incidents, dunno if it works yet "REMOVEMELATER"
                A = np.asmatrix(X_vectors[j] - mu_new_list[i]).T
                N_kalligraf_new = (1/(2*math.pi)**D) * (1/ (np.linalg.det(SIGMA_new_list[i]))**0.5) *  math.exp(-0.5* A.T * (sth_very_little + SIGMA_new_list[i]).I * A)
                N_kalligraf_list_of_one_x_vector_new.append(N_kalligraf_new)
        
            N_kalligraf_LIST_all_x_vectors_new.append(N_kalligraf_list_of_one_x_vector_new)
        
        
        
        
        # ONLY ARI8MHTES of gamma, then their log, FOR NEW ESTIMATED PARAMETERS:
        gamma_ari8m_LIST_new = []                            # len(gamma_ari8m_LIST)    = N
        for j in range(0,len(X_vectors)):   
            gamma_ari8m_sublist_new = []                     # len(gamma_ari8m_sublist) = k 
            for i in range(0,len(mu_new_list)):
                gamma_ari8m_new = pi_new_list[i]*N_kalligraf_LIST_all_x_vectors_new[j][i]
                gamma_ari8m_sublist_new.append(gamma_ari8m_new)
            gamma_ari8m_LIST_new.append(gamma_ari8m_sublist_new)
        
        sum_gamma_ari8m_new_list = [sum(gamma_ari8m_LIST_new[i]) for i in range(0,len(mu_new_list))]
        log_of_sum_gamma_ari8m_new_list = [np.log(sum_gamma_ari8m_new_list[i]) for i in range(0,len(mu_new_list))]
        #==============================================================================
        # 
        # 
        # print("len(gamma_ari8m_LIST_new)",len(gamma_ari8m_LIST_new))    
        # print("gamma_ari8m_LIST_new[0]",gamma_ari8m_LIST_new[0])    
        # print("len(sum_gamma_ari8m_new_list)", len(sum_gamma_ari8m_new_list))    
        # 
        # 
        #==============================================================================
        
        L_new = sum(log_of_sum_gamma_ari8m_new_list)
            
        
        
        #==============================================================================
        #         # LEt's now calculate the distances with the new parameters:
        #==============================================================================

            

        x_of_cluster_LIST = []
        for i in range(0,len(mu_new_list)):
            x_of_cluster_sublist = [X_vectors[j] for j in range(0,len(X_vectors)) if y[j] == i]
            x_of_cluster_LIST.append(x_of_cluster_sublist)
#==============================================================================
#         print("len(x_of_cluster_LIST)",len(x_of_cluster_LIST))
#         for i in range(0,len(x_of_cluster_LIST)):
#             print("len(x_of_cluster_LIST["+str(i)+"]): "  ,len(x_of_cluster_LIST[i]))
#         #%
#         
#==============================================================================
        
        #
        distances_LIST = []
        for i in range(0,len(mu_new_list)):
            distance_of_cluster_sublist = [np.linalg.norm(X_vectors[j] - mu_new_list[i]) for j in range(0,len(X_vectors)) if y[j] == i]
            distances_LIST.append(distance_of_cluster_sublist)
            
            
        #==============================================================================
        # print("len(distances_LIST)",len(distances_LIST))
        # for i in range(0,len(distances_LIST)):
        #     print("len(distances_LIST["+str(i)+"]): "  ,len(distances_LIST[i]))
    
        SSSE = sum([sum(distances_LIST[i]) for i in range(0,len(distances_LIST))])
    

    
        
        if  abs(L_old - L_new)> 0.000001:
            pis    = pi_new_list
            mus    = mu_new_list
            SIGMAS = SIGMA_new_list
            iter_counter += 1
            L_old = L_new
            print("SSSE", SSSE)
            print("\n\nround(abs(L_old - L_new), 20):" , round(abs(L_old - L_new), 20))
            print("\n\niteration:" , iter_counter)
        else:
            print("\n\nabs(L_old - L_new):" , round(abs(L_old - L_new), 20))
            print("\n\nmu_new_list:\n", mu_new_list)    
            print("\n\nSIGMA_new_list\n",SIGMA_new_list)
            print("\n\ny:\n", y)
            print("SSSE:",SSSE)
            from matplotlib import style
             
            style.use('bmh')
            
            colors = [ "teal","coral",  "yellow", "#37BC61", "pink","#CC99CC","teal", 'coral']
            
            for i in range(0, len(mu_new_list)):
                color = colors[i]
                for vector in np.asarray(x_of_cluster_LIST[i]):
                    plt.scatter(vector[0], vector[1], marker="o", color=color, s=2, linewidths=4, alpha=0.876)
            
            for center in range(0,len(mu_new_list)):
                plt.scatter(mu_new_list[center][0], mu_new_list[center][1], marker="x", color="black", s=100, linewidths=4)
                    
            
                
            plt.title("Clustering (MoG) with k = "+str(k)+" and SSSE = "+str(int(SSSE)) )
            plt.savefig("clustering_MoG_with_k_eq_"+str(k)+"_cristina_"+str(int(SSSE))+".png", dpi=300)
            
            
            end = timeit.default_timer()
            runtime = end - start
            print( "runtime",round(runtime, 6))
            return(mu_new_list, SIGMA_new_list, y, SSSE, runtime)
            break




mu_new_list, SIGMA_new_list, y, SSSE, runtime = ( MoG_method_clustering(DATASET,3, 300))


print("y.count(0)",y.count(0))
#==============================================================================
# 
# trials = []
# for i in range(0,):
#     try:
#         trial = ( MoG_paranoia(DATASET,2, 300))
#         trials.append(trial)
#     except:
#         continue
# 
# 
# 
# 
# #%%
# for i in range(0,len(trials)):
#     print("\n\ntrials["+str(i)+"]:\n",trials[i])
# 
#==============================================================================

#%%
# BUGS APPENDIX:
    
# 1. Needs zero division error handling(s)
# 2. Add a "break kai xanaparto apo tin arxi" an vgei kapoios SIGMA singular, kamia det = 0 (to idio )


# To implementation e to max gamma ftaei, eprepe na valo logikous telestes <, > etc

# error:
#==============================================================================
# [array([-0.86133738, -0.92742245]), array([ nan,  nan])]
# Traceback (most recent call last):
# 
#   File "<ipython-input-296-e885ce65425a>", line 1, in <module>
#     runfile('/home/me/def_MoG_paranoia_cristina_SSSE.py', wdir='/home/me')
# 
#   File "/home/me/anaconda3/lib/python3.5/site-packages/spyder/utils/site/sitecustomize.py", line 866, in runfile
#     execfile(filename, namespace)
# 
#   File "/home/me/anaconda3/lib/python3.5/site-packages/spyder/utils/site/sitecustomize.py", line 102, in execfile
#     exec(compile(f.read(), filename, 'exec'), namespace)
# 
#   File "/home/me/def_MoG_paranoia_cristina_SSSE.py", line 430, in <module>
#     trial = ( MoG_paranoia(DATASET,2, 300))
# 
#   File "/home/me/def_MoG_paranoia_cristina_SSSE.py", line 333, in MoG_paranoia
#     N_kalligraf_new = (1/(2*math.pi)**D) * (1/ (np.linalg.det(SIGMA_new_list[i]))**0.5) *  math.exp(-0.5* A.T * (sth_very_little + SIGMA_new_list[i]).I * A)
# 
#   File "/home/me/anaconda3/lib/python3.5/site-packages/numpy/matrixlib/defmatrix.py", line 972, in getI
#     return asmatrix(func(self))
# 
#   File "/home/me/anaconda3/lib/python3.5/site-packages/scipy/linalg/basic.py", line 658, in inv
#     a1 = _asarray_validated(a, check_finite=check_finite)
# 
#   File "/home/me/anaconda3/lib/python3.5/site-packages/scipy/_lib/_util.py", line 228, in _asarray_validated
#     a = toarray(a)
# 
#   File "/home/me/anaconda3/lib/python3.5/site-packages/numpy/lib/function_base.py", line 1033, in asarray_chkfinite
#     "array must not contain infs or NaNs")
# 
# ValueError: array must not contain infs or NaNs    
#==============================================================================
