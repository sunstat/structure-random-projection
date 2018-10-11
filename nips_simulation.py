
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from util import random_matrix_generator, square_tensor_gen
from simulation import Simulation, DimRedux, Simulations
import pickle
import matplotlib


# In[13]:

'''
k = 30
n = 400
dim = 3
X0 = square_tensor_gen(n, r = 5, dim = dim, typ = 'lk', noise_level=0.1)[0]
X = tl.unfold(X0,mode = 0).T
m = X.shape[0]
krao_ms = np.repeat(int(np.sqrt(n)),2*(dim - 1))

ks = np.arange(5,30,5) 
redux1 = DimRedux('g',k, m) 
redux2 = DimRedux('u',k, m)
redux3 = DimRedux('sgn',k, m)
redux4 = DimRedux('sp0',k, m) 
redux5 = DimRedux('sp1',k, m) 
redux6 = DimRedux('g',k, m, krao = True, krao_ms = krao_ms) 
redux7 = DimRedux('g',k, m, krao = True, krao_ms = krao_ms, vr = True, vr_typ = 'geom_median_l2') 
reduxs1 = [redux1, redux2, redux3, redux4, redux5] 
reduxs2 = [redux1, redux6, redux7]

sims_colspace4 = Simulations(X, reduxs2)
sims_colspace4_result = sims_colspace4.run_colspace_varyk(ks)
pickle.dump(sims_colspace4_result, open("data/col_dim"+str(dim)+"_krao_m"+str(m)+".pickle", "wb" ))
# sims_colspace4_result = pickle.load( open("data/col_dim"+str(dim)+"_krao_m"+str(m)+".pickle", "rb" ) )
# sims_colspace4.plot_colspace_varyk(sims_colspace4_result, ks, ["Gaussian","Gaussian Khatri-Rao",     "Gaussian Khatri-Rao Variance Reduced"], str(dim)+"-D "+dim_str(n,dim)+" Tensor Sketching: m = "+str(m),"col_dim"+ str(dim)+"_krao_m"+str(m)) 

'''
# In[ ]:


k = 30
n = 100
dim = 4
X0 = square_tensor_gen(n, r = 5, dim = dim, typ = 'lk', noise_level=0.1)[0]
X = tl.unfold(X0,mode = 0).T
m = X.shape[0]
krao_ms = np.repeat(int(np.sqrt(n)),2*(dim - 1))

ks = np.arange(5,30,5) 
redux1 = DimRedux('g',k, m) 
redux2 = DimRedux('u',k, m)
redux3 = DimRedux('sgn',k, m)
redux4 = DimRedux('sp0',k, m) 
redux5 = DimRedux('sp1',k, m) 
redux6 = DimRedux('g',k, m, krao = True, krao_ms = krao_ms) 
redux7 = DimRedux('g',k, m, krao = True, krao_ms = krao_ms, vr = True, vr_typ = 'geom_median_l2') 
reduxs1 = [redux1, redux2, redux3, redux4, redux5] 
reduxs2 = [redux1, redux6, redux7]



sims_colspace5 = Simulations(X, reduxs2)
sims_colspace5_result = sims_colspace5.run_colspace_varyk(ks)
pickle.dump(sims_colspace5_result, open("data/col_dim"+str(dim)+"_krao_m"+str(m)+".pickle", "wb" ))
# sims_colspace5_result = pickle.load( open("data/col_dim"+str(dim)+"_krao_m"+str(m)+".pickle", "rb" ) )
#sims_colspace5.plot_colspace_varyk(sims_colspace5_result, ks, ["Gaussian","Gaussian Khatri-Rao",     "Gaussian Khatri-Rao Variance Reduced"], str(dim)+"-D "+dim_str(n,dim)+" Tensor Sketching: m = "+str(m), "col_dim"+ str(dim)+"_krao_m"+str(m)) 

