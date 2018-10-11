
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from util import random_matrix_generator, square_tensor_gen, inner_product
from simulation import Simulation, DimRedux, Simulations
import pickle
import matplotlib


import mnist
import scipy.misc


# In[3]:


k = 10
train_images = mnist.train_images()
train_images.shape


# In[11]:


train_x = train_images.reshape(train_images.shape[0], -1).T
d, n = train_x.shape
d, n 


mnist_gauss = DimRedux("g", k, d)
mnist_gauss_krao= DimRedux("g", k, d, krao = True, krao_ms = [28, 28]) 
mnist_gauss_krao_vr = DimRedux("g", k, d, krao = True, krao_ms = [28, 28], vr = True, vr_typ= "norm_mean")
mnist_sp0 = DimRedux("sp0", k, d)
mnist_sp0_krao= DimRedux("sp0", k, d, krao = True, krao_ms = [28, 28]) 
mnist_sp0_krao_vr = DimRedux("sp0", k, d, krao = True, krao_ms = [28, 28], vr = True, vr_typ= "norm_mean")
mnist_sp1 = DimRedux("sp1", k, d)
mnist_sp1_krao= DimRedux("sp1", k, d, krao = True, krao_ms = [28, 28]) 
mnist_sp1_krao_vr = DimRedux("sp1", k, d, krao = True, krao_ms = [28, 28], vr = True, vr_typ= "norm_mean")


# In[14]:
print("dim redux")


train_x_gauss, _ = mnist_gauss.run(train_x)
train_x_gauss_krao, _ = mnist_gauss_krao.run(train_x)
train_x_gauss_krao_vr, _ = mnist_gauss_krao_vr.run(train_x)
train_x_sp0, _ = mnist_sp0.run(train_x)
train_x_sp0_krao, _ = mnist_sp0_krao.run(train_x)
train_x_sp0_krao_vr, _ = mnist_sp0_krao_vr.run(train_x)
train_x_sp1, _ = mnist_sp1.run(train_x)
train_x_sp1_krao, _ = mnist_sp1_krao.run(train_x)
train_x_sp1_krao_vr, _ = mnist_sp1_krao_vr.run(train_x)

print("inner")

# In[15]:


inner_gauss = inner_product(train_x_gauss)
inner_gauss_krao = inner_product(train_x_gauss_krao)
inner_gauss_krao_vr = inner_product(train_x_gauss_krao_vr)
inner_sp0 = inner_product(train_x_sp0)
inner_sp0_krao = inner_product(train_x_sp0_krao)
inner_sp0_krao_vr = inner_product(train_x_sp0_krao_vr)
inner_sp1 = inner_product(train_x_sp1)
inner_sp1_krao = inner_product(train_x_sp1_krao)
inner_sp1_krao_vr = inner_product(train_x_sp1_krao_vr)


# In[2]:


inners = [inner_gauss ,inner_gauss_krao, inner_gauss_krao_vr,inner_sp0 ,inner_sp0_krao ,inner_sp0_krao_vr ,inner_sp1 ,inner_sp1_krao ,inner_sp1_krao_vr]


# In[16]:


inner0 = inner_product(train_x)


# In[18]:


norm_inner0 = np.linalg.norm(inner0)


# In[ ]:

print("eval")

relerrs = [ np.linalg.norm(inner0 -inner)/norm_inner0 for inner in inners]


# In[ ]:


pickle.dump(relerrs, open("data/mnist_errs.pickle", "wb" ))

