# Tensor Random Projection 

## Introduction 

In this project, we implement the Tensor Random Projection based on the Khatri-Rao product and extend it to matrix sketching. We run the experiment with simiulated data and the MNIST data. 

## Files 
1. util.py: Helper functions for data generation, variance reduction, and others. 
2. simulation.py: Function for creating simulation for both random projection and sketching. 
3. nips_simulation.py: Run experiments for norm preservation and sketching with simulated data on server. 
4. minst.py: Evaluate the result for inner product preservation with MNIST data on server. 
5. nips_simulation.ipynb: Evaluate the norm preservation and sketching for simulated data, and inner product preservation for MNIST data (Replicate Table 1,2, Figure 2-6)
6. dist_simulation.ipynb: Evaluate the distance preservation for both simulated and MNIST data. (Replicate Figure 1)  

## Todo 
1. Applied to Kernel regression with Kronecker product