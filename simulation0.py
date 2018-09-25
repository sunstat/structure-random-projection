import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from random_projection import random_matrix_generator

# Implementation of the geometric median from https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points/30299705#30299705
import numpy as np
from scipy.spatial.distance import cdist, euclidean
# a = np.array([[2., 3., 8.], [10., 4., 3.], [58., 3., 4.], [34., 2., 43.]])

def geometric_median(X, eps=1e-5):
    # Compute the geometric median for a list of vectors (rows of X) with the algorithm in http://www.pnas.org/content/pnas/97/4/1423.full.pdf
    y = np.mean(X, 0) 
    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y
        if euclidean(y, y1) < eps:
            return y1
        y = y1
def geometric_median_matrices(Xs, eps = 1e-5, transpose = False): 
    # For all X_i in Xs, we want to apply the geometric median to every row at the same position of the matrix
    if not transpose:
        result = np.zeros(Xs[0].shape)
        num_rows = Xs[0].shape[0]
        for i in range(num_rows):         
            # Stack the ith row of all matrices in X
            mat = np.stack([ Xs[j][i,:] for j in range(len(Xs))],0)
            result[i,:] = geometric_median(mat, eps = eps)
        return result
    else: 
        Xs = [X.T for X in Xs]
        result = np.zeros(Xs[0].shape)
        num_rows = Xs[0].shape[0]
        for i in range(num_rows):         
            # Stack the ith row of all matrices in X
            mat = np.stack([ Xs[j][i,:] for j in range(len(Xs))],0)
            result[i,:] = geometric_median(mat, eps = eps)
        return result.T

class DimRedux(object):
    """docstring for DimRedux"""
    def __init__(self, rm_typ, k, m, krao = False, vr = False, vr_num = 5, vr_typ = "mean"):
        self.rm_typ = rm_typ
        self.k = k 
        self.m = m
        self.krao = krao
        self.vr = vr
        self.vr_num = 1 if vr == False else vr_num 
        self.vr_typ = vr_typ
    def get_info(self): 
        return [self.rm_typ, self.k, self.m, self.krao, self.vr, self.vr_num, self.vr_typ] 
    def update_k(self, k): 
        self.k = k 
        return 
    def run(self,X):
        reduced_mats = [] 
        for i in range(self.vr_num):  
            if not self.krao:
                reduced_mats.append((random_matrix_generator(self.k,self.m,self.rm_typ)/np.sqrt(self.k)) @ X )
            else: 
                # Here, assume for simplicity, m = m0**2
                m0 = int(np.sqrt(self.m))
                mat_kraos = []
                mat_kraos.append(random_matrix_generator(self.k,m0,self.rm_typ).T) 
                mat_kraos.append(random_matrix_generator(self.k,m0,self.rm_typ).T)  
                reduced_mats.append((tl.tenalg.khatri_rao(mat_kraos).T) @ X/np.sqrt(self.k)) 
        if self.vr_typ == "mean": 
            reduced_mat = sum(reduced_mats)/np.sqrt(self.vr_num)
        elif self.vr_typ == "geom_median": 
            reduced_mat = geometric_median_matrices(reduced_mats, transpose = True)*np.sqrt(self.vr_num)
        else: 
            print("Please use either mean or geom_median for vr_typ")
        return(reduced_mat, np.linalg.norm(reduced_mat)**2/np.linalg.norm(X)**2)

class Simulation(object):
    """docstring for Simulation""" 
    '''
    :param X: input matrix of size m x n 
    :param dim_redux: type of the dimension reduction map 
    :param num_runs: Number of simulation runs
    '''
    def __init__(self, X, dim_redux,num_runs = 100, seed = 1):
        self.X = X
        self.dim_redux = dim_redux 
        self.num_runs = num_runs
        self.seed = seed

    def run(self):
        np.random.seed(self.seed)
        rm_typ, k, m, krao, vr, vr_num, vr_typ = self.dim_redux.get_info() 
        m, n = self.X.shape  
        return [self.dim_redux.run(self.X)[1] for i in range(self.num_runs)]  

class Simulations(object):
    """docstring for Simulations"""
    def __init__(self, X, dim_reduxs, num_runs = 100, seed = 1):
        self.X = X
        self.dim_reduxs = dim_reduxs 
        self.num_runs = num_runs
        self.seed = seed
    def run_varyk(self, ks):
        # Note: the k in dim_redux will be overwritten by the k in ks 
        sims_result = []
        for dim_redux in self.dim_reduxs: 
            errs = []
            stds = [] 
            for k in ks: 
                dim_redux.update_k(k)
                sim = Simulation(self.X,dim_redux, self.num_runs, self.seed) 
                sim_result = sim.run() 
                errs = np.append(errs, np.mean(sim_result))
                stds = np.append(stds, np.std(sim_result))
            sims_result.append([errs, stds])
        return(sims_result)
    def plot_varyk(self, sims_results, ks, labels, title = "plot"): 
        plt.figure(figsize=(6,5))
        for i in range(len(self.dim_reduxs)):
            plt.errorbar(ks, sims_result[i][0] , 2*sims_result[i][1], label = labels[i], capsize = 5)
        plt.legend(loc = 'best')
        plt.xlabel('Reduced Dimension')
        plt.ylabel('Relative Squared Length after Transformation')
        plt.title(title)
        # plt.savefig('plots/'+name)
        plt.show()





if __name__ == '__main__':
    k = 30    
    m = 10000
    n = 10
    ks = np.arange(10, 50, 10)
    X = np.random.normal(0,1,(m, n))
    redux1 = DimRedux('g',30, m) 
    redux2 = DimRedux('sp1',30, m) 
    redux3 = DimRedux('g',30, m, krao = True) 
    redux4 = DimRedux('g',30, m, krao = True, vr = True, vr_typ = 'mean') 
    reduxs = [redux1, redux2, redux3, redux4]

    sims = Simulations(X, reduxs)
    sims_result = sims.run_varyk(ks)

    sims.plot_varyk(sims_result, ks, ["Gaussian", "Sparse", "Gaussian Khatri-Rao", "Gaussian Khatri-Rao Variance Reduced"])
    # result = run_sims_multityps_storage(k, ncols,gen_typs)
    # plot_sim_storage_multityp(result, ncols, gen_typs, "multiplots", "multiplots.pdf")
'''
    results_storage1 = run_sims_storage(k, ncols, 'g')
    plot_sim_storage(results_storage1, ncols, 'g', "Random Projection under Storage Constraint, k = 30", "g_store.pdf")  
    results_storage2 = run_sims_storage(k, ncols, 'u')
    plot_sim_storage(results_storage2, ncols, 'u', "Random Projection under Storage Constraint, k = 30", "u_store.pdf")  
    results_storage3 = run_sims_storage(k, ncols, 'sgn')
    plot_sim_storage(results_storage3, ncols, 'sgn', "Random Projection under Storage Constraint, k = 30", "sgn_store.pdf")  

''' 

'''
    # Gaussian case with limited storage
    X1 = np.random.normal(0,1, (2500,1000)) 
    ms = [50, 50] 
    gen_typs = ['g', 'sgn']
    ks = np.arange(10, 110, 10) 
    result1 = run_sims(X1, ks, ms, ['g', 'sgn']) 
    plot_sim(result1, ks, 'g', title = 'Gauss + Sign, $m = 20*20$', name = 'g_sgn_2500.pdf') 

'''    

'''
    # Gaussian Case: 
    X1 = np.random.normal(0,1, (400,1000)) 
    ms = [20, 20] 
    ks = np.arange(10, 110, 10) 
    result1 = run_sims(X1, ks, ms) 
    plot_sim(result1, ks, 'g', title = '$m = 20*20$', name = 'g_400.pdf') 
     

    
    X2 = np.random.normal(0,1, (2500, 1000)) 
    ms = [50, 50] 
    ks = np.arange(10, 110, 10) 
    result2 = run_sims(X2, ks, ms) 
    plot_sim(result2, ks, 'g', title = '$m = 50*50$', name = 'g_2500.pdf')
     

    # Uniform  
    X3 = np.random.normal(0,1, (400,1000)) 
    ms = [20,20] 
    ks = np.arange(10, 110, 10) 
    result3 = run_sims(X3, ks, ms, 'u') 
    plot_sim(result3, ks, 'u', title = '$m = 20*20$', name = 'u_400.pdf')

    X4 = np.random.normal(0,1, (2500,1000)) 
    ms = [50,50] 
    ks = np.arange(10, 110, 10) 
    result4 = run_sims(X4, ks, ms, 'u') 
    plot_sim(result3, ks, 'u', title = '$m = 50*50$', name = 'u_2500.pdf')

    # Sign  
    X5 = np.random.normal(0,1, (400,1000)) 
    ms = [20,20] 
    ks = np.arange(10, 110, 10) 
    result5 = run_sims(X5, ks, ms, 'sng') 
    plot_sim(result3, ks, 'sgn', title = '$m = 20*20$', name = 'sgn_400.pdf')

    X6 = np.random.normal(0,1, (2500,1000)) 
    ms = [50,50] 
    ks = np.arange(10, 110, 10) 
    result6 = run_sims(X6, ks, ms, 'sng') 
    plot_sim(result3, ks, 'sgn', title = '$m = 50*50$', name = 'sgn_2500.pdf')


'''

'''
    X1 = np.random.normal(0,1, (10000,1000)) 
    sim_len = Simulation(X1, 20, [100,100], 'sgn') 
    rel_err = sim_len.run_sim() 

    plt.figure()
    plt.hist(rel_err[0])
    print("reg: mean" + str(np.mean(rel_err[0]))+"variance:" +str(np.var(rel_err[0])))
    plt.hist(rel_err[1])
    print("krao: mean" + str(np.mean(rel_err[1]))+"variance:" +str(np.var(rel_err[1])))
    plt.show()
 ''' 