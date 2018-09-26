import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from random_projection import random_matrix_generator
import hdmedians as hd 

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

def geometric_median_matrices(Xs, eps = 1e-5, transpose = True, typ = 'l1'): 
    # By default, compute the column wise geometric median
    if typ == 'l1':
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
    elif typ == 'l2': 
        if not transpose: 
            result = np.zeros(Xs[0].shape)
            num_rows = Xs[0].shape[0]
            for i in range(num_rows):         
                # Stack the ith row of all matrices in X
                mat = np.stack([ Xs[j][i,:] for j in range(len(Xs))],0)
                result[i,:] = hd.geomedian(mat, axis = 0)
            return result
        else: 
            Xs = [X.T for X in Xs]
            result = np.zeros(Xs[0].shape)
            num_rows = Xs[0].shape[0]
            for i in range(num_rows):         
                # Stack the ith row of all matrices in X
                mat = np.stack([ Xs[j][i,:] for j in range(len(Xs))],0)
                result[i,:] = hd.geomedian(mat, axis = 0)
            return result.T

def norm_mean_matrices(Xs):
    X2s = [X**2 for X in Xs] 
    return np.sqrt(sum(X2s)/len(Xs))

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
                reduced_mats.append((random_matrix_generator(self.k,self.m,\
                    self.rm_typ)/np.sqrt(self.k)) @ X )
            else: 
                # Here, assume for simplicity, m = m0**2
                m0 = int(np.sqrt(self.m))
                mat_kraos = []
                mat_kraos.append(random_matrix_generator(self.k,m0,self.rm_typ).T) 
                mat_kraos.append(random_matrix_generator(self.k,m0,self.rm_typ).T)  
                reduced_mats.append((tl.tenalg.khatri_rao(mat_kraos).T) @ X/np.sqrt(self.k)) 
        if self.vr_typ == "mean": 
            reduced_mat = sum(reduced_mats)/np.sqrt(self.vr_num)
        elif self.vr_typ == "geom_median_l1": 
            reduced_mat = geometric_median_matrices(reduced_mats, transpose = \
                True, typ = "l1")*np.sqrt(self.vr_num)
        elif self.vr_typ == "geom_median_l2": 
            reduced_mat = geometric_median_matrices(reduced_mats, transpose = \
                True, typ = "l2")*np.sqrt(self.vr_num)
        elif self.vr_typ == "norm_mean": 
            reduced_mat = norm_mean_matrices(reduced_mats)
        else: 
            print("Please use either mean or geom_median_l1, or geom_median_l2 for vr_typ")
        return(reduced_mat, np.linalg.norm(reduced_mat)**2/np.linalg.norm(X)**2)
    def run_colspace(self, X): 
        reduced_mats = [] 
        for i in range(self.vr_num):  
            if not self.krao:
                mat = random_matrix_generator(self.k,self.m,self.rm_typ) 
                arm, _ = np.linalg.qr( (mat @ X).T)  
                reduced_mats.append( X @ arm @ arm.T)
            else: 
                # Here, assume for simplicity, m = m0**2
                m0 = int(np.sqrt(self.m))
                mat_kraos = []
                mat_kraos.append(random_matrix_generator(self.k,m0,self.rm_typ).T) 
                mat_kraos.append(random_matrix_generator(self.k,m0,self.rm_typ).T)  
                mat_krao = tl.tenalg.khatri_rao(mat_kraos).T 
                arm, _ = np.linalg.qr( (mat_krao @ X).T)  
                reduced_mats.append( X @ arm @ arm.T)
        if self.vr_typ == "mean": 
            reduced_mat = sum(reduced_mats)/self.vr_num
        elif self.vr_typ == "geom_median_l1": 
            shape0 = reduced_mats[0].shape
            reduced_mats_vec = [reduced_mat.reshape(reduced_mat.size) for \
            reduced_mat in reduced_mats]
            reduced_mat = np.asarray(geometric_median(\
                np.stack(reduced_mats_vec, axis = 0)))
            reduced_mat = reduced_mat.reshape(shape0)
        elif self.vr_typ == "geom_median_l2": 
            shape0 = reduced_mats[0].shape
            reduced_mats_vec = [reduced_mat.reshape(reduced_mat.size) for \
            reduced_mat in reduced_mats]
            reduced_mat = np.asarray(hd.geomedian(np.stack(reduced_mats_vec,\
             axis = 0), axis = 0))
            reduced_mat = reduced_mat.reshape(shape0)
        else: 
            print("Please use either mean or geom_median_l1, or geom_median_l2 for vr_typ")
        return(reduced_mat, np.linalg.norm(reduced_mat - X)**2/np.linalg.norm(X)**2)

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
    def run_colspace(self):
        np.random.seed(self.seed)
        rm_typ, k, m, krao, vr, vr_num, vr_typ = self.dim_redux.get_info() 
        m, n = self.X.shape  
        return [self.dim_redux.run_colspace(self.X)[1] for i in range(self.num_runs)]   

MARKER_LIST = ["s", "x", "o","+","*","d","^"]
LINE_LIST = ['-', '--',':','-.','-.','-.','-.','-.']

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
    def plot_varyk(self, sims_result, ks, labels, title, name, fontsize = 18): 
        plt.figure(figsize=(6,5))
        for i in range(len(self.dim_reduxs)):
            plt.errorbar(ks, sims_result[i][0] , 2*sims_result[i][1], label = \
                labels[i], capsize = 5, marker = MARKER_LIST[i], ls = LINE_LIST[i])
        plt.legend(loc = 'best')
        plt.xlabel('Reduced Dimension')
        plt.ylabel('Relative Squared Length after Random Projection')
        plt.title(title)
        plt.axes().title.set_fontsize(fontsize)
        plt.axes().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e')) 
        plt.axes().xaxis.label.set_fontsize(fontsize)
        plt.axes().yaxis.label.set_fontsize(fontsize)
        plt.rc('legend',fontsize = fontsize)
        plt.rc('xtick', labelsize = fontsize) 
        plt.rc('ytick', labelsize = fontsize) 
        plt.tight_layout()
        plt.savefig('plots/'+name+'.pdf')
        plt.show()
    def run_colspace_varyk(self, ks): 
        # Note: the k in dim_redux will be overwritten by the k in ks 
        sims_result = []
        for dim_redux in self.dim_reduxs: 
            errs = []
            stds = [] 
            for k in ks: 
                dim_redux.update_k(k)
                sim = Simulation(self.X,dim_redux, self.num_runs, self.seed) 
                sim_result = sim.run_colspace() 
                errs = np.append(errs, np.mean(sim_result))
                stds = np.append(stds, np.std(sim_result))
            sims_result.append([errs, stds])
        return(sims_result)

    def plot_colspace_varyk(self, sims_result, ks, labels, title, name, fontsize = 18):
        plt.figure(figsize=(6,5))
        for i in range(len(self.dim_reduxs)):
            plt.errorbar(ks, sims_result[i][0] , 2*sims_result[i][1], label = \
                labels[i], capsize = 5, marker = MARKER_LIST[i], ls = LINE_LIST[i])
        plt.legend(loc = 'best')
        plt.xlabel('Reduced Dimension')
        plt.ylabel('Relative Error')
        plt.title(title)
        plt.axes().title.set_fontsize(fontsize)
        plt.axes().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e')) 
        plt.axes().xaxis.label.set_fontsize(fontsize)
        plt.axes().yaxis.label.set_fontsize(fontsize)
        plt.rc('legend',fontsize = fontsize)
        plt.rc('xtick', labelsize = fontsize) 
        plt.rc('ytick', labelsize = fontsize) 
        plt.tight_layout()
        plt.savefig('plots/'+name + '.pdf')
        plt.show()



if __name__ == '__main__':
    ''' 
    k = 30    
    m = 2500
    n = 10
    ks = np.arange(10, 60, 5)
    X = np.random.normal(0,1,(m, n))
    redux1 = DimRedux('g',k, m) 
    redux2 = DimRedux('u',k, m)
    redux3 = DimRedux('sgn',k, m)
    redux4 = DimRedux('sp0',k, m) 
    redux5 = DimRedux('sp1',k, m) 
    redux6 = DimRedux('g',k, m, krao = True) 
    redux7 = DimRedux('g',k, m, krao = True, vr = True, vr_typ = 'mean') 
    reduxs1 = [redux1, redux2, redux3, redux4, redux5] 
    reduxs2 = [redux1, redux6, redux7]

    sims1 = Simulations(X, reduxs1)
    sims1_result = sims1.run_varyk(ks)
    sims1.plot_varyk(sims1_result, ks, ["Gaussian", "Uniform", "Sign", \
        "Sparse Sign","Very Sparse Sign"], 'Random Projection: Benchmark', 'rp_benchmark')
    sims2 = Simulations(X, reduxs2)
    sims2_result = sims2.run_varyk(ks) 
    sims2.plot_varyk(sims2_result, ks, ["Gaussian","Gaussian Khatri-Rao", \
        "Gaussian Khatri-Rao Variance Reduced"], 'Random Projection: Khatri-Rao', 'rp_krao')

    ''' 



    m = 400
    X = np.random.normal(0,1,(m,10)) @ np.random.normal(0,1,(10, m))
    X = X + np.random.normal(0,0.1,(m, m)) 
    k = 30
    ks = np.arange(5,30,5) 
    redux1 = DimRedux('g',k, m) 
    redux2 = DimRedux('u',k, m)
    redux3 = DimRedux('sgn',k, m)
    redux4 = DimRedux('sp0',k, m) 
    redux5 = DimRedux('sp1',k, m) 
    redux6 = DimRedux('g',k, m, krao = True) 
    redux7 = DimRedux('g',k, m, krao = True, vr = True, vr_typ = 'mean') 
    reduxs1 = [redux1, redux2, redux3, redux4, redux5] 
    reduxs2 = [redux1, redux6, redux7]





