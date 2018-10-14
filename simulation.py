import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from util import random_matrix_generator, square_tensor_gen, geometric_median, \
geometric_median_matrices, norm_mean_matrices
import matplotlib.ticker as ticker
import warnings
import hdmedians as hd 

warnings.filterwarnings('ignore')

class DimRedux(object):
    """docstring for DimRedux"""
    def __init__(self, rm_typ, k, m, krao = False, krao_ms = [], vr = False, vr_num = 5, vr_typ = "mean"): 
        self.rm_typ = rm_typ
        self.k = k 
        self.m = m
        self.krao = krao
        self.krao_ms = krao_ms
        self.vr = vr
        self.vr_num = 1 if vr == False else vr_num 
        self.vr_typ = vr_typ
    def get_info(self): 
        return [self.rm_typ, self.k, self.m, self.krao, self.krao_ms, self.vr, self.vr_num, self.vr_typ] 
    def update_k(self, k): 
        self.k = k 
        return 
    def run(self,X):
        reduced_mats = [] 
        for i in range(self.vr_num):  
            if not self.krao:
                reduced_mats.append((random_matrix_generator(self.k,self.m,\
                    self.rm_typ)/np.sqrt(self.k)) @ X )
            elif self.krao_ms != []:  
                assert(np.prod(np.asarray(self.krao_ms)) == self.m), \
                "Please enter valid Khatri-Rao map, so that the product of Khatri-Rao map dimension matches the input dimension of X" 
                mat_kraos = []
                for i in range(len(self.krao_ms)): 
                    mat_kraos.append(random_matrix_generator(self.k, self.krao_ms[i], self.rm_typ).T)
                reduced_mats.append((tl.tenalg.khatri_rao(mat_kraos)).T @ X/ np.sqrt(self.k))
            else: 
                # Here, if the Khatri-Rao random map dimension is not given, assume m = m0**2
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
        elif self.vr_typ == "median": 
            reduced_mat = reduced_mats[0] 
            for index,x in np.ndenumerate(reduced_mat): 
                reduced_mat[index] = np.median([ mat[index] for mat in reduced_mats])
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
            elif self.krao_ms != []:  
                assert(np.prod(np.asarray(self.krao_ms)) == self.m), \
                "Please enter valid Khatri-Rao map, so that the product of Khatri-Rao map dimension matches the input dimension of X" 
                mat_kraos = []
                for i in range(len(self.krao_ms)): 
                    mat_kraos.append(random_matrix_generator(self.k, self.krao_ms[i], self.rm_typ).T)
                mat_krao = tl.tenalg.khatri_rao(mat_kraos).T
                arm, _ = np.linalg.qr( (mat_krao @ X).T)  
                reduced_mats.append( X @ arm @ arm.T)
            else: 
                # Here, if the Khatri-Rao random map dimension is not given, assume m = m0**2
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
        rm_typ, k, m, krao, krao_ms, vr, vr_num, vr_typ = self.dim_redux.get_info() 
        m, n = self.X.shape  
        return [self.dim_redux.run(self.X)[1] for i in range(self.num_runs)] 
    def run_colspace(self):
        np.random.seed(self.seed)
        rm_typ, k, m, krao, kraos, vr, vr_num, vr_typ = self.dim_redux.get_info() 
        m, n = self.X.shape  
        return [self.dim_redux.run_colspace(self.X)[1] for i in range(self.num_runs)]   

MARKER_LIST = ["s", "x", "o","+","*","d","^"]
LINE_LIST = ['-', '--',':','-.','-.','-.','-.','-.']
COLOR_LIST = ['r', 'g','orange', 'violet', 'b','black','y']

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
            CIs = np.zeros((2,len(ks)))
            for i,k in enumerate(ks):
                dim_redux.update_k(k)
                sim = Simulation(self.X,dim_redux, self.num_runs, self.seed) 
                sim_result = sim.run() 
                errs = np.append(errs, np.mean(sim_result))
                stds = np.append(stds, np.std(sim_result)) 
                CIs[0,i] = np.percentile(sim_result, 2.5)
                CIs[1,i] = np.percentile(sim_result, 97.5) 
            sims_result.append([errs, CIs, stds])
        return(sims_result)
    def plot_varyk(self, sims_result, ks, labels, title, name, fontsize = 18): 
        plt.figure(figsize=(6,5))
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)
        for i in range(len(self.dim_reduxs)):
            plt.plot(ks, sims_result[i][0], label = labels[i], \
                marker = MARKER_LIST[i], ls ='-', color = COLOR_LIST[i])
            #plt.plot(ks, sims_result[i][1][0,:], ls = '--', color = COLOR_LIST[i] )
            #plt.plot(ks, sims_result[i][1][1,:], ls = '--', color = COLOR_LIST[i] )
            plt.plot(ks, sims_result[i][0]+2*sims_result[i][2], ls = '--', color = COLOR_LIST[i] )
            plt.plot(ks, sims_result[i][0]-2*sims_result[i][2], ls = '--', color = COLOR_LIST[i] )
        plt.legend(loc = 'best')
        plt.xlabel('Reduced Dimension')
        plt.ylabel('Ratio of Squared Norm after Random Projection')
        plt.title(title)
        plt.tight_layout()
        plt.savefig('plots/'+name+'.pdf')
        plt.show()
    def run_colspace_varyk(self, ks): 
        # Note: the k in dim_redux will be overwritten by the k in ks 
        sims_result = []
        for dim_redux in self.dim_reduxs: 
            errs = []
            CIs = np.zeros((2,len(ks)))
            for i,k in enumerate(ks): 
                dim_redux.update_k(k)
                sim = Simulation(self.X,dim_redux, self.num_runs, self.seed) 
                sim_result = sim.run_colspace() 
                errs = np.append(errs, np.mean(sim_result))
                CIs[0,i] = np.percentile(sim_result, 2.5)
                CIs[1,i] = np.percentile(sim_result, 97.5) 
            sims_result.append([errs, CIs])
        return(sims_result)

    def plot_colspace_varyk(self, sims_result, ks, labels, title, name, fontsize = 18):
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)
        plt.figure(figsize=(6,5))
        for i in range(len(self.dim_reduxs)):
            plt.plot(ks, sims_result[i][0], label = labels[i], \
                marker = MARKER_LIST[i], ls ='-', color = COLOR_LIST[i])
            plt.plot(ks, sims_result[i][1][0,:], ls = '--', color = COLOR_LIST[i] )
            plt.plot(ks, sims_result[i][1][1,:], ls = '--', color = COLOR_LIST[i] )
        plt.legend(loc = 'best')
        plt.xlabel('Reduced Dimension')
        plt.ylabel('Relative Error')
        plt.title(title)
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





