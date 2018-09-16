import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from random_projection import random_matrix_generator

class Simulation(object):
    """docstring for Simulation""" 
    '''
    :param X: input matrix of size m x n 
    :param sim_typ: type of the simulation "len" (length preservation), "col": 
        (column space)
    :param k: the reduced dimension
    :param split: the projection matrix is the Khatri-Rao product of the matrices of size of k * m1, ..., k * mN 
                  where m = m1 * ... * mN
    '''
    def __init__(self, X, k, ms, gen_typ = 'g', num_runs = 100, seed = 1):
        self.X = X
        self.k = k
        self.ms = ms
        self.num_runs = num_runs
        self.seed = seed 
        self.gen_typ = gen_typ

    def run_sim(self):
        np.random.seed(self.seed)
        m, n = self.X.shape 
        reg_err = [] 
        krao_err = []
        for i in range(self.num_runs):
            # type of the random matrices: "kron" (Kronecker), "krao" (Khatri-Rao)
            mat = random_matrix_generator(self.k,m, self.gen_typ)/np.sqrt(self.k)
            reg_err.append(np.linalg.norm(mat @ self.X)\
                /np.linalg.norm(self.X))
            mat_kraos = []
            for j in np.arange(len(self.ms)):
                # Transpose of each factor matrix, since the Khatri-Rao is normally defined for the row product
                mat_kraos.append(random_matrix_generator(self.k, self.ms[j], self.gen_typ).T)
            mat_krao = tl.tenalg.khatri_rao(mat_kraos).T 
            krao_err.append(np.linalg.norm\
                (mat_krao @ self.X)\
                /np.sqrt(self.k)/np.linalg.norm(self.X)) 
        rel_err = [reg_err,krao_err]
        return rel_err 
    def run_sim_multityps(self, gen_typs):
        np.random.seed(self.seed)
        m, n = self.X.shape 
        reg_err = [] 
        krao_err = []
        for i in range(self.num_runs):
            # type of the random matrices: "kron" (Kronecker), "krao" (Khatri-Rao)
            mat = random_matrix_generator(self.k,m, self.gen_typ)/np.sqrt(self.k)
            reg_err.append(np.linalg.norm(mat @ self.X)\
                /np.linalg.norm(self.X))
            mat_kraos = []
            for j in np.arange(len(self.ms)):
                # Transpose of each factor matrix, since the Khatri-Rao is normally defined for the row product
                mat_kraos.append(random_matrix_generator(self.k, self.ms[j], self.gen_typs[j]).T)
            mat_krao = tl.tenalg.khatri_rao(mat_kraos).T 
            krao_err.append(np.linalg.norm\
                (mat_krao @ self.X)\
                /np.sqrt(self.k)/np.linalg.norm(self.X)) 
        rel_err = [reg_err,krao_err]
        return rel_err 


def run_sims(X, ks, ms, gen_typ = 'g', num_runs = 100, seed = 1): 
    [reg_errs, reg_stds, krao_errs, krao_stds] = [], [], [], []
    for k in ks:
        sim = Simulation(X, k, ms, gen_typ = 'g', num_runs = num_runs, seed = seed) 
        reg_err, krao_err = sim.run_sim()
        reg_errs.append(np.mean(reg_err))
        reg_stds.append(np.std(reg_err))
        krao_errs.append(np.mean(krao_err))
        krao_stds.append(np.std(krao_err)) 
    return [np.asarray(reg_errs), np.asarray(reg_stds), np.asarray(krao_errs), np.asarray(krao_stds)] 

def run_sims_multityps(X, ks, ms, gen_typs, num_runs = 100, seed = 1): 
    [reg_errs, reg_stds, krao_errs, krao_stds] = [], [], [], []
    for k in ks:
        sim = Simulation(X, k, ms, gen_typ = 'g', num_runs = num_runs, seed = seed) 
        reg_err, krao_err = sim.run_sim(gen_typs)
        reg_errs.append(np.mean(reg_err))
        reg_stds.append(np.std(reg_err))
        krao_errs.append(np.mean(krao_err))
        krao_stds.append(np.std(krao_err)) 
    return [np.asarray(reg_errs), np.asarray(reg_stds), np.asarray(krao_errs), np.asarray(krao_stds)] 


def find_label(gen_typ):
    if gen_typ == 'g':
        return ['Gaussian', 'Khatri-Rao Gauss'] 
    elif gen_typ == 'u': 
        return ['Uniform', 'Khatri-Rao Uniform']
    elif gen_typ == 'sgn':
        return ['Rademacher', 'Khatri-Rao Rademacher']

def plot_sim(results, ks, gen_typ, title, name): 
    plt.figure(figsize=(6,5))
    label = find_label(gen_typ)
    plt.errorbar(ks, results[0], 2*results[1], label = label[0], fmt = 'o--', capsize = 5)
    plt.errorbar(ks, results[2], 2*results[3], label = label[1],fmt = 'x-.', capsize = 5, color = 'orange')
    plt.legend(loc = 'best')
    plt.xlabel('Reduced Dimension')
    plt.ylabel('Relative Length after Transformation')
    plt.title(title)
    plt.savefig('plots/'+name)
    plt.show()


def run_sim_storage(k, ncol, gen_typ = 'g', num_runs = 100, seed = 1): 
    np.random.seed(seed)
    reg_err = [] 
    krao_err = []
    for i in range(num_runs):
        # type of the random matrices: "kron" (Kronecker), "krao" (Khatri-Rao)
        X = np.random.normal(0,1,(ncol,100)) 
        mat = random_matrix_generator(k,ncol, gen_typ)/np.sqrt(k)
        reg_err.append(np.linalg.norm(mat @ X)\
            /np.linalg.norm(X))
        X = np.random.normal(0,1,(int(ncol**2/4),100)) 
        mat_kraos = []
        mat_kraos.append(random_matrix_generator(k, int(ncol/2), gen_typ).T)
        mat_kraos.append(random_matrix_generator(k, int(ncol/2), gen_typ).T)
        mat_krao = tl.tenalg.khatri_rao(mat_kraos).T 
        krao_err.append(np.linalg.norm\
            (mat_krao @ X)\
            /np.sqrt(k)/np.linalg.norm(X)) 
    rel_err = [reg_err,krao_err]
    return rel_err


def run_sim_multityps_storage(k, ncol, gen_typs, num_runs = 100, seed = 1): 
    np.random.seed(seed)
    reg_err = [] 
    krao_err = []
    for i in range(num_runs):
        # type of the random matrices: "kron" (Kronecker), "krao" (Khatri-Rao)
        X = np.random.normal(0,1,(ncol,100)) 
        mat = random_matrix_generator(k,ncol, 'g')/np.sqrt(k)
        reg_err.append(np.linalg.norm(mat @ X)\
            /np.linalg.norm(X))
        X = np.random.normal(0,1,(int(ncol**2/4),100)) 
        mat_kraos = []
        mat_kraos.append(random_matrix_generator(k, int(ncol**2/4), gen_typ).T)
        mat_krao = tl.tenalg.khatri_rao(mat_kraos).T 
        krao_err.append(np.linalg.norm\
            (mat_krao @ X)\
            /np.sqrt(k)/np.linalg.norm(X)) 
    rel_err = [reg_err,krao_err]
    return rel_err



def run_sims_storage(k, ncols, gen_typ = 'g', num_runs = 100, seed = 1):  
    # ncols is the number of storage columns 
    # For simplicity, Khatri-Rao has size k * (ncols**2/4), Gaussian has size k * ncols 
    [reg_errs, reg_stds, krao_errs, krao_stds] = [], [], [], []
    for ncol in ncols:
        reg_err, krao_err = run_sim_storage(k, ncol, gen_typ, num_runs, seed)
        reg_errs.append(np.mean(reg_err))
        reg_stds.append(np.std(reg_err))
        krao_errs.append(np.mean(krao_err))
        krao_stds.append(np.std(krao_err)) 
    return [np.asarray(reg_errs), np.asarray(reg_stds), np.asarray(krao_errs), np.asarray(krao_stds)] 


def plot_sim_storage(results, ncols, gen_typ, title, name): 
    plt.figure(figsize=(6,5))
    label = find_label(gen_typ)
    plt.errorbar(ncols, results[0], 2*results[1], label = label[0], fmt = 'o--', capsize = 5)
    plt.errorbar(ncols, results[2], 2*results[3], label = label[1], fmt = 'x-.', capsize = 5, color = 'orange')
    plt.legend(loc = 'best')
    plt.xlabel('Number of Stored Columns')
    plt.ylabel('Relative Length after Transformation')
    plt.title(title)
    plt.savefig('plots/'+name)
    plt.show()


def gauss_sgn(k, ncol, ks, ms, gen_typs = 'g', num_runs = 100, seed = 1): 
    [reg_errs, reg_stds, krao_errs, krao_stds] = [], [], [], []
    for k in ks:
        sim = Simulation(X, k, ms, gen_typ = 'g', num_runs = num_runs, seed = seed) 
        reg_err, krao_err = sim.run_sim()
        reg_errs.append(np.mean(reg_err))
        reg_stds.append(np.std(reg_err))
        krao_errs.append(np.mean(krao_err))
        krao_stds.append(np.std(krao_err)) 
    return [np.asarray(reg_errs), np.asarray(reg_stds), np.asarray(krao_errs), np.asarray(krao_stds)] 


if __name__ == '__main__':
    ncols = np.arange(20, 110, 10)  
    k = 30
    results_storage1 = run_sims_storage(k, ncols, 'g')
    plot_sim_storage(results_storage1, ncols, 'g', "Random Projection under Storage Constraint, k = 30", "g_store.pdf")  
    results_storage2 = run_sims_storage(k, ncols, 'u')
    plot_sim_storage(results_storage2, ncols, 'u', "Random Projection under Storage Constraint, k = 30", "u_store.pdf")  
    results_storage3 = run_sims_storage(k, ncols, 'sgn')
    plot_sim_storage(results_storage3, ncols, 'sgn', "Random Projection under Storage Constraint, k = 30", "sgn_store.pdf")  



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