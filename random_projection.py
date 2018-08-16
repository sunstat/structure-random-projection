import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl


class Simulation(object):
    """docstring for Simulation""" 
    '''
    :param X: input matrix of size m x n 
    :param sim_typ: type of the simulation "len" (length preservation), "col": 
        (column space)
    :param k: the reduced dimension
    :param split: k/split[0] * m/split[1] is the size of the kronecker component 
    '''
    def __init__(self, X, sim_typ, k, split = (10, 10), num_runs = 100, seed = 1):
        self.X = X
        self.sim_typ = sim_typ  
        self.k = k
        self.split = split
        self.num_runs = num_runs
        self.seed = seed 

    def run_sim(self):
        m, n = self.X.shape 
        if self.sim_typ == "len": 
            gaussian_err = [] 
            krao_err = []
            kron_err = [] 
            for i in range(self.num_runs):
                # type of the random matrices: "kron" (Kronecker), "krao" (Khatri-Rao)
                mat_gauss = np.random.normal(0, 1, (self.k,m))/np.sqrt(self.k)
                gaussian_err.append(np.linalg.norm(mat_gauss @ self.X)\
                    /np.linalg.norm(self.X))
                print(self.split[0])
                print(int(m/self.split[0]))
                mat_krao1 = np.random.normal(0, 1, (self.k,int(m/self.split[0]))) 
                print(mat_krao1.shape)
                mat_krao2 = np.random.normal(0, 1, (self.k,self.split[0]))
                print(mat_krao2.shape)
                mat_krao = tl.tenalg.khatri_rao([mat_krao1.T,mat_krao2.T]).T
                print(mat_krao.shape)
                krao_err.append(np.linalg.norm\
                    (mat_krao @ self.X)\
                    /np.sqrt(10)/np.linalg.norm(self.X)) 
                mat_kron1 = np.random.normal(0,1,(self.k,\
                    (m/self.kintron_split))) 
                mat_kron2 = np.random.normal(0,1,(int(m/self.k),\
                 self.kron_split))  
                mat_kron = np.kron(mat_kron1,mat_kron2)
                kron_err.append(np.linalg.norm(mat_kron @ self.X)\
                    /np.sqrt(k)/np.linalg.norm(self.X)) 
            rel_err = [gaussian_err,krao_err,kron_err]
        elif self.sim_typ == "col": 
            gaussian_err = [] 
            krao_err = []
            kron_err = [] 
            for i in range(self.num_runs):
                mat_gauss = np.random.normal(0, 1, (self.k,m))
                mat_krao1 = np.random.normal(0, 1, (self.k,m)) 
                mat_krao2 = np.random.normal(0, 1, (int(m/self.k),m))
                mat_kron1 = np.random.normal(0,1,(self.k,\
                    (m/self.kintron_split))) 
                mat_kron2 = np.random.normal(0,1,\
                    (int(m/self.k), self.kron_split)) 
                mat_krao = tl.tenalg.khatri_rao([mat_krao1.T,mat_krao2.T]).T
                mat_kron = np.kron(mat_kron1,mat_kron2)
                gaussian_err.append(np.linalg.norm(A.T @ A @ mat_gauss-self.X)\
                    /np.linalg.norm(self.X)) 
                krao_err.append(np.linalg.norm(A.T @ A @ mat_krao-self.X)\
                    /np.linalg.norm(self.X))  
                kron_err.append(np.linalg.norm(A.T @ A @ mat_kron-self.X)\
                    /np.linalg.norm(self.X)) 
            rel_err = [gaussian_err,krao_err,kron_err]
        return rel_err

if __name__ == '__main__':
    X1 = np.random.normal(0,1, (10000,1000)) 
    sim_len = Simulation(X1, "len", 10, (5,10)) 
    rel_err = sim_len.run_sim() 

    plt.figure()
    plt.hist(rel_err[gaussian_err])
    plt.hist(rel_err[krao_err])
    plt.hist(rel_err[kron_err])
    plt.show()

    # rank = 10
    X2 =  np.diag(np.repeat(10,10),np.rep(0,615))  
    X2 = X2 + np.random.normal(0,1,(625, 625)) 
    sim_col = Simulation(X2, "len", 25, 25) 
    rel_err = sim_col.run_sim() 

    plt.figure()
    plt.hist(rel_err[gaussian_err])
    plt.hist(rel_err[krao_err])
    plt.hist(rel_err[kron_err])
    plt.show()


