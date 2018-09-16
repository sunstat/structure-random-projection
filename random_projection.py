import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl

def random_matrix_generator(m, n, typ = 'g', std = 1, sparse_factor = 0.5):
    types = set(['g', 'u', 'sp', 'sp0', 'sp1', 'sgn'])
    assert typ in types, "please aset your type of random variable correctly"
    if typ == 'g':
        return np.random.normal(0,1, size = (m,n))*std
    elif typ == 'u':
        return np.random.uniform(low = -1, high = 1, size = (m,n))*np.sqrt(3)*std
    elif typ == 'sgn':
        return np.random.choice([-1,1], size = (m,n), p = [1/2, 1/2])*std 
    elif typ == 'sp0':
        # Result from Achlioptas
        return np.random.choice([-1,0,1], size = (m,n), p = [1/6, 2/3,1/6])*np.sqrt(3)*std 
    elif typ == 'sp1': 
        # Result from Ping Li 
        return np.random.binomial(n = 1,p = np.sqrt(1/np.sqrt(n)),size = (m,n))*\
        np.random.choice([-1,1], size = (m,n))*std*np.sqrt(np.sqrt(n))
    elif typ == 'sp':
        return np.random.binomial(n = 1,p = sparse_factor,size = (m,n))*\
        np.random.choice([-1,1], size = (m,n))*std*np.sqrt(1/sparse_factor)

class Simulation(object):
    """docstring for Simulation""" 
    '''
    :param X: input matrix of size m x n 
    :param sim_typ: type of the simulation "len" (length preservation), "col": 
        (column space)
    :param k: the reduced dimension
    :param split: k/split[0] * m/split[1] is the size of the kronecker component 
    '''
    def __init__(self, X, sim_typ, k, split = (10, 10), gen_typ = 'g', num_runs = 100, seed = 1):
        self.X = X
        self.sim_typ = sim_typ  
        self.k = k
        self.split = split
        self.num_runs = num_runs
        self.seed = seed 
        self.gen_typ = gen_typ

    def run_sim(self):
        np.random.seed(self.seed)
        m, n = self.X.shape 
        if self.sim_typ == "len": 
            gaussian_err = [] 
            krao_err = []
            kron_err = [] 
            for i in range(self.num_runs):
                # type of the random matrices: "kron" (Kronecker), "krao" (Khatri-Rao)
                mat = random_matrix_generator(self.k,m, self.gen_typ)/np.sqrt(self.k)
                gaussian_err.append(np.linalg.norm(mat @ self.X)\
                    /np.linalg.norm(self.X))
                mat_krao1 = random_matrix_generator(self.k,int(m/self.split[1]), self.gen_typ)
                mat_krao2 = random_matrix_generator(self.k,self.split[1], self.gen_typ)
                mat_krao = tl.tenalg.khatri_rao([mat_krao1.T,mat_krao2.T]).T
                krao_err.append(np.linalg.norm\
                    (mat_krao @ self.X)\
                    /np.sqrt(self.k)/np.linalg.norm(self.X)) 
                mat_kron1 = random_matrix_generator(int(self.k/self.split[0]),\
                    int(m/self.split[1]), self.gen_typ)
                mat_kron2 = random_matrix_generator(self.split[0],\
                 self.split[1], self.gen_typ)
                mat_kron = np.kron(mat_kron1,mat_kron2)
                kron_err.append(np.linalg.norm(mat_kron @ self.X)\
                    /np.sqrt(self.k)/np.linalg.norm(self.X)) 
            rel_err = [gaussian_err,krao_err,kron_err]
        elif self.sim_typ == "col": 
            gaussian_err = [] 
            krao_err = []
            kron_err = [] 
            for i in range(self.num_runs):
                print(i)
                mat = random_matrix_generator(self.k,m)
                mat_krao1 = random_matrix_generator(self.k,int(m/self.split[1]), self.gen_typ) 
                mat_krao2 = random_matrix_generator(self.k,int(self.split[1]), self.gen_typ)
                mat_kron1 = random_matrix_generator(self.k/self.split[0],\
                    (m/self.split[1]), self.gen_typ)
                mat_kron2 = random_matrix_generator \
                    (self.split[0], self.split[1])
                mat_krao = tl.tenalg.khatri_rao([mat_krao1.T,mat_krao2.T]).T
                mat_kron = np.kron(mat_kron1,mat_kron2)
                gaussian_err.append(np.linalg.norm(A.T @ A @ mat-self.X)\
                    /np.linalg.norm(self.X)) 
                krao_err.append(np.linalg.norm(A.T @ A @ mat_krao-self.X)\
                    /np.linalg.norm(self.X))  
                kron_err.append(np.linalg.norm(A.T @ A @ mat_kron-self.X)\
                    /np.linalg.norm(self.X)) 
            rel_err = [gaussian_err,krao_err,kron_err]
        return rel_err

#def plot_sim(rel_err, name, label)

if __name__ == '__main__':

    X1 = np.random.normal(0,1, (10000,1000)) 
    sim_len = Simulation(X1, "len", 20, (5,10), 'sgn') 
    rel_err = sim_len.run_sim() 

    plt.figure()
    plt.hist(rel_err[0])
    print("gaussian: mean" + str(np.mean(rel_err[0]))+"variance:" +str(np.var(rel_err[0])))
    plt.hist(rel_err[1])
    print("krao: mean" + str(np.mean(rel_err[1]))+"variance:" +str(np.var(rel_err[1])))
    plt.hist(rel_err[2])
    print("kron: mean" + str(np.mean(rel_err[2]))+"variance:" +str(np.var(rel_err[2])))
    plt.show()




'''
    # Column space
    # rank = 10
    X2 = np.diag(np.hstack((np.repeat(10,10), np.repeat(0,990))))
    X2 = X2 + np.random.normal(0,1,(1000, 1000)) 
    sim_col = Simulation(X2, "len", 625, (25, 25)) 
    print("Gg")
    rel_err = sim_col.run_sim() 

    plt.figure()
    plt.hist(rel_err[0])
    print("gaussian: mean: " + str(np.mean(rel_err[0]))+" variance: "\
     +str(np.var(rel_err[0])))
    plt.hist(rel_err[1])
    print("krao: mean: " + str(np.mean(rel_err[1]))+" variance: " \
        +str(np.var(rel_err[1])))
    plt.hist(rel_err[2])
    print("kron: mean: " + str(np.mean(rel_err[2]))+" variance: " \
        +str(np.var(rel_err[2])))
    plt.show()
'''

