import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import hdmedians as hd 


def random_matrix_generator(m, n, typ = 'g', std = 1, sparse_factor = 0.1):
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
        return np.random.choice([-1, 0, 1], size = (m,n), p = \
            [1/(2*np.sqrt(n)), 1- 1/np.sqrt(n), 1/(2*np.sqrt(n))])*np.sqrt(np.sqrt(n))*std
    elif typ == 'sp':
        return np.random.choice([-1,0,1], size = (m,n), p = [sparse_factor/2, \
            1- sparse_factor, sparse_factor/2])*np.sqrt(1/sparse_factor)*std

def inner_product(X): 
    # Compute pairwise inner product for each column of matrix X  (Here, each column represents a single data point)
    k = 0
    nrow, ncol = X.shape
    prods = np.zeros((ncol, ncol))
    for i in range(nrow):
        for j in range(i, ncol): 
            prods[i,j] = np.inner(X[:,i],X[:,j])
    return prods

def square_tensor_gen(n, r, dim = 3,  typ = 'id', noise_level = 0, seed = None):
    '''
    :param n: size of the tensor generated n*n*...*n
    :param r: rank of the tensor or equivalently, the size of core tensor
    :param dim: # of dimensions of the tensor, default set as 3
    :param typ: identity as core tensor or low rank as core tensor
    :param noise_level: sqrt(E||X||^2_F/E||error||^_F)
    :return: The tensor with noise, and The tensor without noise
    '''
    if seed: 
        np.random.seed(seed) 

    types = set(['id', 'lk', 'fpd', 'spd', 'sed', 'fed'])
    assert typ in types, "please set your type of tensor correctly"
    total_num = np.power(n, dim)

    if typ == 'id':
        elems = [1 for _ in range(r)]
        elems.extend([0 for _ in range(n-r)])
        noise = np.random.normal(0, 1, [n for _ in range(dim)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0 +noise*np.sqrt((noise_level**2)*r/total_num), X0
        
    if typ == 'spd':
        elems = [1 for _ in range(r)]
        elems.extend([1.0/i for i in range(2, n-r+2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0 

    if typ == 'fpd':
        elems = [1 for _ in range(r)]
        elems.extend([1.0/(i*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0

    if typ == 'sed':
        elems = [1 for _ in range(r)]
        elems.extend([np.power(10, -0.25*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0

    if typ == 'fed':
        elems = [1 for _ in range(r)]
        elems.extend([np.power(10, (-1.0)*i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0, X0 

    if typ == "lk":
        core_tensor = np.random.uniform(0,1,[r for _ in range(dim)])
        arms = []
        tensor = core_tensor
        for i in np.arange(dim):
            arm = np.random.normal(0,1,size = (n,r))
            arm, _ = np.linalg.qr(arm)
            arms.append(arm)
            tensor = tl.tenalg.mode_dot(tensor, arm, mode=i)
        true_signal_mag = np.linalg.norm(core_tensor)**2
        noise = np.random.normal(0, 1, np.repeat(n, dim))
        X = tensor + noise*np.sqrt((noise_level**2)*true_signal_mag/np.product\
            (total_num))
        return X, tensor


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


'''
class Simulation(object):
    """
    :param X: input matrix of size m x n 
    :param sim_typ: type of the simulation "len" (length preservation), "col": 
        (column space)
    :param k: the reduced dimension
    :param split: k/split[0] * m/split[1] is the size of the kronecker component 
    """
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
                mat_krao1 = random_matrix_generator(self.k, int(m/self.split[1]), self.gen_typ)
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
                mat_kron1 = random_matrix_generator(int(self.k/self.split[0]),\
                    int(m/self.split[1]), self.gen_typ)
                mat_kron2 = random_matrix_generator \
                    (self.split[0], self.split[1])
                mat_krao = tl.tenalg.khatri_rao([mat_krao1.T,mat_krao2.T]).T
                mat_kron = np.kron(mat_kron1,mat_kron2)
                gaussian_err.append(np.linalg.norm(mat.T @ mat @ self.X-self.X)\
                    /np.linalg.norm(self.X)) 
                krao_err.append(np.linalg.norm(mat_krao.T @ mat_krao @ self.X-self.X)\
                    /np.linalg.norm(self.X))  
                kron_err.append(np.linalg.norm(mat_kron.T @ mat_kron @ self.X-self.X)\
                    /np.linalg.norm(self.X)) 
            rel_err = [gaussian_err,krao_err,kron_err]
        return rel_err

if __name__ == '__main__':
    X1 = np.random.normal(0,1, (10000,1000)) 
    sim_len = Simulation(X1, "len", 20, (5,10), 'sp1') 
    rel_err = sim_len.run_sim() 

    plt.figure()
    plt.hist(rel_err[0])
    print("gaussian: mean" + str(np.mean(rel_err[0]))+"variance:" +str(np.var(rel_err[0])))
    plt.hist(rel_err[1])
    print("krao: mean" + str(np.mean(rel_err[1]))+"variance:" +str(np.var(rel_err[1])))
    plt.hist(rel_err[2])
    print("kron: mean" + str(np.mean(rel_err[2]))+"variance:" +str(np.var(rel_err[2])))
    plt.show()




    # Column space
    # rank = 10
    X2 = np.diag(np.hstack((np.repeat(10,10), np.repeat(0,990))))
    X2 = X2 + np.random.normal(0,1,(1000, 1000)) 
    sim_col = Simulation(X2, "col", 625, (25, 25)) 
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