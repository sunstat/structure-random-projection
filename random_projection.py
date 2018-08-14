import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(0, 1, 10000)

print(x.shape)


rp_res = []
srp_res = []


for i in range(100):
    A = np.random.normal(0, 1, (10,10000))/np.sqrt(10)
    rp_res.append(np.linalg.norm(np.dot(A,x)))
    A = np.random.normal(0, 1, (10, 100))
    B = np.random.normal(0, 1, (10, 100))
    rm = np.kron(A,B)
    srp_res.append(np.linalg.norm(np.dot(np.kron(A,B)/10, x)))

print(np.linalg.norm(x))
print(np.mean(rp_res))
print(np.mean(srp_res))


print(np.std(rp_res))
print(np.std(srp_res))

plt.hist(rp_res)
plt.hist(srp_res)
plt.show()