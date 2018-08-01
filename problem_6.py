import numpy as np
import matplotlib.pyplot as plt
from get_data_set import get_data_set

plot_result = True
plot_like_task = False
same_colorbar = False

tol = 1e-5
max_iter = 2000
tot_iter = 600
lam = 3
eta = 0.1

def J(A_arg, Z_arg):
    return np.nansum(np.abs(A_arg - Z_arg)**2)

def J_grad(A_arg, Z_arg):
    return np.nan_to_num(2*(Z_arg - A_arg))

def proximal(mu_arg, lambda_arg):
    return np.max((np.zeros(np.size(mu_arg)), np.abs(mu_arg) - lambda_arg), 0)*np.sign(mu_arg)

def proximal_gradient(Z, A, J, J_grad, lam, eta, tol, max_iter):
    gradient = J_grad(A, Z)
    Z_prev = 100*Z
    iter = 0
    eta_init = eta
    #while np.linalg.norm(gradient+ lam*np.linalg.norm(Z,'nuc')) > tol and iter < max_iter:
    for k in range(tot_iter):
    #while np.linalg.norm(np.nan_to_num(A) + np.isnan(A)*Z - Z_prev) > tol and iter < max_iter:
        eta = eta_init
        u, s, vh = np.linalg.svd(Z - eta * gradient, full_matrices=False)
        s = proximal(s, lam * eta)
        Z_prev = np.nan_to_num(A) + np.isnan(A)*Z
        Z = np.dot(u, np.dot(np.diag(s), vh))
        # result_vec = np.r_[result_vec, np.linalg.norm(w - w_result)]
        gradient = J_grad(A, Z)
        iter = iter + 1
    return np.nan_to_num(A) + np.isnan(A)*Z, iter

def NMF(A, tol):
    # Lee and Seung's multiplicative update rule
    r = 4
    V = np.nan_to_num(A)
    V_prev = 100*V
    V_mem = np.isnan(A)
    m, n = np.shape(A)
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)
    eps = 0 #1e-12 # term to avoid dividing by 0
    iter = 0
    for k in range(tot_iter):
    #while np.linalg.norm(V - V_prev) > tol and iter < max_iter:
        H = H*np.matmul(W.T,V)/(np.matmul(np.matmul(W.T, W), H) + eps)
        W = W*np.matmul(V, H.T)/(np.matmul(np.matmul(W, H), H.T) + eps)
        #Standardization of col size
        #W = W/(np.sum(W**2, axis=0)[None, :]**0.5)
        #H = H*(np.sum(W**2,axis=0)[:, None]**0.5)
        V_prev = V
        V = V*(~V_mem) + np.matmul(W,H)*V_mem
        iter += 1
    return np.nan_to_num(A) + np.isnan(A)*np.matmul(W,H), iter


A, A_0 = get_data_set(3)

n, r = np.shape(A)
Z_init = np.random.rand(n, r)

Z_NMF, iter_nmf = NMF(A, tol)
Z, iter_prox = proximal_gradient(Z_init, A, J, J_grad, lam, eta, tol, max_iter)
print('Number of iterations')
print(("Proximal gradient iterations: %i") % iter_prox)
print(("NMF iterations: %i") % iter_nmf)



plt.figure()

plt.subplot(311)
plt.imshow(A)
plt.colorbar()
if plot_like_task:
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color='darkslateblue')
plt.title("Dataset")

plt.subplot(312)
plt.imshow(Z)  # , interpolation='nearest', vmin=np.nan)???
plt.colorbar()
plt.title("Recoverey method 1")
plt.subplots_adjust(hspace=0.3)

plt.subplot(313)
plt.imshow(Z_NMF)  # , interpolation='nearest', vmin=np.nan)???
plt.colorbar()
plt.title("Recovery method 2")
plt.subplots_adjust(hspace=0.3)


plt.show(block=False)
plt.show()

