import numpy as np
import matplotlib.pyplot as plt
from get_data_set import get_data_set

font_size = 14 # for comparison plots
legend_font_size = 12

lam = 5
tol = 1.0e-7
max_iter = 300
tot_iterations = 20

x, y = get_data_set(2)
n, d = np.shape(x)

w = np.ones(d)*1


def f(x_arg, w_arg):
    return 2 * (np.dot(x_arg, w_arg) > 0) - 1

def J(x_arg, y_arg, w_arg):
    temp = 0
    for i in range(n):
        temp = temp + np.log(1 + np.exp(-y_arg[i]*np.dot(w_arg, x_arg[i, :])))
    return temp + lam*np.dot(w_arg, w_arg)

def J_grad(x_arg, y_arg, w_arg):
    temp = 0
    for i in range(n):
        temp = temp - y_arg[i]*np.exp(-y_arg[i]*np.dot(w_arg, x_arg[i, :])) \
               / (1 + np.exp(-y_arg[i]*np.dot(w_arg, x_arg[i, :])))*x_arg[i, :]
    return temp + 2*lam*w_arg

def J_hess(x_arg, y_arg, w_arg):
    temp = 0
    for i in range(n):
        temp = temp + ((np.exp(-y_arg[i] * np.dot(w_arg, x_arg[i, :])) \
               / ((1 + np.exp(-y_arg[i] * np.dot(w_arg, x_arg[i, :])))**2)) * np.outer(x_arg[i, :], x_arg[i, :]))
    return temp + 2*lam*np.eye(d)

def stepSizeSimple(x, y, w, J, gradient, direc):
    alpha_init = 1
    c1 = 0.25
    rho = 0.5
    alpha = alpha_init
    while J(x, y, w + alpha * direc) > J(x, y, w) + c1 * alpha * np.dot(gradient, direc):
        alpha = alpha * rho
    return alpha

def steepGrad(w, x, y, J, J_grad, tol, max_iter):
    gradient = J_grad(x, y, w)
    d = np.size(w)
    w_vec = np.zeros((d,1))
    w_vec[:, 0] = w
    iter = 0
    #while np.linalg.norm(gradient) > tol and iter < max_iter:
    for k in range(tot_iterations):
        direc = -gradient
        val = J(x,y,w)
        alpha = stepSizeSimple(x, y, w, J, gradient, direc)
        w = w + alpha * direc
        w_vec = np.c_[w_vec, w]
        gradient = J_grad(x, y, w)
        iter = iter + 1
    return w, w_vec, iter

def newton(w, x, y, J, J_grad, J_hess, tol, max_iter):
    gradient = J_grad(x, y, w)
    iter = 0
    d = np.size(w)
    w_vec = np.zeros((d,1))
    w_vec[:, 0] = w
    #while np.linalg.norm(gradient) > tol and iter < max_iter:
    for k in range(tot_iterations):
        hessian = J_hess(x, y, w)
        direc = -np.linalg.solve(hessian, J_grad(x, y, w))
        alpha = stepSizeSimple(x, y, w, J, gradient, direc)
        w = w + alpha * direc
        w_vec = np.c_[w_vec, w]
        gradient = J_grad(x, y, w)
        iter = iter + 1
    return w, w_vec, iter

def newtonRef(w, x, y, J, J_grad, J_hess, tol, max_iter):
    gradient = J_grad(x, y, w)
    iter = 0
    d = np.size(w)
    w_vec = np.zeros((d,1))
    w_vec[:, 0] = w
    while np.linalg.norm(gradient) > tol and iter < max_iter:
        hessian = J_hess(x, y, w)
        direc = -np.linalg.solve(hessian, J_grad(x, y, w))
        alpha = stepSizeSimple(x, y, w, J, gradient, direc)
        w = w + alpha * direc
        w_vec = np.c_[w_vec, w]
        gradient = J_grad(x, y, w)
        iter = iter + 1
    return w, w_vec, iter

w_init = w
w_ref, w_vec_ref, iter_ref = newtonRef(w_init, x, y, J, J_grad, J_hess, 1e-15, max_iter)
print(('Newton reference iterations: %i') %iter_ref)
w_grad, w_vec_grad, iter_grad = steepGrad(w_init, x, y, J, J_grad, tol, max_iter)
w_newt, w_vec_newt, iter_newt = newton(w_init, x, y, J, J_grad, J_hess, tol, max_iter)

plt.close('all')
plt.figure()
w_d, w_n = np.shape(w_vec_grad)
J_norm_grad = np.zeros(w_n)
J_vec_grad = np.zeros(w_n)
for i in range(w_n):
    J_norm_grad[i] = np.abs(J(x, y, w_vec_grad[:, i]) - J(x, y, w_ref))
    J_vec_grad[i] = J(x, y, w_vec_grad[:, i])
plt.semilogy(J_norm_grad,':*', label='Gradient Descent')
print(('Gradient Descent iterations: %i') %iter_grad)

w_d, w_n = np.shape(w_vec_newt)
J_norm_newt = np.zeros(w_n)
J_vec_newt = np.zeros(w_n)
for i in range(w_n):
    J_norm_newt[i] = np.abs(J(x, y, w_vec_newt[:, i]) - J(x, y, w_ref))
    J_vec_newt[i] = J(x, y, w_vec_newt[:, i])
plt.semilogy(J_norm_newt, ':*', label='Newton')
print(('Newton iterations: %i') % iter_newt)
plt.legend(fontsize=legend_font_size)
plt.xlabel("No. of iterations", fontsize=font_size)
plt.ylabel("$|\mathcal{J} ( w^{(t)}) - \mathcal{J} ( \hat{w})| $", fontsize=font_size)

plt.figure()
plt.plot(J_vec_grad,':*', label='Gradient Descent')
plt.plot(J_vec_newt, ':*', label='Newton')
plt.legend(fontsize=legend_font_size)
plt.xlabel("No. of iterations", fontsize=font_size)
plt.ylabel("$\mathcal{J} ( w^{(t)}) $", fontsize=font_size)


plt.figure()
plt.title("Dataset 2\nOptimizer:  Newton",fontsize=font_size)
max_val = np.max(np.abs(x))
n_boundary = 50
x_1 = np.linspace(-max_val, max_val, n_boundary)
x_2 = np.linspace(-max_val, max_val, n_boundary)
xx = np.meshgrid((x_1, x_2))
for i in range(n_boundary):
    for j in range(n_boundary):
        classify = f([x_1[i], x_2[j]], w_ref)
        if classify == 1:
            plt.scatter(x_1[i], x_2[j], color='lightcoral', marker=',')
        else:
            plt.scatter(x_1[i], x_2[j], color='skyblue', marker=',')
for i in range(n):
    if y[i] == 1:
        plt.scatter(x[i, 0], x[i, 1], facecolors='none', edgecolors='red', marker='o')
    else:
        plt.scatter(x[i, 0], x[i, 1], facecolors='none', edgecolors='blue', marker='s')
plt.show(block=False)
plt.show()
