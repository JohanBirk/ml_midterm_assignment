import numpy as np
import matplotlib.pyplot as plt

use_linesearch = False
font_size = 18
legend_font_size = 12

def J(A_arg, mu_arg, w_arg):
    return np.matmul(np.matmul((w_arg-mu_arg).T,A_arg),w_arg-mu_arg)

def J_grad(A_arg, mu_arg, w_arg):
    return 2*np.matmul((w_arg-mu_arg).T, A_arg)

def proximal(mu_arg, lambda_arg):
    return np.max((np.zeros(np.size(mu_arg)), np.abs(mu_arg) - lambda_arg), 0)*np.sign(mu_arg)

def proximal_gradient(w, A, mu, J, J_grad, eta, tol, max_iter):
    gradient = J_grad(A, mu, w)
    iter = 0
    local_max_iter = 30
    result_vec = np.zeros(1)
    result_vec[0] = np.linalg.norm(w - w_result)
    eta_init = eta
    while np.linalg.norm(gradient+ lam*np.sign(w)) > tol and iter < max_iter:
        rho = 0.5
        eta = eta_init
        if use_linesearch:
            local_iter = 0
            z = proximal(w - eta*gradient, lam*eta)
            while J(A, mu, z) >= J(A, mu, w) + np.dot(gradient, z - w) + (1/(2*eta))*np.linalg.norm(z - w, 2)**2 and local_iter < local_max_iter:
                eta = eta*rho
                local_iter += 1
            w = z
        else:
            w = proximal(w - eta * gradient, lam * eta)
        result_vec = np.r_[result_vec, np.linalg.norm(w - w_result)]
        gradient = J_grad(A, mu, w)
        iter = iter + 1
    return w, result_vec, iter

def q_func(t_arg):
    return (t_arg - 1)/(t_arg + 2)

def accelerated_proximal_gradient(w, A, mu, J, J_grad, eta, tol, max_iter):
    v = w
    gradient = J_grad(A, mu, v)
    iter = 0
    local_max_iter = 30
    result_vec = np.zeros(1)
    result_vec[0] = np.linalg.norm(w - w_result)
    eta_init = eta
    while np.linalg.norm(gradient + lam*np.sign(w)) > tol and iter < max_iter:
        rho = 0.75
        eta = eta_init
        if use_linesearch:
            local_iter = 0
            z = proximal(v - eta * gradient, lam * eta)
            while J(A, mu, z) >= J(A, mu, v) + np.dot(gradient, z - v) + (1 / (2 * eta)) * (np.linalg.norm(z - v,
                                                                                                          2))**2 and local_iter < local_max_iter:
                eta = eta * rho
                local_iter += 1
            w_prev = w
            w = z
        else:
            w_prev = w
            w = proximal(v - eta * gradient, lam * eta)
        v = w + q_func(iter+1) * (w - w_prev)
        result_vec = np.r_[result_vec, np.linalg.norm(w - w_result)]
        gradient = J_grad(A, mu, v)
        iter = iter + 1
    return w, result_vec, iter

def J_grad_ada(A_arg, mu_arg, w_arg, z_arg):
    # Seems kinda useless for this exercise ...
    return 2*np.matmul((w_arg-mu_arg).T, A_arg)

def AdaGrad(w, A, mu, J, J_grad_ada, eta, tol, max_iter):
    n, d = np.shape(A)
    delta = 0.02
    result_vec = np.zeros(1)
    result_vec[0] = np.linalg.norm(w - w_result)
    z = np.random.randint(0, n)
    gradient = J_grad_ada(A, mu, w, z)
    G = np.zeros(n)
    G += gradient**2
    H = G**0.5 + delta
    w_prev = w
    iter = 0
    while np.linalg.norm(gradient + lam*np.sign(w)) > tol and iter < max_iter:
        mu_t = w_prev - eta * gradient / H
        w = proximal(mu_t, lam*eta/H)
        result_vec = np.r_[result_vec, np.linalg.norm(w - w_result)]
        z = np.random.randint(0, n)
        gradient = J_grad_ada(A, mu, w, z)
        G += gradient**2
        H = G ** 0.5 + delta
        w_prev = w
        iter += 1
    return w, result_vec, iter

plt.close('all')
for i in range(5):
    tol = 1e-16
    w_init = np.array([3, -1])
    mu = np.array([1, 2])
    if i in range(4):
        A = np.array([[3, 0.5], [0.5, 1]])
        max_iter = 29
        if i == 0:
            lam = 2
            w_result = np.array([9 / 11, 12 / 11])
        elif i == 1:
            lam = 4
            w_result = np.array([7 / 11, 2 / 11])
        elif i == 2:
            max_iter = 20
            lam = 6
            w_result = np.array([1/3, 0])
        elif i == 3:
            max_iter = 173
            A = np.array([[300, 0.5], [0.5, 1]])
            lam = 0.89
            w_result = np.array([0.9992572, 1.55567702])
    else:
        w_result = np.array([1.00631613, 1.86506452])
        A = np.array([[250, 15], [15, 4]])
        lam = 0.89
    eta_t = 1 / np.max(np.linalg.eig(2 * A)[0])  # max(eig(A)) <= norm(A)???
    if i in range(4):
        plt.figure(i)
        w = w_init
        w_prox, result_vec_prox, iter_prox = proximal_gradient(w, A, mu, J, J_grad, eta_t, tol, max_iter)
        w_acc, result_vec_acc, iter_acc = accelerated_proximal_gradient(w, A, mu, J, J_grad, eta_t, tol, max_iter)
        plt.semilogy(result_vec_prox, '--', label='Proximal gradient')
        plt.semilogy(result_vec_acc, '--', label='Accelerated proximal gradient')
        print(("For \lambda = %i" % lam))
        print(('Proximal gradient iterations: %i') % iter_prox)
        print(('Accelerated proximal gradient iterations: %i') % iter_acc)
        plt.title(('$\lambda$ = %f' % lam), fontsize=font_size)
        plt.xlabel("No. of iterations", fontsize=font_size)
        plt.ylabel("$\|| w^{(t)} - \hat{w} \|| $", fontsize=font_size)
        plt.legend(fontsize=legend_font_size)
        print("Final gradient")
        print(J_grad(A, mu, w_acc) + lam * np.sign(w_acc))
        print(J_grad(A, mu, w_prox) + lam * np.sign(w_prox))
        print("Final w")
        print(w_acc)
        print(w_prox)
    else:
        max_iter = 200
        plt.figure(i)
        w = w_init
        w_prox, result_vec_prox, iter_prox = proximal_gradient(w, A, mu, J, J_grad, eta_t, tol, max_iter)
        w_acc, result_vec_acc, iter_acc = accelerated_proximal_gradient(w, A, mu, J, J_grad, eta_t, tol, max_iter)
        w_ada, result_vec_ada, iter_ada = AdaGrad(w, A, mu, J, J_grad_ada, 500 * eta_t, tol, max_iter)

        plt.semilogy(result_vec_prox, '--', label='Proximal gradient')
        plt.semilogy(result_vec_acc, '--', label='Accelerated proximal gradient')
        plt.semilogy(result_vec_ada, '--', label='AdaGrad')
        print(("For \lambda = %i" % lam))
        print(('Proximal gradient iterations: %i') % iter_prox)
        print(('Accelerated proximal gradient iterations: %i') % iter_acc)
        print(("AdaGrad iterations: %i") % iter_ada)
        plt.title(("$\lambda$ = %f" % lam), fontsize=font_size)
        plt.xlabel("No. of iterations", fontsize=font_size)
        plt.ylabel("$\|| w^{(t)} - \hat{w} \|| $", fontsize=font_size)
        plt.legend(fontsize=legend_font_size)
        print("Final gradient")
        print(J_grad(A, mu, w_ada) + lam * np.sign(w_ada))
        print("Final w")
        print(w_prox)
        print(w_acc)
        print(w_ada)
plt.show()

