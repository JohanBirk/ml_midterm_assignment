import numpy as np

def get_data_set(data_set):
    # Generates data sets. Stops execution though
    # For 1,2 it returns X, y; for 3 it returns A
    if data_set == 1:
        n = 100
        x = 3*(np.random.rand(n, 2)-0.5)
        radius = np.power(x[:, 0], 2) + np.power(x[:, 1], 2)
        y = np.bitwise_and(radius > 0.7 + 0.1 * np.random.rand(n), radius < 2.2 + 0.1 * np.random.rand(n))
        y = 2*y-1
        return x, y
    elif data_set == 2:
        n = 40
        omega = np.random.normal(0, 1, 1)
        noise = 0.8 * np.random.normal(0, 1, n)
        x = np.random.normal(0, 1, (n, 2))
        y = 2 * (omega * x[:, 0] + x[:, 1] + noise > 0) - 1
        return x, y
    elif data_set == 3:
        np.random.seed(18)
        m = 20
        n = 40
        r = 2
        A = np.matmul(np.random.rand(m, r), np.random.rand(r, n))
        A_0 = A
        ninc = 100
        Q = np.random.permutation(m * n) - 1
        Q = Q[0:ninc]
        # A bit redundant, but it should do exactly the same as the proposed matlab-dataset
        A = np.reshape(A.T, (1, m*n))
        for i in range(ninc):
            A[0, Q[i]] = np.nan
        A = np.reshape(A, (n, m)).T
        return A, A_0
    else:
        print("Invalid dataset! Choose from 1 to 3")
