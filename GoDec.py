import numpy as np

def LowRankMatrix(A,L,rank,power):
    _, n = A.shape
    Y2 = np.random.rand(n,rank)
    for i in range(power+1):
        Y1 = L.dot(Y2)
        Y2 = L.T.dot(Y1)
    Q,_ = np.linalg.qr(Y2,mode='reduced')
    L = (L.dot(Q)).dot(Q.T)
    return L

def SparseMatrix(A,S,card):
    A_vec = A.reshape(-1)
    S_vec = S.reshape(-1)
    idx = abs(A_vec).argsort()[::-1]
    S_vec[idx[:card]] = A_vec[idx[:card]]
    S = S_vec.reshape(A.shape)
    return S

def ComputeError(X,L,S):
    E = np.linalg.norm(X-L-S,'fro')**2/np.linalg.norm(X,'fro')**2
    return E

def GoDecFast(X,rank,card,power,epsilon):
    MaxIter = 100
    t = 1
    L = X
    S = np.zeros(X.shape)
    E = [1]
    while True:
        L = LowRankMatrix(X-S,L,rank,power)
        S = SparseMatrix(X-L,S,card)
        E.append(ComputeError(X,L,S))
        error = abs(E[t] - E[t-1])
        if (error <= epsilon) or (t >= MaxIter):
            break
        else:
            t = t + 1
    return L,S,E
