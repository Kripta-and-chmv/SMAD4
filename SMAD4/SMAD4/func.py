import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st

####################################
def get_x1_x2(fname):
    str_file = []
    x1 = []
    x2 = []
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    for i in range(1, len(str_file)):
        s = str_file[i].expandtabs(1).rstrip()
        x1_el, x2_el = s.split(' ')
        x1.append(float(x1_el))
        x2.append(float(x2_el))
    return x1, x2

def get_y(fname):
    str_file = []
    y = []
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    for i in range(1, len(str_file)):
        s = str_file[i].expandtabs(1).rstrip()
        u, ej, y_el = s.split(' ')
        y.append(float(y_el))
    return y
def get_s(fname):
    str_file = []
    y = []
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    for i in range(1, len(str_file)):
        s = str_file[i].expandtabs(1).rstrip()
        y.append(float(s))
    return y
##############################
def WritingInFile(names, sequences, fileName):
    with open(fileName, 'w') as f:
        for i in range(len(names)):
            f.write(names[i] + ': ')
        f.write('\n')
        for j in range(len(sequences[0])):
            for i in range(len(names)):
                f.write(str(sequences[i][j]) + ' ')
            f.write('\n')

def FindResponds(x1, x2, outputFile, N):

    U = []
    y = []
    sigm = []
    ej = []
    for i in range(N):
        U.append(1. + func_2(x1[i]) - func_3(x1[i]) + func_4(x2[i]))
        sigm.append(math.sqrt(0.01 * x1[i] ** 2 + 0.1 * x2[i] ** 2))
    for j in range(N):
        ej.append(np.random.normal(0, sigm[j]))
    for i in range(N):
        y.append(U[i] + ej[i])   
    WritingInFile(['sigma'], [sigm], 'sigma.txt')
    WritingInFile(['U', 'ej', 'y'], [U, ej, y], outputFile)
    return sigm

####################################
def create_X_matr(x1, x2, N):
    X = [[1., func_2(el1), func_3(el1), func_4(el2) ] for el1, el2 in zip(x1, x2)]
    return np.array(X, dtype=float)

def parameter_estimation_tetta(matr_X, Y):
    XtX = np.matmul(matr_X.T, matr_X)
    XtX_1 = np.linalg.inv(XtX)
    XtX_1_Xt = np.matmul(XtX_1, matr_X.T)
    est_tetta = np.matmul(XtX_1_Xt, Y)
    return est_tetta

########################################
def residual(Y, x1, x2, est_tetta, N):
    e_t = []
    e_t_2 = []
    nu = []
    for i in range(N):
        f1 = []
        f1.append(1.)
        f1.append(func_2(x1[i]))
        f1.append(func_3(x1[i]))
        f1.append(func_4(x2[i]))
        f1 = np.array(f1, dtype=float)
        nu.append(np.matmul(f1.T, est_tetta))
    for j in range(N):
        e_t.append(Y[j] - nu[j])
        e_t_2.append(e_t[j] ** 2)
    est_sigm = math.sqrt(np.sum(e_t_2) / N)
    return e_t, est_sigm, e_t_2

def Z_t(sigm, N):
    #z_t = []
    for i in range(N):
        #z_t.append([])
        #z_t[i].append(1.)
        #z_t[i].append(0.01 * x1[i] ** 2 + 0.1 * x2[i] ** 2)
        z_t = [[1., el ** 2] for el in sigm]
    return np.array(z_t, dtype=float)
###############################
def regres_construction(e_t_2, est_sigm, z_t, N):
    ESS = []
    c_t = []
    for i in range(N):
        c_t.append(e_t_2[i] / (est_sigm ** 2))
    est_alpha = parameter_estimation_tetta(z_t, c_t)
    est_c_t = np.matmul(est_alpha.T, z_t.T)
    M_c_t = np.sum(c_t) / N
    for i in range(N):
        ESS.append((est_c_t[i] - M_c_t) ** 2)
    Ess = np.sum(ESS) / 2.0
    hi = 5.99####
    a = False
    #гетероскедастичность присутствует
    if(Ess / 2 > hi):
        a = True
    else:
        a = False
    return a, Ess

def test_Breusch_Pagan(x1, x2, sigm, Y, est_tetta, N):
    e_t, est_sigm, e_t_2 = residual(Y, x1, x2, est_tetta, N)
    z_t = Z_t(sigm, N)
    a, ESS = regres_construction(e_t_2, est_sigm, z_t, N)
    return ESS
#######################################
def test_Goldfeld_Quandt(x1, x2, Y, N):
    arr = []
    for i in range(N):
        arr.append(np.array([x1[i], x2[i], Y[i]]))
    new_arr = sorted(arr, key=lambda x: 0.01 * x[0] ** 2 + 0.1 * x[1] ** 2 )
    #граница 1
    n_c1 = int(N / 3)    
    n_c3 = N - n_c1
    #для первой части
    x1_c1 = np.array([new_arr[i][0]  for i in range(n_c1)])
    x2_c1 = np.array([new_arr[i][1]  for i in range(n_c1)])
    y_c1 = np.array([new_arr[i][2]  for i in range(n_c1)])
    #для последней части
    x1_c3 = np.array([new_arr[i][0] for j in range(n_c3, N)])
    x2_c3 = np.array([new_arr[i][1] for j in range(n_c3, N)])
    y_c3 = np.array([new_arr[i][2] for j in range(n_c3, N)])

    matrX_c1 = create_X_matr(x1_c1, x2_c1, n_c1)
    est_tetta_c1 = parameter_estimation_tetta(matrX_c1, y_c1)
    matrX_c3 = create_X_matr(x1_c3, x2_c3, n_c1)
    est_tetta_c3 = parameter_estimation_tetta(matrX_c3, y_c3)

    XTet_1 = np.matmul(matrX_c1, est_tetta_c1)
    difY_XTet_1 = y_c1 - XTet_1
    RSS_1 = np.matmul(difY_XTet_1.T, difY_XTet_1)

    XTet_3 = np.matmul(matrX_c3, est_tetta_c3)
    difY_XTet_3 = y_c3 - XTet_3
    RSS_2 = np.matmul(difY_XTet_3.T, difY_XTet_3)
    rss = RSS_2 / RSS_1
    F = 1.26457
    a =  False
    #гетероскедастичность присутствует
    if(rss > F):
        a = True
    else:
        a = False
    return rss
#######################################################
def parameter_estimation_OMNK(sigm, matr_X, Y):
    V = np.diag(sigm)
    V_1 = np.linalg.inv(V)
    X1 = np.matmul(matr_X.T, V_1)
    X2 = np.matmul(X1, matr_X)
    X3 = np.linalg.inv(X2)
    X4 = np.matmul(X3, matr_X.T)
    X5 = np.matmul(X4, V_1)
    est_tetta = np.matmul(X5, Y)
    return est_tetta

def check_est(est_tetta, est_omnk):
    tetta = [1., 1., -1., 1.]
    distMNK = np.sum((tetta - est_tetta) ** 2)
    distOMNK = np.sum((tetta - est_omnk) ** 2)
    return distMNK, distOMNK 
def func_2(x):
    return x

def func_3(x):
    return sp.exp(-x ** 2)

def func_4(x):
    return x ** 2
