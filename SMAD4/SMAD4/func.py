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
    WritingInFile(['sigma'], [[sigm]], 'sigma'+outputFile[6:])
    WritingInFile(['U', 'ej', 'y'], [U, ej, y], outputFile)
    return y

####################################
def create_X_matr(x1, x2, N):
    X = []
    for i in range(N):
        X.append([])
        X[i].append(1.)
        X[i].append(func_2(x1[i]))
        X[i].append(func_3(x1[i]))
        X[i].append(func_4(x2[i]))
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

def Z_t(x1, x2, N):
    z_t = []
    for i in range(N):
        z_t.append([])
        z_t[i].append(1.)
        z_t[i].append(x1[i] ** 2 + x2[i] ** 2)
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
    hi = 5.99
    a = False
    #гетероскедастичность присутствует
    if(Ess > hi):
        a = True
    else:
        a = False
    return a
def test_Breusha_Pagana(x1, x2, Y, est_tetta, N):
    e_t, est_sigm, e_t_2 = residual(Y, x1, x2, est_tetta, N)
    z_t = Z_t(x1, x2, N)
    ESS = regres_construction(e_t_2, est_sigm, z_t, N)
    return ESS
#######################################

def func_2(x):
    return x

def func_3(x):
    return sp.exp(-x ** 2)

def func_4(x):
    return x ** 2
