import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import func as f


N = 200
#x1 = np.random.uniform(-1, 1, N)
#x2 = np.random.uniform(-1, 1, N)
#f.WritingInFile(['x1', 'x2'], [x1, x2], 'x1x2.txt')
x1, x2 = f.get_x1_x2('x1x2.txt')
f.FindResponds(x1, x2, 'u_y_ej_x1_x2.txt', N)
Y = np.array(f.get_y('u_y_ej_x1_x2.txt'))
matr_X = f.create_X_matr(x1, x2, N)
est_tetta = f.parameter_estimation_tetta(matr_X, Y)
e_t, est_sigm, e_t_2 = f.residual(Y, x1, x2, est_tetta, N)
z_t = f.Z_t(x1, x2, N)
ESS = f.regres_construction(e_t_2, est_sigm, z_t, N)