
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import deepxde as dde
import winsound
import time
from decimal import *

if __name__ == "__main__":
    start = time.time()
    
    def pde(x, y):
        dy_x = tf.gradients(y, x)
        dy_xx = tf.gradients(dy_x, x)[0]
        return -eps*dy_xx + dy_x
    
    def diff(x, y):
        return tf.gradients(y, x)[0]
    
    def boundary(x, on_boundary):
        return on_boundary
    
    def func(x):
        return x
        
    epslist = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    colp_list = [64, 128, 256, 512, 1024]
    error = np.zeros((5,2))

    erroru = np.zeros((len(epslist),len(colp_list)))
    errorux = np.zeros((len(epslist),len(colp_list)))
    errorf = np.zeros((len(epslist),len(colp_list)))

    for eps in [10**-2]:
        epss = Decimal(eps)
        ksi = (2/epss)/(Decimal(np.exp(1/epss))-1)
        mi = ksi**2*epss/2*(np.exp(2/epss)-1)
        lamb = 1/(1+mi)
        lamb = float(lamb)/50.0
        lamb = 1/2
        weights = [lamb, 1-lamb] #[int, bdry]
        for num_colp in [128]:
            
            geom = dde.geometry.Interval(0, 1)
            bc = dde.DirichletBC(geom, func, boundary)
            data = dde.data.PDE(geom, pde, bc, num_colp, 2, solution=func, num_test=100)
            
            activation = "tanh"
            initializer = "Glorot uniform"
            net = dde.maps.FNN(layer_size, activation, initializer)
            
            model = dde.Model(data, net)                      
            model.compile("L-BFGS-B", lr=0.001, metrics=["l2 relative error"], loss_weights=weights)
            
            checkpointer = dde.callbacks.ModelCheckpoint(
                "./model/model.ckpt", verbose=1, save_better_only=True
            )
            movie = dde.callbacks.MovieDumper(
                "model/movie", [-1], [1], period=100, save_spectrum=True, y_reference=func
            )
            losshistory, train_state = model.train(
                epochs=10000, callbacks=[checkpointer, movie]    
            )
            
            #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
            
            # Plotting u and u_x
            #x = geom.uniform_points(1000, True)
            xas = 10000
            x = np.linspace(0,1,num=xas).reshape((xas,1))  # points for plotting
            u_pred = model.predict(x, operator=None)
            u_x_pred = model.predict(x, operator=diff)
            u_star = np.zeros((xas,1))
            u_x_star = np.zeros((xas,1))
            #xx = x[1:xas]
            for i in range(0,xas):
                xx = Decimal(x[i,0])
                u_star[i,0] = (1-Decimal(np.exp(xx/epss)))/(1-Decimal(np.exp(1/epss))) # original bdry conditions
                u_x_star[i,0] = Decimal(np.exp(xx/epss))/epss/(Decimal(np.exp(1/epss))-1) # "
   
            # Plotting everything
            f = model.predict(x, operator=pde)
            plt.figure()
            plt.plot(x, f)
            plt.xlabel("x")
            plt.ylabel("PDE residue")
            plt.savefig(str(m)+'res eps = ' + str(eps) + ' colp ' + str(num_colp)+'.png')
            #plt.close()
            
            plt.figure()
            plt.plot(x, u_x_pred, 'b', label='du/dx prediction')
            plt.plot(x, u_x_star, 'r--', label='du/dx exact')
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("du/dx")
            plt.title('eps = ' + str(eps))
            plt.savefig(str(m)+'dudx eps = ' + str(eps) + ' colp ' + str(num_colp)+'.png')
            #plt.close()
            
            plt.figure()
            plt.plot(x,u_pred,'b', label='u prediction')
            plt.plot(x,u_star,'r--', label='u exact')
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.title('eps = ' + str(eps))
            plt.savefig(str(m)+'u eps = ' + str(eps) + ' colp ' + str(num_colp)+'.png')
            #plt.close()
            
            plt.figure()
            xl=9900
            plt.plot(x[xl:xas],u_pred[xl:xas],'b', label='u prediction')
            plt.plot(x[xl:xas],u_star[xl:xas],'r--', label='u exact')
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.title('eps = ' + str(eps))
            #plt.savefig('u eps = ' + str(eps) + ' colp ' + str(num_colp)+'.png')
            #plt.close()            

            
            i = epslist.index(eps)
            j = colp_list.index(num_colp)
            erroru[i,j] = np.linalg.norm(u_pred[:,0].reshape((xas,1))-u_star)/np.linalg.norm(u_star,2)
            errorux[i,j] = np.linalg.norm(u_x_pred[:,0].reshape((xas,1))-u_x_star)/np.linalg.norm(u_x_star,2)
            errorf[i,j] = np.linalg.norm(f)
            print('Error u: ' + str(erroru[i,j]))
            print('Error f: ' + str(errorf[i,j]))

            with open("errors.dat", "wb") as f:
                pickle.dump([error], f)
    end = time.time()
    print("Running time: " + str(end - start))
    winsound.Beep(440, 1000)
    with open("errors.dat", "rb") as f:
        xxx = pickle.load(f)
