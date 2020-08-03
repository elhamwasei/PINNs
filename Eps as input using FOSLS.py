import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.pyplot as plt
import winsound
from decimal import *
import pickle

###############################################################################
############################## Helper Functions ###############################
###############################################################################

def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

def blockmatrix(weights1, weights2):
    weights = []
    num_layers = len(weights1)

    for i in range(0,num_layers):
        W1 = weights1[i]
        W2 = weights2[i]
        if i==0:
            W = tf.concat([W1, W2],1)
        else:
            a1 = tf.zeros(shape=W1.shape, dtype = tf.float64)
            a2 = tf.zeros(shape=W2.shape, dtype = tf.float64)
            b1 = tf.concat([W1,a1],1)
            b2 = tf.concat([a2,W2],1)
            W = tf.concat([b1, b2],0)
        weights.append(W)
    return weights

def initialize_NN(layers):
    weights1 = []
    weights2 = []
    biases = []
    num_layers = len(layers)
    for l in range(0,num_layers-1):   
        # weights to map from layer l to layer (l+1)
        W1 = xavier_init(size=[layers[l], layers[l+1]])
        W2 = xavier_init(size=[layers[l], layers[l+1]])
        # biases added at layer (l+1)
        b = tf.Variable(tf.zeros([1,2*layers[l+1]], dtype=tf.float64), dtype=tf.float64)
        weights1.append(W1)
        weights2.append(W2)
        biases.append(b)
    return weights1, weights2, biases

def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    # initialize H as the NN's input 
    H = X
    # step through the first (n-1) layers 
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        # H <- activation_function(W*H + b)
        H = tf.math.tanh(tf.add(tf.matmul(H, W), b))
    # at the last layer, change activation to identity 
    W = weights[-1]
    b = biases[-1]
    # H <- W*H + b
    H = tf.add(tf.matmul(H, W), b)
    return H

###############################################################################
################################ Advection-diffusion Class ####################
###############################################################################

class AdvDiff:
    def __init__(self, eps, P, dx, layers, lb, ub, lamb):
        self.dx = dx
        self.lamb = lamb

            
        # store lower and upper bounds for parameters
        self.lb = lb
        self.ub = ub

        # Initiatization of PDE solver
        self.init(eps, P, layers)

        # Initialize tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def callback(self, loss):
        print('Loss: %e' % (loss))

    def init(self, eps, P_f, layers):
        # PDE solver is trained by:
        # * reproducing boundary conditions (i.e., solution at x = 0 and x = 1)
        # * minimizing PDE residual in the interior
        
        self.eps_lb = eps
        self.x_lb = 0*eps + self.lb[1]
        # for upper boundary
        self.eps_ub = eps
        self.x_ub = 0*eps + self.ub[1]
        # for minimizing PDE residual
        self.eps_f = P_f[:,0:1]
        self.x_f = P_f[:,1:2]
        
        ################
        # Initialize neural network for u
        ################
        self.weights1, self.weights2, self.biases = initialize_NN(layers)
        # store layer sizes
        self.layers = layers
        
        ################
        # Create placeholder variables for tensorflow.
        # These help create the neural network and define the loss function.
        # The placeholders can be replaced with actual values to train the network, 
        # or to predict the solution based on input (eps, x). 
        ################
        # tf placeholder variables
        # * for the lower boundary
        self.eps_lb_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.x_lb_tf = tf.placeholder(tf.float64, shape=[None, 1])
        # * for the upper boundary
        self.eps_ub_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.x_ub_tf = tf.placeholder(tf.float64, shape=[None, 1])
        # * for interior collocation points
        self.eps_f_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float64, shape=[None, 1])

        # tf graphs
        # * solution at lb is a function of eps_lb, x_lb
        self.u_lb_pred = self.net_u(self.eps_lb_tf, self.x_lb_tf)
        self.u_lb_pred = self.u_lb_pred[:,0]
        # * solution at ub is a function of eps_ub, x_ub
        self.u_ub_pred = self.net_u(self.eps_ub_tf, self.x_ub_tf)
        self.u_ub_pred = self.u_ub_pred[:,0]
        # * PDE residual is a function of eps_f, x_f
        self.f1_pred = self.net_f1(self.eps_f_tf, self.x_f_tf)
        self.f2_pred = self.net_f2(self.eps_f_tf, self.x_f_tf)
        # * solution is a function of eps, x
        self.u_pred  = self.net_u(self.eps_f_tf, self.x_f_tf)

        ################
        # Loss function and solvers
        ################
        # loss at lower dirichlet boundary + loss at upper dirichlet boundary
        # + PDE residual loss at internal collocation points
        self.loss = ( tf.reduce_sum(tf.square(self.u_lb_pred)) + \
                        tf.reduce_sum(tf.square(self.u_ub_pred-1.0)) )*(1-self.lamb) + \
                      (  tf.reduce_sum(tf.multiply(tf.square(self.f1_pred),self.dx)) + \
                        tf.reduce_sum(tf.multiply(tf.square(self.f2_pred),self.dx)))*self.lamb

        # Optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                             var_list = self.weights1 + self.weights2 + self.biases,
                             method = 'L-BFGS-B',
                             options = {'maxiter': 50000,
                                        'maxfun': 50000,
                                        'maxcor': 50,
                                        'maxls': 50,
                                        'gtol': 1e-8,
                                        'ftol': 1.0*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                 var_list = self.weights1 + self.weights2 + self.biases)

    def net_u(self, eps, x):
        # Neural network mapping [eps, x] to u
        X = tf.concat([eps,x],1)
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #rescale to -1 to 1
        weights = blockmatrix(self.weights1, self.weights2)
        u = neural_net(H, weights, self.biases)
        return u

    def net_f1(self, eps, x):    
        # Neural network mapping [eps,x] to PDE residual dsig/dx + du/dx=0
        
        # network for solution u
        u = self.net_u(eps,x)
        
        # solution network's derivatives
        u_x = tf.gradients(u[:,0], x)[0]
        sig_x = tf.gradients(u[:,1], x)[0]
        # PDE residual 1
        f = u_x + sig_x
        return f
        
    def net_f2(self, eps, x):
        # Neural network mapping [eps,x] to PDE residual  sig + eps*du/dx = 0
        
        # network for solution u
        u = self.net_u(eps,x)
        # solution network's derivatives
        u_x = tf.gradients(u[:,0], x)[0]
        v = tf.reshape(u[:,1], shape=tf.shape(u_x))
        # PDE residual 2      
        f = 1/tf.math.sqrt(eps)*v + tf.math.sqrt(eps)*u_x
        return f
    
    def train(self, N_iter):
        # To train the networks, the tf placeholders are replaced by the training data supplied
        tf_dict = {self.eps_lb_tf: self.eps_lb, self.x_lb_tf: self.x_lb,
                   self.eps_ub_tf: self.eps_ub, self.x_ub_tf: self.x_ub,
                   self.eps_f_tf: self.eps_f, self.x_f_tf: self.x_f}

        # Optimize
        start_time = time.time()
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()
        self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)

    def predict(self, eps, x):
        # Based on optimized NNs, predict solution and residual by replacing
        # placeholders eps_f_tf and x_f_tf with supplied values of eps and x
        u = self.sess.run(self.u_pred, {self.eps_f_tf: eps, self.x_f_tf: x})
        f1 = self.sess.run(self.f1_pred, {self.eps_f_tf: eps, self.x_f_tf: x})
        f2 = self.sess.run(self.f2_pred, {self.eps_f_tf: eps, self.x_f_tf: x})
        return u, f1, f2

###############################################################################
################################ Main Function ################################
###############################################################################

if __name__ == "__main__":
    start = time.time()
    
    layers = [2] + [20]*4 + [1]
    lamb = 0.5
    epslb = 10**-5
    epsub = 10**-3
    ncolx = 64
    for ncoleps in [128]:

        
        lb = np.array([epslb, 0.0]) # lower bound [epsilon, x]
        ub = np.array([epsub, 1.0]) # upper bound [epsilon, x]
    
        ######
        # Load collocation points for epsilon and x.
        # These are the points where the loss function is minimized for training
        # the neural network.
        ######
    
        eps = np.linspace(epslb,epsub,ncoleps).reshape((ncoleps,1))
        x = np.linspace(0,1,num=ncolx).reshape((ncolx,1))
        dx = x[1,0]-x[0,0]
        # all possible combinations of input parameters
        Eps, X = np.meshgrid(eps,x)
        # flatten parameter data and exact solution
        eps_star = Eps.flatten()[:,None]
        x_star = X.flatten()[:,None]
    
        # stack parameters together
        P_star = np.hstack((eps_star, x_star))
        
        model = AdvDiff(eps, P_star, dx, layers, lb, ub, lamb)
    

        model.train(N_iter=0)
        # use the solver to predict the solution and the residual at all data points
        xas = 1000
        eps = np.ones((xas,1))*10**-5
        x = np.linspace(0,1,num=xas).reshape((xas,1))  # points for plotting
        pred, f1, f2 = model.predict(eps, x)
        u_star = np.zeros((xas,1))
        u_x_star = np.zeros((xas,1))
        eps = eps[0,0]
        axes = plt.gca()
        for i in range(0,xas):
            epss = Decimal(eps)
            xx = Decimal(x[i,0])
            u_star[i,0] = (1-Decimal(np.exp(xx/epss)))/(1-Decimal(np.exp(1/epss)))
            u_x_star[i,0] = Decimal(np.exp(xx/epss))/epss/(Decimal(np.exp(1/epss))-1)
        u_pred = pred[:,0].reshape((1000,1))
        u_x_pred = -pred[:,1].reshape((1000,1))/eps
        plt.plot(x,u_pred,'b', label='u predict')
        plt.plot(x,u_star, 'r--', label='u exact')
        plt.legend()
        plt.title('eps = ' + str(eps))
        plt.xlabel("x")
        plt.ylabel("u(x)")
        axes.set_xlim([0,1.1])
        
        plt.figure()
        xl=950
        plt.plot(x[xl:xas],u_pred[xl:xas],'b', label='u prediction')
        plt.plot(x[xl:xas],u_star[xl:xas],'r--', label='u exact')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title('eps = ' + str(eps))
        #plt.savefig('u eps = ' + str(eps) + ' colp ' + str(num_colp)+'.png')
        #plt.close()   
        
        # error at all data points
        error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
        error_ux = np.linalg.norm(u_x_star-u_x_pred,2)/np.linalg.norm(u_x_star,2)
        print('Error u: %e' % (error_u))
        print('Error dudx: %e' % (error_ux))
    
        end = time.time()
        print("Running time: " + str(end - start))    
        winsound.Beep(440, 1000)        
