import random
import numpy as np
import math

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return values*(1-values)

def tanh_derivative(values): 
    return 1. - values ** 2
# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args): 
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class GRUParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        # weight matrices
        self.wz = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wr = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bz = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.br = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 

        # diffs (derivative of loss function w.r.t. all parameters)
        self.wz_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wr_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bz_diff = np.zeros(mem_cell_ct) 
        self.br_diff = np.zeros(mem_cell_ct) 
        self.bg_diff = np.zeros(mem_cell_ct) 
    def apply_diff(self, lr = 1):
        self.wz -= lr * self.wz_diff
        self.wr -= lr * self.wr_diff
        self.wg -= lr * self.wg_diff
        self.bz -= lr * self.bz_diff
        self.br -= lr * self.br_diff
        self.bg -= lr * self.bg_diff
        # reset diffs to zero
        self.wz_diff = np.zeros_like(self.wz)
        self.wr_diff = np.zeros_like(self.wr) 
        self.wg_diff = np.zeros_like(self.wg) 
        self.bz_diff = np.zeros_like(self.bz)
        self.br_diff = np.zeros_like(self.br) 
        self.bg_diff = np.zeros_like(self.bg) 
    
class GRUState:
    def __init__(self, mem_cell_ct, x_dim):
        self.z = np.zeros(mem_cell_ct)
        self.r = np.zeros(mem_cell_ct)
        self.g = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)

class GRUNode:

    def __init__(self, gru_param, gru_state):
        # store reference to parameters and to activations
        self.state = gru_state
        self.param = gru_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None
        self.xrh = None
    def bottom_data_is(self, x, h_prev = None):
        # if this is the first lstm node in the network
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))
        self.state.z = sigmoid(np.dot(self.param.wz, xc) + self.param.bz)
        self.state.r = sigmoid(np.dot(self.param.wr, xc) + self.param.br)
        
        # concatenate x(t) and r(t) * h(t-1)
        xrh = np.hstack((x, self.state.r * h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xrh) + self.param.bg)
        self.state.h = (np.ones_like(self.state.z) - self.state.z) * self.state.g + self.state.z * self.h_prev
        self.xc = xc
        self.xrh = xrh
    def top_diff_is(self, top_diff_h):
        dz = (self.h_prev - self.state.g) * top_diff_h
        dg = (np.ones_like(self.state.z) - self.state.z) * top_diff_h
        
        # diffs w.r.t. vector inside sigma / tanh function
        dz_input = sigmoid_derivative(self.state.z) * dz 
        dg_input = tanh_derivative(self.state.g) * dg
        
        # dr
        dr = dg_input * (self.param.wg.T[self.param.x_dim:].dot(self.h_prev) + self.param.bg)
        #dr_input
        dr_input = sigmoid_derivative(self.state.r) * dr
        
        # diffs w.r.t. inputs
        self.param.wz_diff += np.outer(dz_input, self.xc)
        self.param.wr_diff += np.outer(dr_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xrh)
        self.param.bz_diff += dz_input
        self.param.br_diff += dr_input       
        self.param.bg_diff += dg_input       

        # compute bottom diff
        dxc = np.zeros_like(self.state.z)
        dxc += np.dot(self.param.wz.T[self.param.x_dim:], dz_input)
        dxc += np.dot(self.param.wg.T[self.param.x_dim:], dg_input * self.state.r)
        dxc += np.dot(self.param.wr.T[self.param.x_dim:], dr_input)
        
        # save bottom diffs
        self.state.bottom_diff_h = dxc + top_diff_h * self.state.z
    
        
class GRUNetwork():

    def __init__(self, gru_param):
        self.gru_param = gru_param
        self.gru_node_list = []
        # input sequence
        self.x_list = []
    
    def y_list_is(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence 
        with corresponding loss layer. 
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.gru_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.gru_node_list[idx].state.h, y_list[idx])
        self.gru_node_list[idx].top_diff_is(diff_h)
        idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h

        while idx >= 0:
            loss += loss_layer.loss(self.gru_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.bottom_diff(self.gru_node_list[idx].state.h, y_list[idx])
            diff_h += self.gru_node_list[idx + 1].state.bottom_diff_h
            self.gru_node_list[idx].top_diff_is(diff_h)
            idx -= 1 
        return loss
    def x_list_clear(self):
        self.x_list = []
    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.gru_node_list):
            # need to add new gru node, create new state mem
            gru_state = GRUState(self.gru_param.mem_cell_ct, self.gru_param.x_dim)
            self.gru_node_list.append(GRUNode(self.gru_param, gru_state))
        
        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.gru_node_list[idx].bottom_data_is(x)
        else:
            h_prev = self.gru_node_list[idx - 1].state.h
            self.gru_node_list[idx].bottom_data_is(x, h_prev)


        




    
    
