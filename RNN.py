import numpy as np

class RNN:

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
        
    def __init__(self):
        # 10 parameters per input, morphed to 100 from input layer, gives one value as output
        self.NO_OF_INPUTS = 20
        self.INPUT_SIZE = 10
        self.HIDDEN_SIZE = 100
        self.OUTPUT_SIZE = 1
        # Input and Output values X and Y assigned at random
        self.X=np.random.randint(0,2,(self.NO_OF_INPUTS,self.INPUT_SIZE))
        self.Y=np.random.randint(0,2,(self.NO_OF_INPUTS,self.OUTPUT_SIZE))
        # RNN Forward Prop Values
        self.iv = np.random.uniform(-0.01,0.01, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.RNN_hv = np.random.uniform(-0.01,0.01, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.ov = np.random.uniform(-0.01,0.01, size=(self.NO_OF_INPUTS,self.OUTPUT_SIZE))
        # RNN Weights
        self.W_ih = np.random.uniform(-0.01,0.01, size=(self.INPUT_SIZE,self.HIDDEN_SIZE))
        self.RNN_W_hh = np.random.uniform(-0.01,0.01, size=(self.HIDDEN_SIZE,self.HIDDEN_SIZE))
        self.W_ho = np.random.uniform(-0.01,0.01, size=(self.HIDDEN_SIZE,self.OUTPUT_SIZE))
        # RNN Back Prop Values
        self.RNN_dov = np.random.uniform(-0.01,0.01, size=(self.NO_OF_INPUTS,self.OUTPUT_SIZE))
        self.RNN_dhv = np.random.uniform(-0.01,0.01, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.RNN_dhvprev = self.RNN_dhv
        self.RNN_div = np.random.uniform(-0.01,0.01, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.RNN_hvprev=self.RNN_hv
        '''
        # LSTM Forward Prop Values
        self.LSTM_iv = np.random.uniform(0,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_fv = np.random.uniform(0,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_candv = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_cv = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_ov = np.random.uniform(0,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_hv = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_hvprev = self.LSTM_hv
        self.LSTM_cvprev = self.LSTM_cv
        self.vector = np.concatenate((self.LSTM_hvprev, self.iv), axis=1)
        # LSTM Weights
        self.LSTM_W_i = np.random.uniform(0,1, size=(self.LSTM_iv.shape[1] + self.LSTM_hvprev.shape[1], self.HIDDEN_SIZE))
        self.LSTM_W_f = np.random.uniform(0,1, size=(self.LSTM_iv.shape[1] + self.LSTM_hvprev.shape[1], self.HIDDEN_SIZE))
        self.LSTM_W_cand = np.random.uniform(-1,1, size=(self.LSTM_iv.shape[1] + self.LSTM_hvprev.shape[1], self.HIDDEN_SIZE))
        self.LSTM_W_o = np.random.uniform(0,0.01, size=(self.LSTM_iv.shape[1] + self.LSTM_hvprev.shape[1], self.HIDDEN_SIZE))
        # LSTM Back Prop Values
        self.LSTM_dov = np.random.uniform(0,1, size=(self.NO_OF_INPUTS,self.OUTPUT_SIZE))
        self.LSTM_dhv = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_dhvprev = self.LSTM_dhv
        self.LSTM_div = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_divprev = self.LSTM_div
        self.LSTM_dcandv = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_dcandvprev = self.LSTM_dcandv
        self.LSTM_dfv = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_dfvprev = self.LSTM_dfv
        self.LSTM_dov = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_dovprev = self.LSTM_dov
        self.LSTM_dcv = np.random.uniform(-1,1, size=(self.NO_OF_INPUTS,self.HIDDEN_SIZE))
        self.LSTM_dcvprev = self.LSTM_dcv
        self.LSTM_W_i_future = self.LSTM_W_i
        self.LSTM_W_f_future = self.LSTM_W_f
        self.LSTM_W_o_future = self.LSTM_W_o
        self.LSTM_W_cand_future = self.LSTM_W_cand

    def LSTM_forward(self):
        self.LSTM_hvprev = self.LSTM_hv
        self.LSTM_cvprev = self.LSTM_cv
        self.iv = self.X.dot(self.W_ih) + 0.0001
        self.vector = np.concatenate((self.LSTM_hvprev, self.iv), axis=1)
        self.LSTM_iv = self.vector.dot(self.LSTM_W_i)+0.0001
        self.LSTM_candv = np.tanh(self.vector.dot(self.LSTM_W_cand) + 0.0001)
        self.LSTM_fv = self.sigmoid(self.vector.dot(self.LSTM_W_f) + 0.0001)
        self.LSTM_cv = self.LSTM_cvprev*self.LSTM_fv + self.LSTM_iv*self.LSTM_candv
        self.LSTM_ov = self.sigmoid(self.vector.dot(self.LSTM_W_o) + 0.001)
        self.LSTM_hv = self.LSTM_ov*np.tanh(self.LSTM_cv)
        self.ov = self.LSTM_hv.dot(self.W_ho) + 0.0001
        
    def LSTM_backprop(self):
        self.RNN_dov = self.Y - self.ov
        self.LSTM_dhv = self.RNN_dov.dot(self.W_ho.T) + self.LSTM_divprev.dot(self.LSTM_W_i_future[:100].T) + self.LSTM_dovprev.dot(self.LSTM_W_o_future[:100].T) + self.LSTM_dfvprev.dot(self.LSTM_W_f_future[:100].T) + self.LSTM_dcandvprev.dot(self.LSTM_W_cand_future[:100].T)
        self.LSTM_dcv = self.LSTM_dhv*self.LSTM_ov*(1-np.tanh(self.LSTM_cv)*np.tanh(self.LSTM_cv)) + self.LSTM_dcvprev*self.LSTM_fv
        self.LSTM_dov = self.LSTM_dhv*np.tanh(self.LSTM_cv)
        self.LSTM_div = self.LSTM_dcv*self.LSTM_candv
        self.LSTM_dcandv = self.LSTM_dcv*self.LSTM_iv
        self.LSTM_dfv = self.LSTM_dcv*self.LSTM_cvprev
        self.RNN_div = self.LSTM_div.dot(self.LSTM_W_i[100:200].T) + self.LSTM_dov.dot(self.LSTM_W_o[100:200].T) + self.LSTM_dfv.dot(self.LSTM_W_f[100:200].T) + self.LSTM_dcandv.dot(self.LSTM_W_cand[100:200].T)

        self.W_ho += self.LSTM_hv.T.dot(self.RNN_dov)*0.01
        self.LSTM_W_o += self.vector.T.dot(self.LSTM_dov)*0.01
        self.LSTM_W_f += self.vector.T.dot(self.LSTM_dfv)*0.01
        self.LSTM_W_i += self.vector.T.dot(self.LSTM_div)*0.01
        self.LSTM_W_cand += self.vector.T.dot(self.LSTM_dcandv)*0.01
        self.W_ih += self.X.T.dot(self.RNN_div)*0.01
        
        self.LSTM_divprev = self.LSTM_div
        self.LSTM_dovprev = self.LSTM_dov
        self.LSTM_dfvprev = self.LSTM_dfv
        self.LSTM_dcandvprev = self.LSTM_dcandv
        self.LSTM_dcvprev = self.LSTM_dcv
        self.LSTM_W_i_future = self.LSTM_W_i
        self.LSTM_W_o_future = self.LSTM_W_o
        self.LSTM_W_f_future = self.LSTM_W_f
        self.LSTM_W_cand_future = self.LSTM_W_cand
'''
    def RNN_forward(self):
        self.iv=np.dot(self.X,self.W_ih)
        self.RNN_hv=np.tanh(self.iv + np.dot(self.RNN_hvprev, self.RNN_W_hh))
        self.ov=np.dot(self.RNN_hv,self.W_ho)
        self.RNN_hvprev=self.RNN_hv

    def RNN_backprop(self):
        self.RNN_dov=self.Y-self.ov
        self.RNN_dhv=(self.RNN_dov.dot(self.W_ho.T)+self.RNN_dhvprev.dot(self.RNN_W_hh.T))*(1-self.RNN_hv*self.RNN_hv)
        self.RNN_div=(self.RNN_dov.dot(self.W_ho.T))*(1-self.RNN_hv*self.RNN_hv)
        
        self.W_ho+=self.RNN_hv.T.dot(self.RNN_dov)*0.01
        self.RNN_W_hh+=self.RNN_hvprev.T.dot(self.RNN_dhv)*0.01
        self.W_ih+=self.X.T.dot(self.RNN_div)*0.01
        
        self.RNN_dhvprev=self.RNN_dhv
        '''GRADIENT CLIPPING
        if(np.sum(np.abs(self.RNN_dov))>100 && self.RNN_dov>0.0001):
            self.RNN_dov=(100/np.sum(np.abs(self.RNN_dov)))*self.RNN_dov
        if(np.sum(np.abs(self.RNN_dhv))>100 && self.RNN_dhv>0.0001):
            self.RNN_dhv=(100/np.sum(np.abs(self.RNN_dhv)))*self.RNN_dhv
        if(np.sum(np.abs(self.RNN_div))>100 && self.RNN_div>0.0001):
            self.RNN_div=(100/np.sum(np.abs(self.RNN_div)))*self.RNN_div
        if(np.sum(np.abs(self.RNN_dhvprev))>100 && self.RNN_dhvprev>0.0001):
            self.RNN_dhvprev=(100/np.sum(np.abs(self.RNN_dhvprev)))*self.RNN_dhvprev
        '''


obj = RNN()
'''
for _ in np.arange(10000):
    obj.LSTM_forward()
    obj.LSTM_backprop()
print "INPUT:"
print obj.X
print "-------------"
print "OUTPUT:"
print obj.Y
print "-------------"
print "PREDICTED:"
print obj.ov
print "-------------"
print "ERROR:"
error=np.sum(np.abs(obj.Y-obj.ov))
print error,"%"
'''
for _ in np.arange(100000):
    obj.RNN_forward()
    obj.RNN_backprop()
print "INPUT:"
print obj.X
print "-------------"
print "OUTPUT:"
print obj.Y
print "-------------"
print "PREDICTED:"
print np.abs(np.round(obj.ov))
print "-------------"
print "ERROR:"
error=np.sum(np.abs(obj.Y-obj.ov))
print error
