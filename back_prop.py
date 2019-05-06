import os
import math
import struct
import numpy as np   
from keras.utils import to_categorical
#换softmax函数##
#定义激活函数   
#隐藏层没有进行激活
def softmax (X): 
    X_exp = np.exp(X)
    sigma = np.sum(X_exp)
    X = X_exp / sigma
    return X

#定义神经网络   
class NN:   
    def __init__(self, ni, nh, no):       

        #输入层加入一个bias结点   
        self.ni = ni + 1   
        self.nh = nh   
        self.no = no   

        self.ai = np.array([1.0] * self.ni).reshape((-1, 1))   
        self.ah = np.array([1.0] * self.nh).reshape((-1, 1))   
        self.ah_not_activate = np.array([1.0] * self.nh).reshape((-1, 1))   
        self.ao = np.array([1.0] * self.no).reshape((-1, 1))   
        self.ao_not_activate = np.array([1.0] * self.no).reshape((-1, 1))   
        self.wi = np.random.rand(self.nh, self.ni) * 2 - 1   
        self.wo = np.random.rand(self.no, self.nh) * 2 - 1   

    def forward(self, inputs):       

        if len(inputs) != self.ni-1:   
            raise ValueError('error')   

        #将inputs的值赋给ai 注意第一个结点的值恒为1   
        self.ai[1: ] = inputs   

        #计算隐藏层列向量   
        self.ah_not_activate = self.wi.dot(self.ai)
        self.ah_not_activate = (self.ah_not_activate - self.ah_not_activate.mean())/self.ah_not_activate.std()

        self.ah = self.ah_not_activate
        #计算输出层结点   
        self.ao_not_activate = self.wo.dot(self.ah)
        self.ao_not_activate = (self.ao_not_activate - self.ao_not_activate.mean()) / self.ao_not_activate.std()

        self.ao = softmax(self.ao_not_activate)   
        
        return self.ao   
    def backpropagate(self,targets,learning_rate):   
        #计算输出层的误差   
        #经过符合函数求导得到输出层的delta
        output_deltas = (self.ao - targets) 
        #计算隐藏层误差   
        hidden_deltas = self.wo.T.dot(output_deltas)   
        #更新输出层权重   
        self.wo = self.wo - learning_rate * (output_deltas.dot(self.ah.T))      
        #更新隐藏层权重   
        self.wi = self.wi - learning_rate * (hidden_deltas.dot(self.ai.T))   
        #误差函数应该选择交叉熵
        error = targets * (np.log(self.ao)) * (-1)
        return error.sum()   
    def train(self, X_train, y_train, iterations = 10, learning_rate = 0.1):   
        for i in range(iterations):   
            error = 0.0   
            for j in range(100):   
                inputs = np.array(X_train[j]).reshape(-1,1)   
                targets = np.array(y_train[j]).reshape(-1,1)      
                self.forward(inputs)   
                error = error + self.backpropagate(targets,learning_rate)   
            if (i % 2 == 0):   
                error = error / 1000
                print ("error = %.5f"% error)   

    def weights(self):   
        print("输入层权重")   
        print(self.wi)   
        print("输出层权重")   
        print(self.wo)   

    def test(self,X_train, y_train): 
        cnt = 0
        n = 100
        for j in range(n):   
            t = np.array(X_train[j]).reshape(-1, 1)      
            x = self.forward(t)
            a = np.where(x == np.max(x))
            if (a[0][0] == y_train[j]):
                cnt = cnt + 1
        print("准确率为：", cnt / n)


def load_mnist(path, kind='train'):      
    labels_path = os.path.join(path,   
                                '%s-labels.idx1-ubyte'   
                                % kind)   
    images_path = os.path.join(path,    
                                '%s-images.idx3-ubyte'   
                                % kind)   
    with open(labels_path, 'rb') as lbpath:   
        magic, n = struct.unpack('>II',   
                                  lbpath.read(8))   
        labels = np.fromfile(lbpath,    
                              dtype=np.uint8)   

    with open(images_path, 'rb') as imgpath:   
        magic, num, row, cols = struct.unpack('>IIII',    
                                                imgpath.read(16))   
        images = np.fromfile(imgpath,    
                              dtype=np.uint8).reshape(len(labels), 784)   

    return images, labels   #定义神经网络   

def demo():   
    #读入数据   
    X_train, y_train = load_mnist(r"C:/Users/12592/Python/study_myself/mnist")   
    #对数据进行标准化处理   
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())  
    #将标签转化成one_hot编码，每一行是一个样本的标签   
    one_hots = to_categorical(y_train)   
    n = NN(784, 800, 10)   
    n.train(X_train[:1000, :], one_hots[:1000, :])   
    n.test(X_train[:100, :], y_train[:100])   
    n.weights()   
if __name__ == '__main__':       
    demo() 
