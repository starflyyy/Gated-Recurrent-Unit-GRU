<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
#Gated Recurrent Unit(GRU) Neural Network
##网络上介绍GRU的资料非常多，但是大多缺少实现或者是直接调用现成的机器学习库。作为一个初学者，我希望能够探索GRU内在的数学原理，在前人提供的众多资料的协助下，完成了一个GRU的手工搭建。其中参考到的资料有：
####[一个LSTM的实现]https://github.com/nicodjimenez/lstm
---
####[GRU的数学公式的推导]https://blog.csdn.net/oBrightLamp/article/details/85109589
---
####[如何理解LSTM]http://colah.github.io/posts/2015-08-Understanding-LSTMs/
---
####[GRU的理解]https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
##一些更加详细的介绍
####我所使用的模型：
![blockchain](http://gwjyhs.com/t6/702/1557068782x2918527194.png "gru")
####模型公式：
$$\large z_t = sigmoid(W_z \cdot [x_t, h_{t - 1}] + b_z)$$
$$\large r_t = sigmoid(W_r \cdot [x_t, h_{t - 1}] + b_r)$$
$$\large g_t = tanh(W_g \cdot [x_t, r_t \odot h_{t - 1}] + b_g)$$
$$\large h_t = z_t \odot h_{t-1} + (1 - z_t) \odot g_t$$
其中 \\(a\odot b\\)是Hadamard积(就是对应元素相乘，在numpy直接用 "*" 实现)
####求导公式：
记：
$$\large A_z = W_z \cdot [x_t, h_{t - 1}] + b_z$$
$$\large A_r = W_r \cdot [x_t, h_{t - 1}] + b_r$$
$$\large A_g = W_g \cdot [x_t, r_t \odot h_{t - 1}] + b_g$$
---
$$\large\delta z_t = \delta h_t \odot (h_{t-1} - g_t)$$
$$\large\delta g_t = \delta h_t \odot (1 - z_t)$$
$$\large\delta A_z = \delta z_t \odot z_t \odot (1 - z_t)$$
$$\large\delta A_g = \delta g_t \odot (1 - g_t)^2$$
---
$$\large\delta r_t = \delta A_g \odot (W_{gh}^T \cdot h_{t-1} + b_g)$$
$$\large\delta A_r = \delta r_t \odot r_t \odot (1 - r_t)$$
---
$$\large\delta h_{t - 1} = \delta h_t \odot z_t + W_{rh} \cdot \delta A_r +
    W_{zh} \cdot \delta A_z + W_{gh} \cdot (\delta A_g \odot r_t)$$
注：上面出现的\\(W_{zh}\\)的意思是取\\(W_{z}\\) 的后h列，也就是h对应的列，
具体实现见代码。
对参数的求导的具体实现参见代码。
##*强烈建议首先研究清楚 numpy.dot() 和numpy.outer()的用法*

