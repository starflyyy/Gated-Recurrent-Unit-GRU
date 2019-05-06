# Gated-Recurrent-Unit-GRU-
An implementation of Gated Recurrent Unit
## 网络上介绍GRU的资料非常多，但是大多缺少实现或者是直接调用现成的机器学习库。作为一个初学者，我希望能够探索GRU内在的数学原理，在前人提供的众多资料的协助下，完成了一个GRU的手工搭建。其中参考到的资料有：
#### [一个LSTM的实现]https://github.com/nicodjimenez/lstm
---
#### [GRU的数学公式的推导]https://blog.csdn.net/oBrightLamp/article/details/85109589
---
#### [如何理解LSTM]http://colah.github.io/posts/2015-08-Understanding-LSTMs/
---
#### [GRU的理解]https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
## 一些更加详细的介绍
#### 我使用的模型
<img src=http://gwjyhs.com/t6/702/1557153496x2728278877.png />
#### 模型公式：
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large&space;z_t&space;=&space;sigmoid(W_z&space;\cdot&space;[x_t,&space;h_{t&space;-&space;1}]&space;&plus;&space;b_z)" title="\large z_t = sigmoid(W_z \cdot [x_t, h_{t - 1}] + b_z)" />
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large&space;g_t&space;=&space;tanh(W_g&space;\cdot&space;[x_t,&space;r_t&space;\odot&space;h_{t&space;-&space;1}]&space;&plus;&space;b_g)" title="\large g_t = tanh(W_g \cdot [x_t, r_t \odot h_{t - 1}] + b_g)" />
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large&space;h_t&space;=&space;z_t&space;\odot&space;h_{t-1}&space;&plus;&space;(1&space;-&space;z_t)&space;\odot&space;g_t" title="\large h_t = z_t \odot h_{t-1} + (1 - z_t) \odot g_t" />
其中<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;a\odot&space;b" title="a\odot b" />是Hadamard积(就是对应元素相乘，在numpy直接用 "*" 实现)
#### 求导公式：
记：
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large&space;A_z&space;=&space;W_z&space;\cdot&space;[x_t,&space;h_{t&space;-&space;1}]&space;&plus;&space;b_z" title="\large A_z = W_z \cdot [x_t, h_{t - 1}] + b_z" />
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large&space;A_r&space;=&space;W_r&space;\cdot&space;[x_t,&space;h_{t&space;-&space;1}]&space;&plus;&space;b_r" title="\large A_r = W_r \cdot [x_t, h_{t - 1}] + b_r" />
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large&space;A_g&space;=&space;W_g&space;\cdot&space;[x_t,&space;r_t&space;\odot&space;h_{t&space;-&space;1}]&space;&plus;&space;b_g" title="\large A_g = W_g \cdot [x_t, r_t \odot h_{t - 1}] + b_g" />
---
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large\delta&space;z_t&space;=&space;\delta&space;h_t&space;\odot&space;(h_{t-1}&space;-&space;g_t)" title="\large\delta z_t = \delta h_t \odot (h_{t-1} - g_t)" />
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large\delta&space;g_t&space;=&space;\delta&space;h_t&space;\odot&space;(1&space;-&space;z_t)" title="\large\delta g_t = \delta h_t \odot (1 - z_t)" />
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large\delta&space;A_z&space;=&space;\delta&space;z_t&space;\odot&space;z_t&space;\odot&space;(1&space;-&space;z_t)" title="\large\delta A_z = \delta z_t \odot z_t \odot (1 - z_t)" />
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large\delta&space;A_g&space;=&space;\delta&space;g_t&space;\odot&space;(1&space;-&space;g_t)^2" title="\large\delta A_g = \delta g_t \odot (1 - g_t)^2" />
---
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large\delta&space;r_t&space;=&space;\delta&space;A_g&space;\odot&space;(W_{gh}^T&space;\cdot&space;h_{t-1}&space;&plus;&space;b_g)" title="\large\delta r_t = \delta A_g \odot (W_{gh} \cdot h_{t-1} + b_g)" />
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large\delta&space;A_r&space;=&space;\delta&space;r_t&space;\odot&space;r_t&space;\odot&space;(1&space;-&space;r_t)" title="\large\delta A_r = \delta r_t \odot r_t \odot (1 - r_t)" />
---
<img src="https://latex.codecogs.com/png.latex?\fn_cs&space;\large\delta&space;h_{t&space;-&space;1}&space;=&space;\delta&space;h_t&space;\odot&space;z_t&space;&plus;&space;W_{rh}&space;\cdot&space;\delta&space;A_r&space;&plus;&space;W_{zh}&space;\cdot&space;\delta&space;A_z&space;&plus;&space;W_{gh}&space;\cdot&space;(\delta&space;A_g&space;\odot&space;r_t)" title="\large\delta h_{t - 1} = \delta h_t \odot z_t + W_{rh} \cdot \delta A_r + W_{zh} \cdot \delta A_z + W_{gh} \cdot (\delta A_g \odot r_t)" />
注：上面出现的W_{zh}的意思是取W_{z} 的后h列，也就是h对应的列，
具体实现见代码。
对参数的求导的具体实现参见代码。

## 强烈建议首先研究清楚 numpy.dot() 和numpy.outer()的用法
