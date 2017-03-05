# Deep Residual Learning for Image Recognition

The network depth, which is the number of stacked layers, is a common problem in Deep Convolutional Neural Network (CNN)
because if the depth increases, the accuracy gets saturated (degradation problem). So, this paper addresses this problem by
introducing a deep residual learning framework. 

It considers a mapping H(x) fitted by different stacked layers and from this mapping, it approximates the residual function by
doing F(x) = H(x) - x, where x is the input vector of the corresponding layer. Once the formulation of residual learning is
redefined, the goal is to look for the optimum mappings a to deal with the problem of degradation.

In this paper cases with F(x) function with two or three layers, more layers is possible.

Two network architectures are tested:

Plain network is inspired in VGG, it uses 3x3 convolutional layers and the number of filters is related with the feature map
size.

Residual network is very similar to Plain Network but with the difference that it inserts shortcut connections.

In tests, the degradation problem are observed in plain network. But in Residual Networks better results are obtained, the
training error is reduced and the degradation problem is mitigated.

Once this taken into account, the authors CNN are presented, this CNN is based in a bottleneck architecture. It uses a
residual functions F(x) with a stack of three layers. The three layers are 1×1, 3×3, and 1×1 convolutions. 

In this part of the paper different ResNet are defined with the difference that the depth is increased (starts with 50 layers
until 152) and each 2-layer block is substituted with this 3-layer bottleneck block.

Different depths are tested (50-layer, 101-layer and 152-layer) and the results are compared with the state-of-the-art
architectures, and the results are improved, the best training/validation error rate is obtained with 152 layers.re improved, the best training/validation error rate are obtained with 152 layers.
