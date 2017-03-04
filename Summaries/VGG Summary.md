# Very Deep Convolutional Networks for Large-Scale Image Recognition
In this paper, the effect of the convolution network depth on its accuracy in large-scale image recognition is studied. Studying from a very small 3x3 architecture to a 16-19 weight layers.

CNN have enjoyed a great succes in large-scale image and video recognition thanks to the large public repositories and the increase of high performance computing systems (GPUs). These improvmenets have been showed in the ImageNet Challenge over the years, that has served as a test for the lasts generations. 
Thanks to the previous improvements, it has been possible to achieve a better performance than the original architecture of Krizhevsky et al (2012), for example, the best submissions for ILSVRC-2013 used smaller receptive window size and smaller strides for the first conv. layer. 
Finally, in this paper they also take into account the effect of depth in the CNN (fixing the parameters and adding more convolutiona layers) using very small (3x3) convolution filters in all the layers. Thanks to this idea, they achieved a more accurate ConvNet architectures  that outperform in ILSVRC state of the art and were able to work in different image recognition datasets.

All the configurations explained in the paper are inspired in Ciresan et al (2011) and Krizhevsky et al (2012).
In the architecture, the input of ConvNets is a fixed 224x224 RGB image with a preprocessing of substracting the mean RGB value for each pixel in the whole dataset. The filters used are a 3x3, with a stride of 1, spatial padding of 1 pixel and five max-pooling layers over a 2x2 pixel windows with stride 2 that follow some conv.layers.
At the end of the architecture, we can find three Fully-Connected layers (FC), where the first two have 4096 channels and the last one 1000 channels (1 for each class in the database). Also there is one last layer which is the softmax layer. 
The hidden layers have the rectification ReLU in order to apply non-linearity and Local Response Normalisation for normalisation (LRN).

The different configurations explained in the paper only vary in the depth, from 11 weight layers in the network A (8conv +3FC) to 19 weigth layers in configuration E (16conv + 3FC). The width of the convolutional layers start in 64 and increase with a factor of 2.

The important change due to the configurations explained in the paper are that small receptive fields (3x3) compared with the large from previous works (7x7) as a stack of 3 small receptive fields gives the same responsa than a larger one with less parameters to be computed. Also, this method allows to introduce more non-linearities that make the decision function more discriminative. In some configurations, a 1x1 conv. layer has been added in order to introduce even more non-linearities without affecting the receptive fields of the conv.layers.
It is possible to see that the number of parameters varies from 133M to 144M from configuration A to configuration E.

The training process follows the krizhevsky et al procedure, optimising the multinomial logistic regression with mini batch gradient descent with momentum. Also, the training was regularised by weigh decay with an L2 penalty and a dropout regularisation for the first two fully connected layers of 0.5.
The initial learning rate was set to 0.01 and decreased by a factor of 10 when the validation set stopped improving (3 times decreased). Finally, a total number of 74 epochs was used.
The initialisation of the weigths was a very important matter, solved with a random initialisation with weigths sampled from a normal distribution with zero mean and 0.01 variance. 
In order to compute the training, it was also necessary to crop the images by the same size. Two approaches were considered, where the crop size S was fixed in two different values (256 & 384). The second approach was to set S as a multi-scale training, where each image was individually rescaled by randomly sampling S from a certain range (256-512).

At test time, given a trained ConvNet, first of all the image is rescaled to the pre-defined smallest image side (Q), which does not need to be equal to S. The network is applied over the scaled image. Then, the fully-convolutional net is applied to the whole uncropped image, where the result is a class score map with the number of channels equals to the number of classes. Finally, in order to obtain a fixed-size vector of class scores, the class score map is spatially averaged (sum-pooled).
The network applied to the whole uncropped image is less efficient as it requires network re-computation for each crop.

The implementation is derived from the public available C++ Caffe toolbox, containing modifications that allow to perform training and evaluations on multiple GPUs, exploiting data parallelism.

The classification performance is evaluated by two measures, top1 and top5 error.
First of all, the performance of individual ConvNet models ata a single scale with the previous layers configurations is evaluated. The conclusions that can be extracted are that local response normalisation does not improve without any normalisation layers, the classification error decreases with the increased ConvNet depth and that scale jittering at training time between 256-512 leads to significant better results than training images in a fixed scale. 
Being able to reach a top1 error of 25.5% and a top5 error of 8%.

Furthermore, the effect of scale jittering at test time is also evaluated. It consists on running a model over several rescaled versions of a test image and averaging the resulting class posteriors. The conclusions extracted are that the scale jittering at test time leads to better performance compared to evaluating the same model at a single scale, the deepest configurations work better and scale jittering is better than training with a fixed smalles size S. 
Being able to reach a top1 error of 24.8% and a top5 error of 7.5%.

Also, a dense ConvNet evaluation is compared with a multi-crop evaluation, showing that using multiple crops performs slightly better than dense evaluation, and the combination of the two approaches even reaches better results.
Bein able to reach a top1 error of 24.4% and a top5 error of 7.1%.

Moreover, the combination of the outputs of several models by averaging their soft-max class posteriors is also evaluated, improving the performance due to complementarity of the models. The results showed that they were able to reach a 24.7% of error in top1 and a 7.3% of error in top5 (test).

Comparing the final results with the state of the art, they were able to achieve a 2nd place with a 7.3% of error, and improving it after the submission using an ensemble of 2 models, reaching a 6.8% of error. 
Where the classification class winner was GoogLeNet with a 6.7% of error. 

As final conclusions, it can be said that a very deep convolutional network (up to 19 weigth layers) has demonstrated that the representation depth is beneficial for the classification accuracy, reaching and improving the state of the art. 


### References
[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
