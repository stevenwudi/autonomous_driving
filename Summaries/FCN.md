
# Fully Convolutional Networks for Semantic Segmentation

Convolutional networks improve classifcation for whole-image and also make progress on local tasks with structured output as bounding box object detection, part and key-point prediction and local correspondence. Each pixel is labeled with the class of its enclosing object or region.

It is showed that a convolutional network (FCN) trained end-to-end, pixels-to-pixels on semantic segmentation exceed the state-of-the-art without further machinery. This method is efficient, transfering recent succeses in classification to dense prediction by reinterpreting classification nets as fully convolutional and fine-tuning from learned representations. 

## Related Work
First of all there is a transfer of knowledge between fully convolutional networks in order to extend a convnet to arbitrary-sized inputs first appeared in Matan et al. 
Fully convolutional computation has also been exploited in recent works, with features like sliding window, semantic segmentation, and image restoration. 

## Fully Convolutional Layers
Each layer of data in a convnet is a 3D array where, for example, the first one is the image. Locations in higher layers correspond to the locations in the image they are path-connected to (receptive fields).

Their components (convolution, pooling and activation functions) operate on local input regions and depend only on relative spatial coordinates.

While a general deep net computes a general nonlinear function, a net with only layers of convolution, max-pooling and elementwise nonlinearity, computes a nonlinear filter which we call a deep filter or fully convolutional network. FCN operates on an input of any size and produces an output of corresponding (possibly resampled) spatial dimensions.

### Adapting classifier for dense prediction
Recognition nets (Alexnet, etc.) take fixed-size inputs and produce non-spatial outputs. The fully connected layers of these nets have fixed dimensions and throw away spatial coordinates.

The spatial output maps of these convolutionalized models make them a natural choice of dense problems like semantic segmentation. With groundtruth available, both the forward and backward passes are straightforward and both take advantage of inherent computation. 

### Shift-and-stich & Upsampling
Dense predictions can be obtained from coarse outputs by stitching together output from shifted versions of the input. This transformation increases the cost by a factor of f square (f being the downsampling factor).

Decreasing subsampling withing a net is a traedoff, the filters see the finer information but have smaller receptive fields and take longer to compute.

Another way to connect coarse outputs to dense pixels is interpolation. Upsampling with factor f is convolution with a a fractional input stride of 1/f. The deconvolutional filter in such layer need not be fixed  but can be learned. A stack of deconvolutional layers and activation functions can even learn a nonlinear upsampling. 

Finally, in stochastic optimization the gradient computation is driven by the training distribution.

## Segmentation Architecture
ILSVRC classifiers into FCNs and augment them for dense prediction in-network upsampling and a pixelwise loss. The network is trianed for segmentation by fine-tuning, with a per-pixel multinomial logistic loss and validate with the standard metric of mean pixel intersection over union with the mean taken over all classes. Training ignores pixels that are masked out in the groundtruth.

The architectures chosen are the VGG16, Alexnet and GoogleNet. Each net is decapitated by discarding the final classifier layer and converting all fully connected layers to convolutions. 

Fine-tuning from classification to segmentation gave reasonable predictions of each net. The new fully convolutional (FCN) for segmentation that combines layers of the features hierarchy and refines the spatial precision of the output. 

The output stride is divided by predicting from a 16 pixel stride layer. Adding a 1x1 convolution layer on top of layer pool14. Decreasing the stride of pooling layers is the most straightforward way to obtain finer predictions. Another way to obtain finer predictions is to use the shift-and-stich trick.

## Experimental Framework
The Optimization is made by the training with the SGD and momentum. The minibatch size is 20 images and the momentum chosed is 0.9 and the weight decay is 0.00005 or 0.0002.

All the layers are fine-tuned by back-propagation through the hole net. Just fine-tuning the output classifier alone yields 70%. Training from scratch is not feasible considering the time required to learn the base classification nets. 

The PASCAL VOC 2011 is the training data used as it has 1112 images. Full convolutional training can balance classes by weighting or sampling the loss and the scores are upsampled  to the input dimensions by deconvolution layers.

## Results & Conclusions




