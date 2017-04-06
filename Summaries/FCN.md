
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
