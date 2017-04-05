
# Fully Convolutional Networks for Semantic Segmentation

Convolutional networks improve classifcation for whole-image and also make progress on local tasks with structured output as bounding box object detection, part and key-point prediction and local correspondence. Each pixel is labeled with the class of its enclosing object or region.

Each layer of data in a convnet is a 3D array where, for example, the first one is the image. Locations in higher layers correspond to the locations in the image they are path-connected to (receptive fields).

Their components (convolution, pooling and activation functions) operate on local input regions and depend only on relative spatial coordinates.

