# Segnet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

## Introduction & State of the art
Semantic Segmentation has a wide range of applications, from scene understanding to autonomous driving. With the deep learning techniques have accomplished big performances in handwritten recognition, speech and object detection. The motivation under Segnet is the need of mapping the low resolution features to input resolution for pixel-wise classification. 

Segnet is primarly motivated by road scene understanding applications where there is a necessity of modelling the appearance (road, etc), shape (car, pedestrian) and understand the spatial-relationship between different classes. It is important to delineate objects based on its shape and keep the boundary information. 

Segnet is topollogically identical to VGG16 but without the fully connected layers, making Segnet smaller and easier to train. Its key correspond to the decoder network that consists in hiercarchy of decoders one corresponding to each encoder. The appropiates decoders use max-pooling indices received from the input feature maps, which improves the boundary delineation, reduces the number of parameters and it allows the incorporation to any encoder-decoder architecture. 

Before deep learning techniques, the best performances among semantic pixel-wised relied on hand engineered features classifying pixels independently. More recent approaches predict the labels for all the pixels in a patch instead of only the centered pixel, giving better results and performances. 

New architectures particularly designed for segmentation have advanced the state-of-the-art by learning how to decode and or map low resolution representations in all of these architectures in the VGG16 classification network. 

## Architecture
Segnet has an encoder network and a corresponding decoder network with a final pixelwise classification layer. The encoder network is formed by 13 convolutional layers (same as VGG16), therefore, it can use the weigths from pre-trained classification applications for training. The fully connected layers are discarded in order to retain higher resolution maps. 

Each enconder performs a convolution with a filter bank to produce a set of features maps. Then the ReLu is applied with a maxpooling 2x2 window with stride 2 and the resulting output is sub-sampled by 2. Max-Pooling is used to achieve translation invariance over small spatial shifts. The increasingly loss image representation is not beneficial for segmentation as it blurrs boundary delineation. That is why the boundary information is capture and stored in the encoder feature maps before sub-sampling. In Segnet, there is an improving inside the boundary information caputre as it stores only the max-pooling indices (locations of the maximum feature value in each pooling window).

The decoders decode the feature maps using the memorized max-pooling indices from the corresponding feature map. These feature maps are then convolved with filter bank to produce dense feature maps with and applied batch normalization. The output of the softmax is a K channel image of probabilities where K is the number of classes. The predicted segmentation correspond to the class with maximum probability at each pixel.

