# Segnet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

Semantic Segmentation has a wide range of applications, from scene understanding to autonomous driving. With the deep learning techniques have accomplished big performances in handwritten recognition, speech and object detection. The motivation under Segnet is the need of mapping the low resolution features to input resolution for pixel-wise classification. 

Segnet is primarly motivated by road scene understanding applications where there is a necessity of modelling the appearance (road, etc), shape (car, pedestrian) and understand the spatial-relationship between different classes. It is important to delineate objects based on its shape and keep the boundary information. 

Segnet is topollogically identical to VGG16 but without the fully connected layers, making Segnet smaller and easier to train. Its key correspond to the decoder network that consists in hiercarchy of decoders one corresponding to each encoder. The appropiates decoders use max-pooling indices received from the input feature maps, which improves the boundary delineation, reduces the number of parameters and it allows the incorporation to any encoder-decoder architecture. 

