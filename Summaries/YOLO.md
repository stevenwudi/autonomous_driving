# You Only Look Once: Unified, Real-Time Object Detection

YOLO is an approach to object detection as a regression problem, from image pixels to bounding boxes and class probabilites, and its design enables end-to-end learning and real-time speed. 
The methodology is simple:
1) Resizing the image. The system divides the input image into a SxS grid. If the center of an object falls into a grid cell, that grid cell is reponsible for detecting that object.
2) Running a single Convolutional Neural Network. Each grid cell predicts B bounding boxes (x,y,w,h) and confidence score for those boxes. Each grid cell also predicts C conditional class probabilities. The network is inspired by the GoogLeNet model for image classification and it has 24 convolutional layers followed by 2 fully connected layers. It uses 1x1 reduction layers followed by 3x3 convolutional layers. It also uses a leaky rectified linear activation function for the final layer and all other layers.
3) Thresholding the resulting detections by the model's confidence (non-max supression).

Fast version of YOLO designed to push the boundaries of fast object detection. It uses a neural network with 9 convolutional network and fewer filters in those layers. The fast YOLO is the fastest detector on record for PASCAL VOC detection.

YOLO trains and tests on full images and even it is fast (base network runs at 45 frames per second and fast version 150 fps). YOLO is trained on a loss function that directly corresponds to detection performance and the entire model is trained jontly. In fact, it achieves more than twice the mean average precision of other real-time systems and makes less than half the number of background error compared to Fast R-CNN. For this reason, YOLO is the state-of-the-art in real-time object detection.
