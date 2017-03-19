# SSD: Single Shot Multibox Detector

The SSD approach is based on a feed-forward convolutional network for detecting object in images. It discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location achieving higher accuracy. The network produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. 

In the training, it is needed to determine which default boxes correspond to a ground truth detection. The main difference respect to a typical detector is that ground truth information has to be assigned to specific outputs in the fixed set of detector outputs. Once the assignment is determined, the loss function and back propagation are applied end-to-end.

SSD is single-shot detector for multiple categories that is faster than YOLO. The improvement in speed comes from eliminating bounding box proposals and the subsequent pixel or feature resampling stage. The improvement of the performance is due to the use of multi-scale convolutional bounding box attached to multiple feature maps at the top of the network.
