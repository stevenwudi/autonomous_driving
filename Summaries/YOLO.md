# You Only Look Once: Unified, Real-Time Object Detection

YOLO is an approach to object detection as a regression problem, from image pixels to bounding boxes and class probabilites.
The methodology is simple:
1) Resizing the image. 
2) Running a single convolutional network. It predicts bounding boxes and class probabilites for the boxes.
3) Thresholding the resulting detections by the model's confidence (non-max supression).

YOLO trains and tests on full images but it is fast (base network runs at 45 frames per second and fast version 150 fps). In fact, it achieves more than twice the mean average precision of other real-time systems and makes less than half the number of background error comparted to Fast R-CNN.



