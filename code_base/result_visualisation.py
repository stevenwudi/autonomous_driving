import numpy as np
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from PIL import Image


def main():
    data_array = np.load('/media/samsumg_1tb/synthia/prepared_data_shuffle.npy')
    test_data_array = data_array[2]
    data_mean = data_array[3]
    data_std = data_array[4]
    test_data_array = (test_data_array * data_std) + data_mean

    item_data = test_data_array[693]
    img_path ='/media/samsumg_1tb/synthia/FinalVisual'
    img_list = sorted(os.listdir(img_path))

    fig = plt.figure(1)
    fig.clear()
    tracking_figure_axes = fig.add_subplot(111, aspect='equal')
    #tracking_figure_axes.set_title('Green: detecion; Red: tracking. Image: %s' % img_name)

    for i, item in enumerate(item_data[:10]):
        rect = item[:4]
        im = Image.open(os.path.join(img_path, img_list[i]))
        tracking_figure_axes.imshow(im)
        # Create a Rectangle patch
        tracking_rect = Rectangle(
            xy=(rect[1], rect[0]),
            width=rect[3],
            height=rect[2],
            facecolor='none',
            edgecolor='r',
        )
        tracking_figure_axes.add_patch(tracking_rect)
        #tracking_figure_axes.annotate(str(car_idx), xy=(pos.left(), pos.top() + 20), color='red')
        plt.waitforbuttonpress(0.01)

if __name__ == "__main__":
    main()