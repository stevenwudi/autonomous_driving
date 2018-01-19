import numpy as np
import os
import matplotlib
from PIL import Image
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches


def main():
    data_array = np.load(r'C:\Users\steve\Desktop\cvpr_figure\Kalman_filter\prepared_data_shuffle.npy')
    test_data_array = data_array[2]
    data_mean = data_array[3]
    data_std = data_array[4]
    test_data_array = (test_data_array * data_std) + data_mean

    item_data = test_data_array[693]
    img_path =r'C:\Users\steve\Desktop\cvpr_figure\Final_VISUAL'
    img_list = sorted(os.listdir(img_path))

    fig = plt.figure(1)

    tracking_figure_axes = fig.add_subplot(111, aspect='equal')
    for i,item in enumerate(item_data):

        rect = item[:4]
        print(rect)
        im = Image.open(os.path.join(img_path, img_list[i]))
        tracking_figure_axes.imshow(im)
        # Create a Rectangle patch
        tracking_rect = Rectangle(
            xy=(rect[1]-rect[3]/2., rect[0]-rect[2]/2),
            width=rect[3],
            height=rect[2],
            facecolor='none',
            edgecolor='r',
        )
        tracking_figure_axes.add_patch(tracking_rect)
        plt.waitforbuttonpress(1)
    fig.clear()



# Entry point of the script
if __name__ == "__main__":
    main()