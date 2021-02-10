'''
author: Jesús Enrique Cartas Rascón
repo: https://github.com/jesi-rgb/extraccion-rasgos
'''

import numpy as np
from skimage import util
from numpy.lib.stride_tricks import sliding_window_view

class LBPDescriptor():
    def __init__(self):
        self.neighbors = 8
        self.radius = 3

    
    def compute(self, img):
        # img is a 2-dim np array

        # first take the neighbourhood
        h, w = img.shape

        windows = sliding_window_view(img, window_shape=(self.radius, self.radius))
        reshaped = np.reshape(windows, newshape=(windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3]))
        a = map(self.take_adjacents, reshaped)


    def take_adjacents(self, window):
        '''
        Take a window from an image and return the corresponding
        number that this center pixel should have.
        '''
        print(window)
        r, c = window.shape
        center = window[int(r/2), int(c/2)]
        final_number = []
        final_number.extend(window[0])
        final_number.extend(window[1:,c-1])
        final_number.extend(reversed(window[r-1,:-1]))
        final_number.extend(reversed(window[1:-1,0]))
        value = sum(v<<i for i, v in enumerate(final_number[::-1] >= center))
        return value


if __name__ == "__main__":
    import cv2
    PATH_TO_IMG = "mnist_data/train/zero/0.png"

    lbp = LBPDescriptor()
    lbp.compute(cv2.imread(PATH_TO_IMG, 0))