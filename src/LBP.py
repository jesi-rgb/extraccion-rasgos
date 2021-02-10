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
        print(windows.shape)
        self.take_adjacents(windows[0])

    def take_adjacents(self, window):
        '''
        Take a window from an image and return the corresponding
        number that this center pixel should have.
        '''
        # print(window)
        # print(window[0])




if __name__ == "__main__":
    import cv2
    PATH_TO_IMG = "mnist_data/train/zero/0.png"

    lbp = LBPDescriptor()
    lbp.compute(cv2.imread(PATH_TO_IMG, 0))