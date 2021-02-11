'''
author: Jesús Enrique Cartas Rascón
repo: https://github.com/jesi-rgb/extraccion-rasgos
'''

import numpy as np, cv2
from numpy.lib.stride_tricks import sliding_window_view
from numpy.core.records import fromarrays

PATH_TO_TRAIN_0 = "mnist_data/train/zero"

class LBPDescriptor():
    def __init__(self, r=3):
        self.radius = r

    
    def compute(self, img):
        # img is a 2-dim np array

        # first take the neighbourhood
        h, w = img.shape

        # Calculate all the possible windows for this image with the shape
        # specified by the radius.
        windows = sliding_window_view(img, window_shape=(self.radius, self.radius))
        
        # Reshape the windows matrix to be a flat array
        # of (radius, radius) little matrices to obtain LBP
        reshaped_windows = np.reshape(windows, newshape=(windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3]))
        
        # Calculate the corresponding value for every 
        # window and return the resulting vector
        values = [self.take_adjacents(window) for window in reshaped_windows]
        
        # Reshape that vector to be a new "LBP IMG"
        img_values = np.reshape(values, newshape=(windows.shape[0], windows.shape[1]))
        
        # Since windows didn't take the borders into account,
        # we pad the array with wrap mode, to get those borders
        # back in a thoughtful way
        lbp_img = np.pad(img_values, ((1, 1), (1, 1)), 'wrap')

        # Calculate the histogram for the lbp img and return it

        return (np.histogram(lbp_img, bins=256, density=False)[0] / (h*w)).astype('float32')

    def take_adjacents(self, window):
        '''
        Take a window from an image and return the corresponding
        number that this center pixel will have, given the local 
        binary pattern.
        '''

        # we take the window shape
        r, c = window.shape

        # calculate the center value, to be compared
        center = window[int(r/2), int(c/2)]

        # now we create a list with the different elements to be
        # chosen from around the center. This concatenation makes 
        # a list of numbers and returns the flattened array 
        # of the neighbourhood, in clock-wise order.
        final_number = np.concatenate([window[0], window[1:,c-1], np.flip(window[r-1,:-1]), np.flip(window[1:-1,0])]).ravel()


        # Finally, we compare all to the center, which gives us the binary
        # array we need. Then, we reverse it and enumerate it, which gives
        # pair (index, value). Since the value is always going to be 0 o 1,
        # we can bitwise shift it. We shift it as many times as this value's
        # index is, resulting in an array with decimal numbers like the example.

        # [0, 0, 0, 1, 1, 1, 0, 0] becomes [0, 0, 4, 8, 16, 0, 0, 0] 
        # (binary read from right to left)
        # This will sum up to 28.

        # Then, sum up everything and return.
        value = sum(v<<i for i, v in enumerate(final_number[::-1] >= center))
        return value



# SAMPLE EXECUTION. THIS CODE WONT BE USED UNLESS THE
# FILE IS EXPLICITLY CALLED AS "python LBP.py".
if __name__ == "__main__":
    import cv2
    import multiprocessing as mp
    import os
    import time
    PATH_TO_IMG = "mnist_data/train/zero/0.png"

    imgs = []
    for filename in os.listdir(PATH_TO_TRAIN_0):
        imgs.append(cv2.imread(os.path.join(PATH_TO_TRAIN_0, filename), 0))

    lbp = LBPDescriptor()

    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    a = pool.map(lbp.compute, imgs)
    pool.close()
    pool.join()
    # print(lbp.compute(cv2.imread(PATH_TO_IMG, 0)))
    print(len(a))
    print("\n\n--- Total execution took {} seconds ---".format(time.time() - start_time))
