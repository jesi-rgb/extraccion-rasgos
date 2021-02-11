'''
author: Jesús Enrique Cartas Rascón
repo: https://github.com/jesi-rgb/extraccion-rasgos
'''

"""
En una primera aproximación, seguir el esquema 
del ejercicio de clase: entrenar un clasificador 
SVM, con HoG como descriptor, y usarlo para 
predecir la clase de una serie de imágenes 
de entrada dadas (en este caso, usando las 
imágenes de dígitos en lugar de las de peatones).

En mi caso, mi DNI acaba en 01, así que aprenderemos
los dígitos 0 y 1.
"""
import cv2
import numpy as np
import os
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as TPool
from sklearn.model_selection import KFold
import pandas as pd
from LBP import LBPDescriptor

import time


PATH_TO_TRAIN_0 = "mnist_data/train/zero"
PATH_TO_TRAIN_1 = "mnist_data/train/one"
PATH_TO_TEST_0 = "mnist_data/test/zero"
PATH_TO_TEST_1 = "mnist_data/test/one"

# parameters for hog
WIN_SIZE = (28, 28)
BLOCK_SIZE = (8, 8)
STEP_SIZE = (2, 2)
CELL_SIZE = (4, 4)
N_BINS = 9


HOG = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, STEP_SIZE, CELL_SIZE, N_BINS)
LBPDESC = LBPDescriptor()

def compute_hog(img):
    '''
    Helper function to parallelize the computing 
    of the HOGs in the images
    '''
    return HOG.compute(img)

def compute_lbp(img):
    '''
    Helper function to parallelize the computing 
    of the HOGs in the images
    '''
    return LBPDESC.compute(img)

def read_img_bw(path):
    '''
    Helper function to parallelize the reading of
    all the images in the dataset
    '''
    return cv2.imread(path, 0)
    

def load_training_data(histogram="hog"):
    """
    Carga las imágenes de entrenamiento y devuelve sus histogramas.

    `histogram`: "hog" o "lbp". Crea los histogramas con el método
    HOG, o el método LBP.
    """ 
    print("Cargando imágenes")

    

    # labels array
    classes = []  

    # all the imgs will lie here  
    img_paths = []

    # get all the paths for 0s and 1s, and append the labels
    for filename in os.listdir(PATH_TO_TRAIN_0):
        # using path.join guarantees compatibility across platforms
        img_paths.append(os.path.join(PATH_TO_TRAIN_0, filename))
        classes.append(0)

    for filename in os.listdir(PATH_TO_TRAIN_1):
        # using path.join guarantees compatibility across platforms
        img_paths.append(os.path.join(PATH_TO_TRAIN_1, filename))
        classes.append(1)

    # create a pool with the number of cores
    pool = mp.Pool(mp.cpu_count())

    # having all the paths, we can read all the imgs
    # in parallel, which is much faster
    images = pool.map(read_img_bw, img_paths)


    print("Calculando histogramas de {} imágenes".format(len(images)))
    # and compute the hogs also in parallel
    predictors = None
    if(histogram == "hog"):
        predictors = pool.map_async(compute_hog, images).get()
    elif histogram == "lbp":
        predictors = pool.map_async(LBPDESC.compute, images).get()
    
    # important: always close the pool
    pool.close()
    pool.join()

    print("Histogramas generados")

    # return all the data collected
    return np.array(predictors), np.array(classes)
    

def train_kernels(training_data, classes, kernel=cv2.ml.SVM_LINEAR):
    """
    Entrena el clasificador

    Parameters:
    training_data (np.array): datos de entrenamiento
    classes (np.array): clases asociadas a los datos de entrenamiento

    Returns:
    cv2.SVM: un clasificador SVM
    """

    print("Training on kernel:", kernel)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(kernel)
    svm.setDegree(2)
    svm.train(training_data, cv2.ml.ROW_SAMPLE, classes)

    return svm


def get_sample_tests(n=10, histogram="hog"):
    img_arrays = []
    classes = []

    for filename in os.listdir(PATH_TO_TEST_0)[:n]:
        filename = os.path.join(PATH_TO_TEST_0, filename)
        img_arrays.append(cv2.imread(filename, 0))
        classes.append(0)

    for filename in os.listdir(PATH_TO_TEST_1)[:n]:
        filename = os.path.join(PATH_TO_TEST_1, filename)
        img_arrays.append(cv2.imread(filename, 0))
        classes.append(1)

 
    pool = mp.Pool(mp.cpu_count())

    hogs = None
    if(histogram == "hog"):
        hogs = pool.map_async(compute_hog, img_arrays).get()
    elif histogram == "lbp":
        hogs = pool.map_async(LBPDESC.compute, img_arrays).get()

    pool.close()
    pool.join()

    print("Tomados histogramas para test")
    return hogs, classes

    
def test(model, predict_imgs_hog):
    return [int(model.predict(hog.reshape(1, -1))[1][0][0]) for hog in predict_imgs_hog]



if __name__ == "__main__":

    hist_mode = "lbp"
    start_time = time.time()

    training_data, classes = load_training_data(histogram=hist_mode)

    kernels = {"linear":cv2.ml.SVM_LINEAR, "polynomial":cv2.ml.SVM_POLY, "rbf":cv2.ml.SVM_RBF, "sigmoid":cv2.ml.SVM_SIGMOID}
    df = pd.DataFrame(columns=kernels.keys())
    kfold = KFold(n_splits=5)

    i = 1
    for train_index, test_index in kfold.split(training_data):
        X_train, X_test = training_data[train_index], training_data[test_index]
        y_train, y_test = classes[train_index], classes[test_index]

        pool = TPool(mp.cpu_count())
        classifiers = [pool.apply(train_kernels, args=(X_train, y_train, kernel)) for kernel in kernels.values()]

        predictions = [pool.apply(test, args=(model, X_test)) for model in classifiers]
        pool.close()
        pool.join()

        
        scores = [[p == l for p, l in zip(pred, y_test)] for pred in predictions]
        accuracies = [np.count_nonzero(score) / len(X_test) * 100 for score in scores]
        df = df.append(dict(zip(kernels, accuracies)), ignore_index=True)
        print("Finished fold #{}\n".format(i))
        i = i + 1

    print("\n\n")
    print(df)
    print("\n\n")
        
    
    # get some images to predict
    test_hogs, test_labels = get_sample_tests(-1, histogram=hist_mode)


    for model in classifiers:
        test_pred = test(model, test_hogs)
        # # Very simple score measure
        score = np.count_nonzero([p == l for p, l in zip(test_pred, test_labels)]) / len(test_labels)
        print("Score right/all: {}%".format(score * 100))


    print("--- Total execution took {} seconds ---".format(time.time() - start_time))
