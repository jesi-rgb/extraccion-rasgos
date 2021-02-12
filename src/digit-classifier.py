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
import pandas as pd
import numpy as np
import argparse
import time
import os

import multiprocessing as mp

import cv2

from sklearn.model_selection import KFold

from LBP import LBPDescriptor




PATH_TO_TRAIN = "mnist_data/train"
PATH_TO_TEST = "mnist_data/test"

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
    

def load_training_data(n=1000, histogram="hog"):
    """
    Carga las imágenes de entrenamiento y devuelve sus histogramas.

    `histogram`: "hog" o "lbp". Crea los histogramas con el método
    HOG, o el método LBP.
    """ 
    print("\n> Cargando imágenes")

    

    # labels array
    classes = []  

    # all the imgs will lie here  
    img_paths = []

    for folder in os.listdir(PATH_TO_TRAIN):
        # get all the paths for 0s and 1s, and append the labels
        for filename in os.listdir(os.path.join(PATH_TO_TRAIN, folder))[:n]:
            # using path.join guarantees compatibility across platforms
            img_paths.append(os.path.join(PATH_TO_TRAIN, folder, filename))
            
            if folder == 'zero':
                classes.append(0)

            elif folder == 'one':
                classes.append(1)
                
            elif folder == 'two':
                classes.append(2)
                
            elif folder == 'three':
                classes.append(3)
                
            elif folder == 'four':
                classes.append(4)
                
            elif folder == 'five':
                classes.append(5)
                
            elif folder == 'six':
                classes.append(6)
                
            elif folder == 'seven':
                classes.append(7)
                
            elif folder == 'eight':
                classes.append(8)

            elif folder == 'nine':
                classes.append(9)
                

    # for filename in os.listdir(PATH_TO_TRAIN_1):
    #     # using path.join guarantees compatibility across platforms
    #     img_paths.append(os.path.join(PATH_TO_TRAIN_1, filename))
    #     classes.append(1)

    # create a pool with the number of cores
    pool = mp.Pool(mp.cpu_count())

    # having all the paths, we can read all the imgs
    # in parallel, which is much faster
    images = pool.map(read_img_bw, img_paths)


    print("\n> Calculando histogramas {} de {} imágenes".format(histogram.upper(), len(images)))
    # and compute the hogs also in parallel
    predictors = None
    if(histogram == "hog"):
        predictors = pool.map_async(compute_hog, images).get()
    elif histogram == "lbp":
        predictors = pool.map_async(LBPDESC.compute, images).get()
    
    # important: always close the pool
    pool.close()
    pool.join()

    print("\n> Histogramas generados")

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


def get_sample_tests(n=500, histogram="hog"):
    print("\n> Calculating histograms for test images.")
    img_paths = []
    classes = []

    for folder in os.listdir(PATH_TO_TEST):
        # get all the paths for 0s and 1s, and append the labels
        for filename in os.listdir(os.path.join(PATH_TO_TEST, folder))[:n]:
            # using path.join guarantees compatibility across platforms
            img_paths.append(os.path.join(PATH_TO_TEST, folder, filename))
            
            if folder == 'zero':
                classes.append(0)

            elif folder == 'one':
                classes.append(1)
                
            elif folder == 'two':
                classes.append(2)
                
            elif folder == 'three':
                classes.append(3)
                
            elif folder == 'four':
                classes.append(4)
                
            elif folder == 'five':
                classes.append(5)
                
            elif folder == 'six':
                classes.append(6)
                
            elif folder == 'seven':
                classes.append(7)
                
            elif folder == 'eight':
                classes.append(8)

            elif folder == 'nine':
                classes.append(9)

 
    pool = mp.Pool(mp.cpu_count())

    img_arrays = pool.map(read_img_bw, img_paths)

    hogs = None
    if(histogram == "hog"):
        hogs = pool.map(compute_hog, img_arrays)
    elif (histogram == "lbp"):
        hogs = pool.map(LBPDESC.compute, img_arrays)

    pool.close()
    pool.join()

    print("\n> Finished test histograms.")
    return hogs, classes

    
def test(model, predict_imgs_hog):
    return [int(model.predict(hog.reshape(1, -1))[1][0][0]) for hog in predict_imgs_hog]



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-hist", "--histogram", required=False,
    help="Histogram mode: hog or lbp", default="hog")
    
    ap.add_argument("-f", "--fold", required=False,
    help="Number of folds for the CV", default=5)
    
    args = ap.parse_args()

    hist_mode = args.histogram

    start_time = time.time()
    
    training_data, classes = load_training_data(n=50, histogram=hist_mode)

    kernels = {"linear":cv2.ml.SVM_LINEAR, "polynomial":cv2.ml.SVM_POLY, "rbf":cv2.ml.SVM_RBF, "sigmoid":cv2.ml.SVM_SIGMOID}
    df = pd.DataFrame(columns=kernels.keys())
    kfold = KFold(n_splits=int(args.fold))

    print("\n> Starting SVM training\n")
    i = 1
    for train_index, test_index in kfold.split(training_data):
        X_train, X_test = training_data[train_index], training_data[test_index]
        y_train, y_test = classes[train_index], classes[test_index]

        classifiers = [train_kernels(X_train, y_train, kernel) for kernel in kernels.values()]

        predictions = [test(model, X_test) for model in classifiers]
                
        scores = [[p == l for p, l in zip(pred, y_test)] for pred in predictions]
        accuracies = [np.count_nonzero(score) / len(X_test) * 100 for score in scores]
        
        df = df.append(dict(zip(kernels, accuracies)), ignore_index=True)
        
        print("Finished fold #{}\n".format(i))
        
        i = i + 1

    print("\n> Training results (validation accuracy per fold):\n")
    print(df)
        
    
    # get some images to predict
    test_hogs, test_labels = get_sample_tests(50, histogram=hist_mode)


    print("\n> Test results:")
    for model, name in zip(classifiers, kernels.keys()):
        test_pred = test(model, test_hogs)
        # # Very simple score measure
        score = np.count_nonzero([p == l for p, l in zip(test_pred, test_labels)]) / len(test_labels)
        print("{}: \t\t{}%".format(name, score * 100))


    print("\n> --- Total execution took {} seconds ---".format(time.time() - start_time))
