'''
author: Jesús Enrique Cartas Rascón
repo: https://github.com/jesi-rgb/extraccion-rasgos
'''

"""
En una primera aproximación, seguir el esquema 
del ejercicio de clase: entrenar un clasificador 
SVM2, con HoG3 como descriptor, y usarlo para 
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


hog = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, STEP_SIZE, CELL_SIZE, N_BINS)

def compute_hog(img):
    return hog.compute(img)
    

def load_training_data():
    """
    Lee las imágenes de entrenamiento (positivas y negativas) y calcula sus
    descriptores para el entrenamiento.

    returns:
    np.array: numpy array con los descriptores de las imágenes leídas
    np.array: numpy array con las etiquetas de las imágenes leídas
    """ 
    print("Cargando imágenes")

    training_data = []
    classes = []    

    # Train para 0
    zero_images = []
    for filename in os.listdir(PATH_TO_TRAIN_0):
        # using path.join guarantees compatibility across platforms
        filename = os.path.join(PATH_TO_TRAIN_0, filename)

        # 0 is a flag to read the img in grayscale, removing the channels
        img = cv2.imread(filename, 0)
        classes.append(0)
        
        zero_images.append(img)


    pool = mp.Pool(mp.cpu_count())

    predictors = pool.map(compute_hog, zero_images)
    training_data.extend(predictors)

    pool.close()
    pool.join()

    # Train para 1
    one_images = []
    for filename in os.listdir(PATH_TO_TRAIN_1):
        filename = os.path.join(PATH_TO_TRAIN_1, filename)

        img = cv2.imread(filename, 0)
        classes.append(1)
        
        one_images.append(img)
        

    pool = mp.Pool(mp.cpu_count())

    predictors = pool.map(compute_hog, one_images)
    training_data.extend(predictors)

    pool.close()
    pool.join()

    print("Imágenes cargadas")

    return np.array(training_data), np.array(classes)

def train(training_data, classes, kernel=cv2.ml.SVM_LINEAR):
    """
        Entrena el clasificador

        Parameters:
        training_data (np.array): datos de entrenamiento
        classes (np.array): clases asociadas a los datos de entrenamiento

        Returns:
        cv2.SVM: un clasificador SVM
    """
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(kernel)
    svm.train(training_data, cv2.ml.ROW_SAMPLE, classes)
    
    print("Finalizado training")
    return svm


def get_random_sample_tests(n=40):
    img_arrays = []
    classes = []

    # os listdir order is arbitrary, so each time
    # we'll get a different set of images
    for filename in os.listdir(PATH_TO_TEST_0)[:n]:
        filename = os.path.join(PATH_TO_TEST_0, filename)
        img_arrays.append(cv2.imread(filename, 0))
        classes.append(0)

    for filename in os.listdir(PATH_TO_TEST_1)[:n]:
        filename = os.path.join(PATH_TO_TEST_1, filename)
        img_arrays.append(cv2.imread(filename, 0))
        classes.append(1)

    imgs_classes = list(zip(img_arrays, classes))

    #shuffles in place
    np.random.shuffle(imgs_classes)

    print("Tomando imágenes aleatorias para test")
    return imgs_classes

    

def run_example(predict_imgs, labels):
    """
    Prueba de entrenamiento de un clasificador
    """

    training_data, classes = load_training_data()       

    classifier = train(training_data, classes)
    print("Clasificador entrenado")          

    
    # Clasificamos

    pool = mp.Pool(mp.cpu_count())
    hogs = pool.map(compute_hog, predict_imgs)

    predictions = [classifier.predict(hog.reshape(1, -1))[1][0][0] for hog in hogs]
    print(predictions)

    # Very simple score measure
    score = np.count_nonzero(predictions == labels)
    print("Score right/all: {}".format(score))


if __name__ == "__main__":
    random_samples = get_random_sample_tests()

    random_imgs = [pair[0] for pair in random_samples]
    random_labels = [pair[1] for pair in random_samples]

    run_example(random_imgs, random_labels)
