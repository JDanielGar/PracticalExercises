"""
Get the data for Machine Learning
"""
import camera
import numpy as np
from sklearn.utils import shuffle

def get_photo_data(objects=1, number=1):
    """
    Uses camera module to take image and process images

    :Args:
    objects (int): Number of objects tha you'll pull of the array
    number (int): Number of photos

    :Returns: Numpy array of bites
    """
    data = []
    targets = []
    counter = 0
    while counter < objects:
        counter += 1
        input('Press some key for take the last {0} photos to a object'.format(objects-counter+1))
        for x in range(number):
            camera.take_photo('index_image')
            data.append(camera.process_photo('./index_image.jpg'))
            targets.append(counter-1)
    data, targets = shuffle(data, targets)
    return data, targets

def save_as_csv(data):
    """
    Save data of numpy array into a CSV.
    """
    np.savetxt("data.csv", data, delimiter=",")
