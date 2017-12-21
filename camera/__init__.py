"""
Used to process images to do Machine Learning+
"""
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def take_photo(name):
    """
    Takes a photo with fswebcam and save it in directory file. 
        
    :Args:
        name (str): Name of jpg file
    """
    os.system('fswebcam -r 160x120 --no-banner {0}.jpg'.format(name))
    return print('(+) Photo taked! Congrats')

def process_photo(photo_dir):
    """
    Process photo to have an array with bites of image.

    :Args:
        photo_dir (str): Path dir of the photo.

    Returns: Numpy Array
    """
    img = Image.open(str(photo_dir)).convert('L').resize((48, 48), Image.ANTIALIAS)
    return np.array(list(img.getdata()))/255.0

def show_image(data, title=None):
    """
    Show array image with matplotlib

    :Args:
        data (numpy array): Numpy array with bits of image
        title (str): Title of photo taken
    """
    plt.imshow(data.reshape(48, 48), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()
    return print('(+) Successfully printed image.')
