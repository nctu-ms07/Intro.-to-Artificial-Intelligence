import glob
import numpy
from PIL import Image

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    for image_file in glob.glob(dataPath + "/face/*.pgm"):
      image = Image.open(image_file).convert("L")
      image_array = numpy.asarray(image, dtype=numpy.uint8)
      data = (image_array, 1)
      dataset.append(data)
    for image_file in glob.glob(dataPath + "/non-face/*.pgm"):
      image = Image.open(image_file).convert("L")
      image_array = numpy.asarray(image, dtype=numpy.uint8)
      data = (image_array, 0)
      dataset.append(data)
    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset
