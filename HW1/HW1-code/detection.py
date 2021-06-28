import os
import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    with open(dataPath, newline='') as csv_file:
      rows = list(csv.reader(csv_file, delimiter = ' '))
      i = 0
      while i < len(rows):
        imgFile = rows[i][0]
        nFace = int(rows[i][1])
        
        img = Image.open(os.path.dirname(dataPath) + '/' + imgFile).convert("L")
        fig , ax = plt.subplots()
        ax.axis('off')
        ax.set_title(imgFile)
        ax.imshow(img, cmap='gray')

        i += 1
        for j in range(nFace):
          x = int(rows[i + j][0])
          y = int(rows[i + j][1])
          w = int(rows[i + j][2])
          h = int(rows[i + j][3])

          face_img = img.crop((x,y,x + w, y + h)).resize((19,19))
          face_array = np.asarray(face_img, dtype=np.uint8)

          if clf.classify(face_array) == 1:
            ax.add_patch(Rectangle((x,y),w,h,linewidth=1,edgecolor='g',facecolor='none'))
          else:
            ax.add_patch(Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none'))

          j += 1
        i += j
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
