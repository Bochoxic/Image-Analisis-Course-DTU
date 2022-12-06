from skimage import io
import matplotlib.pyplot as plt
import math
import numpy as np
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage.util import img_as_float

in_dir = "Exercise 7/data/"
src_name = "Hand1.jpg"
dst_name = "Hand2.jpg"
src_img = io.imread(in_dir+src_name)
dst_img = io.imread(in_dir+dst_name)


dst = np.array([[624, 298], [379, 162], [195, 274], [274, 440], [594, 446]])

plt.imshow(dst_img)
plt.plot(dst[:,0], dst[:,1], '.r', markersize=12)
plt.show()