import glob
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as pp


storage = []

def create_npz(dir_name) :
  a = glob.glob(dir_name + '\*.png')
  idx = 1
  for x in a :
    d = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    #storage.append(d)
    np.savez_compressed('test_image_' + str(idx), image2D = d) 
    idx = idx + 1
  #np.savez_compressed(npz_name, *storage)
  #return npz_name

def test_npz(npz_name) :
  a = np.load(npz_name)
  im_arr = a['image2D'] 
  pp.imshow(im_arr, cmap = 'gray')
  pp.show()

if __name__ == "__main__":
  create_npz('C:\\Users\\Sriram\\Documents\\2DVoxelmorph\\data\\test_images\\')
  test_npz(r'C:\Users\Sriram\Documents\2DVoxelmorph\src\test_images\test_image_1.npz')
