import cv2
from scipy import ndimage
import numpy as np
import math

def getBestShift(img):
  cy,cx = ndimage.measurements.center_of_mass(img)

  rows,cols = img.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return shiftx,shifty

def shift(img, sx, sy):
  rows, cols = img.shape
  M = np.float32([[1, 0, sx], [0, 1, sy]])
  shifted = cv2.warpAffine(img, M, (cols, rows))
  return shifted

def _save_img_file(file, img):
  #cv2.imwrite(file, img)
  pass

def threshold(im_gray, method):
    if method == 'fixed':
        (thresh, threshed_im) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)
        print(f"used threshold: {thresh}")
    elif method == 'mean':
        threshed_im = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -22)
    elif method == 'gaussian':
        threshed_im = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)
    else:
        return None

    return threshed_im 

def transform_image_mnist(gray, target_size = (28, 28)):
  """
  transform image to 28x28x3
  """
  # gray
  gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
  _save_img_file("outputs/test_1_gray.png", gray)

  # invert
  gray = 255-gray
  _save_img_file("outputs/test_1_gray_invert.png", gray)
  
  # rescale it
  gray = cv2.resize(gray, target_size)
  _save_img_file('outputs/test_2_rescale.png',gray)

  # better black and white version
  gray = threshold(gray, "mean")
  _save_img_file('outputs/test_3_thresh.png',gray)

  while np.sum(gray[0]) == 0:
      gray = gray[1:]

  while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

  while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

  while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

  _save_img_file('outputs/test_4.png',gray)
  #print(gray.shape)
  rows,cols = gray.shape

  if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols * factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols, rows))
  else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows * factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols, rows))

  colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
  rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
  gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
  _save_img_file('outputs/test_5.png',gray)

  shiftx, shifty = getBestShift(gray)
  shifted = shift(gray, shiftx, shifty)
  gray = shifted
  
  _save_img_file('outputs/test_final.png',gray)

  return gray

# python -m dstest.preprocess.mnist
if __name__ == '__main__':
  from . import datauri_util
  uri = datauri_util.imgfile_to_datauri("inputs/mnist/hard_6.jpg")
  #print(uri)
  img = datauri_util.base64str_to_ndarray(uri)
  transform_image_mnist(img)

