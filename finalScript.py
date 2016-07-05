import cv2
import numpy as np
import math
import os

from SR import SR
from boundingbox import boundingbox
from datetime import datetime

DEBUG_MODE = False
SAVE_IMAGE = False

img_name = '222'
img = cv2.imread(img_name+'.jpg')
print('Image read')

startTime = datetime.now()
img_rectified = boundingbox(img,img_name)
print datetime.now() - startTime
cv2.imwrite(img_name+'_rectified.jpg',img_rectified)
print('Image rectified')

img_SR = SR(img_rectified,img_name)
print('SR done')

cv2.imshow('frame',img_SR)
cv2.waitKey()
cv2.imwrite(img_name+'_SR.jpg',img_SR)
