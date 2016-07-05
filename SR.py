import cv2
import numpy as np
import math

from brightnessAdjust import brightnessAdjust
from datetime import datetime

def SR(img,img_name,DEBUG_MODE=False,SAVE_IMAGE=False):
    h,w,c = img.shape

    startTime = datetime.now()
    img_bright = brightnessAdjust(img,img_name)
    print datetime.now() - startTime

    startTime = datetime.now()
    # sharpen the image using a Teager filter
    sharpened = np.zeros([h,w,c],np.uint8)
    img_float = np.zeros([h+2,w+2,c])

    img_float[1:-1,1:-1,:] = img_bright.astype('float')
    img_float = img_float/255
    teager_filtered = (3*img_float[1:-1,1:-1,:]**2 - 1/2*img_float[2:,2:,:]*img_float[0:-2,0:-2,:] - 1/2*img_float[2:,0:-2,:]*img_float[0:-2,2:,:] \
                       - img_float[2:,1:-1,:]*img_float[0:-2,1:-1,:] - img_float[1:-1,2:,:]*img_float[1:-1,0:-2,:])*255
    sharpened = (np.maximum(0,np.minimum(255,teager_filtered))).astype('uint8')

    sharpened = cv2.convertScaleAbs(sharpened)

    if DEBUG_MODE == True:
        cv2.imshow('frame',sharpened)
        cv2.waitKey()
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_sharpened.jpg',sharpened)

    img_sharp_bright = brightnessAdjust(sharpened,img_name)

    # boundary extraction
    gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
    kernel_sz = 10
    img_eroded = cv2.erode(gray,np.ones((kernel_sz,kernel_sz)));
    img_dilated = cv2.dilate(gray,np.ones((kernel_sz,kernel_sz)));
    img_boundary = cv2.absdiff(img_eroded,img_dilated)

    if DEBUG_MODE == True:
        cv2.imshow('frame',img_boundary)
        cv2.imshow('frame2',img_eroded)
        cv2.imshow('frame3',img_dilated)
        cv2.waitKey()
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_eroded.jpg',img_eroded)
        cv2.imwrite(img_name+'_dilated.jpg',img_dilated)
        cv2.imwrite(img_name+'_boundary.jpg',img_boundary)

    # blob up the boundary
    ##kernel_sz2 = 1
    ##img_boundary_blob = cv2.dilate(img_boundary,np.ones((kernel_sz2,kernel_sz2)));

    # create a mask - 1 for where the boundary is, and 0 otherwise
    ret,mask_sharp = cv2.threshold(img_boundary,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret,mask_sharp = cv2.threshold(img_boundary,30,255,cv2.THRESH_BINARY)
    # the page edges should not be included in the mask
    edge_sz = 20
    mask_edge = np.zeros([h,w],np.uint8)
    mask_edge[edge_sz:-edge_sz,edge_sz:-edge_sz] = 255
    mask_sharp = cv2.bitwise_and(mask_sharp,mask_edge)
    # create an inverse mask
    mask_sharp_inv = cv2.bitwise_not(mask_sharp)

    if DEBUG_MODE == True:
        cv2.imshow('frame',mask_sharp)
        cv2.waitKey()
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_mask.jpg',mask_sharp)

    # combine the images appropriately
    #img_back = cv2.bitwise_and(img_bright,img_bright,mask = mask_sharp_inv)
    #img_sharp = cv2.bitwise_and(img_sharp_bright,img_sharp_bright,mask = mask_sharp)
    #img_combined = cv2.add(img_back,img_sharp)
    img_boundary3D = np.zeros((h,w,3))
    img_boundary3D[:,:,0] = img_boundary
    img_boundary3D[:,:,1] = img_boundary
    img_boundary3D[:,:,2] = img_boundary
    img_boundary3D /= 255
    img_combined = img_bright*(1-img_boundary3D) + img_sharp_bright*img_boundary3D
    img_combined = img_combined.astype('uint8')

    # resize the sharpened image
    img_combined = cv2.resize(img_combined,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    img_combined2 = cv2.resize(sharpened,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)

    if DEBUG_MODE == True:
        cv2.imshow('back',img_back)
        cv2.imshow('text',img_sharp)
        cv2.imshow('frame',img_combined)
        #cv2.imshow('frame2',img_combined2)
        cv2.waitKey()
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_SR.jpg',img_combined)

    print datetime.now() - startTime
    return img_combined
