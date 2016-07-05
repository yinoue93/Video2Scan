import cv2
import numpy as np
import math

def brightnessAdjust(img,img_name,DEBUG_MODE=False,SAVE_IMAGE=False):
    h,w,c = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_sz = 8
    img_dilated = cv2.dilate(gray,np.ones((kernel_sz,kernel_sz)));
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_dilated.jpg',img_dilated)

    # figure out where 'significant objects' do not exist
    img_diff = cv2.absdiff(gray,img_dilated)
    # dilate it again
    img_diff = cv2.dilate(img_diff,np.ones((4,4)));

    # create a mask
    ret,mask = cv2.threshold(img_diff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # the page edges should not be included in the mask
    edge_sz = 30
    mask_edge = np.zeros([h,w],np.uint8)
    mask_edge[edge_sz:-edge_sz,edge_sz:-edge_sz] = 255
    mask = cv2.bitwise_and(mask,mask_edge)

    # find contours and fill them
    im2, contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, 255, -1)
    mask = cv2.bitwise_not(mask)

    if DEBUG_MODE == True:
        cv2.imshow('frame',mask)
        cv2.waitKey()
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_mask.jpg',mask)

    img_marked = img.copy()
    # collect data points
    sample_sz = 30
    window_sz = 1
    freqs = 4
    margin = 30
    h_pts = np.linspace(margin,h-margin,sample_sz).astype('int')
    w_pts = np.linspace(margin,w-margin,sample_sz).astype('int')
    freqH = np.pi/h
    freqW = np.pi/w
    A = np.empty((0,4*freqs+1), int)
    y = np.empty((0,1), int)
    for hloc in h_pts:
        for wloc in w_pts:
            if mask[hloc,wloc]:
                sample_coeff = []
                for i in range(1,freqs+1):
                    sample_coeff = sample_coeff + [np.cos(i*freqH*hloc),np.sin(i*freqH*hloc) \
                                                        ,np.cos(i*freqW*wloc),np.sin(i*freqW*wloc)]
                sample_coeff.append(1)
                #avg = np.mean(img_dilated[hloc-window_sz:hloc+window_sz, \
                #                          wloc-window_sz:wloc+window_sz])
                avg = img_dilated[hloc,wloc]
                A = np.vstack((A,sample_coeff))
                y = np.vstack((y,avg))

                cv2.circle(img_marked,(wloc,hloc),5,(0,0,255),-1)

    coeff = np.dot(np.linalg.pinv(A),y)
    brightness_img = np.ones((h,w))*coeff[freqs*4]

    for i in range(1,freqs+1):
        cosH = np.cos(i*freqH*np.arange(1,h+1))
        brightness_img += np.transpose(np.tile(coeff[(i-1)*4]*cosH,(w,1)))
        sinH = np.sin(i*freqH*np.arange(1,h+1))
        brightness_img += np.transpose(np.tile(coeff[(i-1)*4+1]*sinH,(w,1)))

        cosW = np.cos(i*freqW*np.arange(1,w+1))
        brightness_img += np.tile(coeff[(i-1)*4+2]*cosW,(h,1))
        sinW = np.sin(i*freqW*np.arange(1,w+1))
        brightness_img += np.tile(coeff[(i-1)*4+3]*sinW,(h,1))

    if DEBUG_MODE == True:
        cv2.imshow('frame',np.clip(brightness_img,0,255).astype('uint8'))
        cv2.waitKey()
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_brightMap.jpg',brightness_img)
        
    brightness_img_3D = np.zeros((h,w,3))
    brightness_img_3D[:,:,0] = brightness_img
    brightness_img_3D[:,:,1] = brightness_img
    brightness_img_3D[:,:,2] = brightness_img
    img_adjusted = np.clip(img/brightness_img_3D*255,0,255)
    
    if DEBUG_MODE == True:
        cv2.imshow('frame',img_adjusted.astype('uint8'))
        cv2.waitKey()
        cv2.imshow('frame2',img_marked)
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_brightAdjusted.jpg',img_adjusted)

    return img_adjusted.astype('uint8')
