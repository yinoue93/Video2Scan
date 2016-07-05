import cv2
import numpy as np
import math
import os

def boundingbox(img,DEBUG_MODE=False,SAVE_IMAGE=False):
    if DEBUG_MODE == True:
        cv2.imshow('frame',img)
        cv2.waitKey()

    color_weights = [[1.0/3, 1.0/3, 1.0/3], [1.0/2, 1.0/2, 0], [1.0/2, 0, 1.0/2], [0, 1.0/2, 1.0/2], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # dilate the image
    kernel_sz = 15
    img_dilated = cv2.dilate(img,np.ones((kernel_sz,kernel_sz)));

    if DEBUG_MODE == True:
        cv2.imshow('frame',img_dilated)
        cv2.waitKey()
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_watershed.jpg',img_dilated)

    for bWeight,gWeight,rWeight in color_weights:
        try:
            # transform to 1 channel according to the color_weight, then edge detect
            gray = np.uint8(img_dilated[:,:,0]*bWeight+img_dilated[:,:,1]*gWeight+img_dilated[:,:,2]*rWeight)
            edges = cv2.Canny(gray,50,150,apertureSize = 3)
            if DEBUG_MODE == True:
                cv2.imshow('frame',gray)
                cv2.waitKey()

            # hough transform/line identification
            interThresh = 40
            thetaThresh = 0.8
            interRecordX = []
            interRecordY = []
            thetaRecord = []
            line_cands = []
            num_lines = 15
            lines = cv2.HoughLines(edges,1,np.pi/180,30)
            lines_len = len(lines)
            if os.name == 'posix':
                lines = lines[0]
            img_lined = img.copy()
            num_lines = min(num_lines, len(lines))

            for i in range(0,num_lines):
                if os.name == 'posix':
                    rho, theta = lines[i]
                else:
                    rho, theta = lines[i][0]
                interX = rho/np.sin(theta) if np.sin(theta)!=0 else float('inf')
                interY = rho/np.cos(theta) if np.cos(theta)!=0 else float('inf')
                
                tooSimilar = False
                for j in range(0,len(interRecordX)):
                    if (abs(interRecordX[j]-interX)<interThresh or abs(interRecordY[j]-interY)<interThresh) \
                    and (abs(thetaRecord[j]-theta)<thetaThresh or abs(thetaRecord[j]+theta)>(math.pi-thetaThresh)):
                        tooSimilar = True
                        break

                col = (255,255,255)
                if tooSimilar:
                    col = (0,0,255)
                else:
                    interRecordX.append(interX)
                    interRecordY.append(interY)
                    thetaRecord.append(theta)
                    line_cands.append([[rho,theta]])
                
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 2000*(-b))
                y1 = int(y0 + 2000*(a))
                x2 = int(x0 - 2000*(-b))
                y2 = int(y0 - 2000*(a))

                cv2.line(img_lined,(x1,y1),(x2,y2),col,2)

            if SAVE_IMAGE == True:
                cv2.imwrite(img_name+'_watershed_boundingbox.jpg',img_lined)
            if DEBUG_MODE == True:
                cv2.imshow('frame',img_lined)
                cv2.waitKey()

            inter_pts = []
            img_intersection = img_lined.copy()
            # find the intersection points
            for i in range(0,len(line_cands)):
                rho1,theta1 = line_cands[i][0]
                for j in range(i,len(line_cands)):
                    rho2,theta2 = line_cands[j][0]
                    diff_angle = abs(theta1-theta2)%np.pi
                    if diff_angle>np.pi/3 and diff_angle<2*np.pi/3:
                        line_eq1 = np.array((math.cos(theta1),math.sin(theta1),-rho1))
                        line_eq2 = np.array((math.cos(theta2),math.sin(theta2),-rho2))
                        intersection_pt = np.cross(line_eq1,line_eq2)
                        intersection_pt = (intersection_pt[0]/intersection_pt[2],intersection_pt[1]/intersection_pt[2])
                        intersection_pt = tuple(int(round(a)) for a in intersection_pt)

                        cv2.circle(img_intersection,intersection_pt,10,(255,0,0),-1)
                        inter_pts.append(intersection_pt)

            if SAVE_IMAGE == True:
                cv2.imwrite(img_name+'_watershed_boundingbox_intersection.jpg',img_intersection)
            if DEBUG_MODE == True:
                cv2.imshow('frame',img_intersection)
                cv2.waitKey()

            # figure out which point belongs to which corner groups
            hei,wid,ch = img.shape
            corners = [[0,0],[0,hei],[wid,hei],[wid,0]]
            pts_group = [[],[],[],[]]
            for i in range(0,len(inter_pts)):
                x,y = inter_pts[i]
                minDistSqrd = x**2+y**2
                minIndx = 0
                for j in range(1,4):
                    distSqrd = (x-corners[j][0])**2+(y-corners[j][1])**2
                    if distSqrd<minDistSqrd:
                        minDistSqrd = distSqrd
                        minIndx = j
                pts_group[minIndx].append(inter_pts[i])

            img_pt_groups = img_lined.copy()
            for i in range(0,4):
                if i==0:
                    color = (255,0,0)
                elif i==1:
                    color = (0,255,0)
                elif i==2:
                    color = (0,255,255)
                else:
                    color = (255,255,0)
                for pt in pts_group[i]:
                    cv2.circle(img_pt_groups,pt,10,color,-1)
            if SAVE_IMAGE == True:
                cv2.imwrite(img_name+'_watershed_boundingbox_vertex_grouped.jpg',img_pt_groups)
            if DEBUG_MODE == True:
                cv2.imshow('frame',img_pt_groups)
                cv2.waitKey()

            # find the intersection points that results in the largest area
            maxArea = 0
            maxPts = (0,0,0,0)
            counter = 0
            for x1,y1 in pts_group[0]:
                for x2,y2 in pts_group[1]:
                    for x3,y3 in pts_group[2]:
                        for x4,y4 in pts_group[3]:
                            area = abs(x1*y2+x2*y3+x3*y4+x4*y1-x2*y1-x3*y2-x4*y3-x1*y4)
                            if area>maxArea:
                                maxArea = area
                                maxPts = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

            img_vertices = img_lined.copy()
            for i in range(0,4):
                cv2.circle(img_vertices,tuple(maxPts[i]),10,(255,0,0),-1)
            if SAVE_IMAGE == True:
                cv2.imwrite(img_name+'_watershed_boundingbox_vertex.jpg',img_vertices)
            if DEBUG_MODE == True:            
                cv2.imshow('frame',img_vertices)
                cv2.waitKey()
            # nice! you made it through! the color setting must have been pretty good, now break
            break
        #except (ZeroDivisionError):
        except (RuntimeError, TypeError, NameError):
            print(bWeight,gWeight,rWeight,' failed')
            pass

    # pull the corners closer to each other to account for the effect made by the dilation
    oriCorners = np.float32(maxPts)
    pull_amount = 8
    delta1 = oriCorners[0]-oriCorners[2]
    slope1 = delta1[1]/delta1[0]
    dilation_correction1 = slope1*pull_amount
    oriCorners[0] += dilation_correction1
    oriCorners[2] -= dilation_correction1

    delta2 = oriCorners[1]-oriCorners[3]
    slope2 = delta2[1]/delta2[0]
    dilation_correction2 = slope2*pull_amount
    oriCorners[1][0] -= dilation_correction2
    oriCorners[1][1] += dilation_correction2
    oriCorners[3][0] += dilation_correction2
    oriCorners[3][1] -= dilation_correction2

    # figure out the perspective transform and apply the transform to the image
    # also guess the dimension of the paper
    len1 = np.linalg.norm(np.array(oriCorners[0])-np.array(oriCorners[1]))
    len2 = np.linalg.norm(np.array(oriCorners[1])-np.array(oriCorners[2]))
    dstCorners = np.float32([[0,0],[0,len1],[len2,len1],[len2,0]])
    H = cv2.getPerspectiveTransform(oriCorners,dstCorners)

    img_rectified = cv2.warpPerspective(img,H,(len2,len1))

    if DEBUG_MODE == True:
        cv2.imshow('frame',img_rectified)
        cv2.waitKey()
    if SAVE_IMAGE == True:
        cv2.imwrite(img_name+'_rectified.jpg',img_rectified)
    
    return img_rectified


