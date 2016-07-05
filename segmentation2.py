import numpy as np
import cv2, math
import matplotlib.pyplot as plt
from scipy import signal
import os


video_list = [ ]

for video in video_list:
	print video
	cap = cv2.VideoCapture('videos/' + video +'.mp4')
	path_name = "segmented/" + video + "/"
	if not os.path.exists(path_name):
	    os.makedirs(path_name)

	prev_frame = None
	difference_arr = []
	difference_arr2 = []
	frame_arr = []
	x = 0
	#zero_affine = np.asarray(np.matrix([[1,0,0],[0,1,0]])).reshape(-1)
	y = 0
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		y += 1

		if cv2.waitKey(1) & 0xFF == ord('q') or ret== False:
		    break

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		f = np.matrix(gray)
		
		if x == 0 and prev_frame is None:
			prev_frame = f
		elif x == 0:
			affine_arr =  np.asarray(cv2.estimateRigidTransform(prev_frame, f, True)).reshape(-1)
			#print cv2.estimateRigidTransform(prev_frame, f, True)
			if affine_arr[0] is not None:
				frame_arr.append(frame)
				trans_vec = affine_arr[-2:]
				difference_arr.append(np.linalg.norm(trans_vec, ord=4))
				difference_arr2.append(np.linalg.norm(affine_arr, ord=4))
			#difference_arr.append(np.linalg.norm(np.asarray(f).reshape(-1) - np.asarray(prev_frame).reshape(-1)) )
			prev_frame = f

	cap.release()


	difference_arr /= max(difference_arr)
	difference_arr2 /= max(difference_arr2)

	# difference_arr[5] = .5
	# difference_arr[6] = 1
	# difference_arr[7] = .5
	# difference_arr2[5] = .5
	# difference_arr2[6] = 1
	# difference_arr2[7] = .5

	win = signal.hann(30)
	filtered = signal.convolve(difference_arr, win, mode='same') / sum(win)
	filtered2 = signal.convolve(difference_arr2, win, mode='same') / sum(win)

	sub = filtered[1:] - filtered[:-1]
	sub2 = filtered2[1:] - filtered2[:-1]


	key_frames = []
	key_frames2 = []
	for i in range(1,len(sub)):
		if(sub[i] > 0 and sub[i-1] < 0) or i==10:
			key_frames.append(i)
		if(sub2[i] > 0 and sub[i-1] < 0):
			key_frames2.append(i)

	print 'KEY FRAME (Translation vector):', key_frames

	# IMAGE SAVING PORTION
	for i in range(len(key_frames)):
		cv2.imwrite(path_name + str(i) + ".jpg", frame_arr[key_frames[i]])
