import numpy as np
import cv2, math
import matplotlib.pyplot as plt
from scipy import signal
import os


video_list = []

pathname = 'no_hands/'
pathname2 = 'post_surf/'
surf = cv2.SURF()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

def getint(name):
	num = name.split('.')[0]
	return int(num)
for video in video_list:

	write_path = pathname2 + video
	if not os.path.exists(write_path):
	    os.makedirs(write_path)

	img_names = os.listdir(pathname + video)
	img_names.sort(key=getint)


	match_arr = []
	prev_des = None
	curr_des = None
	x = 0
	img_arr = []
	for name in img_names:
		img = cv2.imread(pathname + video + name)
		img_arr.append(img)
		res = cv2.resize(img,None,fx=.3, fy=.3, interpolation = cv2.INTER_AREA)

		if prev_des is None:
			kp1, prev_des = surf.detectAndCompute(res, None)
		else:
			kp1, curr_des = surf.detectAndCompute(res, None)
			matches = flann.knnMatch(curr_des,prev_des,k=2)
			prev_des = curr_des
			# Need to draw only good matches, so create a mask
			matchesMask = [[0,0] for i in range(len(matches))]
	        
	        # ratio test as per Lowe's paper
			for i,(m,n) in enumerate(matches):
			    if m.distance < 0.7*n.distance:
			        matchesMask[i]=[1,0]
			num_match = np.sum(matchesMask)
			num_match = float(num_match) / len(matches)
			match_arr.append(num_match)
			# print x, x+1, num_match, len(matches)
			x += 1

	threshold = np.array(match_arr) > .2
	threshold = threshold.astype('int')
	# threshold.astype('int')
	print video, threshold
	first = None
	last = None
	for i in range(len(threshold)):
		if threshold[i] == 0:
			if first is None:
				first = i
			last = i
			name = img_names[i]
			cv2.imwrite(write_path + name, img_arr[i])
	if first is not None:
		if first != 0:
			name = img_names[first-1]
			cv2.imwrite(write_path + name, img_arr[first-1])
		name = img_names[last+1]
		cv2.imwrite(write_path + name, img_arr[last+1])		

# plt.plot(range(len(match_arr)), match_arr)
# plt.show()

#########################################################
# des_arr = []
# for name in img_names:
# 	print name
# 	img = cv2.imread('no_hands/' + video_list[0] + '/' + name)
# 	res = cv2.resize(img,None,fx=.5, fy=.5, interpolation = cv2.INTER_AREA)
# 	kp1, des = surf.detectAndCompute(res, None)
# 	des_arr.append(des)

# match_mat = np.zeros((len(des_arr), len(des_arr)))
# for i in range(len(des_arr)):
# 	for j in range(i+1, len(des_arr)):
# 		matches = flann.knnMatch(des_arr[i], des_arr[j], k=2)
# 		val = len(matches)
# 		match_mat[i][j] = val
# 		match_mat[j][i] = val
# for i in match_mat:
# 	print i









###########################################################################
# sub = np.array(match_arr[2:]) - np.array(match_arr[:-2])
# key_frames = []
# for i in range(2,len(sub)):
# 	if(sub[i] > 0 and sub[i-1] < 0):
# 		key_frames.append(i)

# print key_frames


# for video in video_list:
# 	print video
# 	cap = cv2.VideoCapture('../videos/' + video +'.mp4')
# 	path_name = "segmented_surf/" + video + "/"
# 	if not os.path.exists(path_name):
# 	    os.makedirs(path_name)

# 	prev_frame = None

# 	frame_arr = []
# 	match_arr = []
# 	x = 0
# 	#zero_affine = np.asarray(np.matrix([[1,0,0],[0,1,0]])).reshape(-1)
# 	y = 0
# 	while(True):
# 		# Capture frame-by-frame
# 		ret, frame = cap.read()
# 		y += 1
# 		print y
# 		if cv2.waitKey(1) & 0xFF == ord('q') or ret== False:
# 		    break

# 		# Our operations on the frame come here
# 		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 		# f = np.matrix(gray)
		
# 		if x == 0 and prev_frame is None:
# 			# prev_frame = f
# 			kp1, prev_des = surf.detectAndCompute(frame,None)
# 		elif x == 0:
# 			kp1, curr_des = surf.detectAndCompute(frame,None)
# 			matches = bf.match(prev_des,curr_des)
# 			match_arr.append(len(matches))
# 			frame_arr.append(frame)

# 			prev_des = curr_des
# 			#difference_arr.append(np.linalg.norm(np.asarray(f).reshape(-1) - np.asarray(prev_frame).reshape(-1)) )
# 			# prev_frame = f

# 	cap.release()

# 	plt.plot(range(len(match_arr)), match_arr)
# 	plt.show()

# 	match_arr = match_arr / max(match_arr)
		

# 	win = signal.hann(30)
# 	filtered = signal.convolve(match_arr, win, mode='same') / sum(win)

# 	sub = filtered[2:] - filtered[:-2]


# 	key_frames = []
# 	for i in range(2,len(sub)):
# 		if(sub[i] > 0 and sub[i-1] < 0):
# 			key_frames.append(i)


# 	print 'KEY FRAME (Translation vector):', key_frames

# 	# IMAGE SAVING PORTION
# 	for i in range(len(key_frames)):
# 		cv2.imwrite(path_name + str(i) + ".jpg", frame_arr[key_frames[i]])
