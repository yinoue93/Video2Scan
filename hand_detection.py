import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

video_list = []


# video = 'VIDEO0027/'
for video in video_list:
	print video
	pathname ='rectified/' + video
	pathname2 = 'no_hands/' + video

	if not os.path.exists(pathname2):
	    os.makedirs(pathname2)

	for img_name in os.listdir(pathname):

		img_path  = pathname + img_name
		img = cv2.imread(img_path)

		kernel_sz = 10
		original = img
		img = cv2.dilate(img,np.ones((kernel_sz,kernel_sz)));
		img = img.astype('float')

		grand_total = []
		for i in range(3):
			# left_col = np.asarray(img[:5,:,i]).reshape(-1)
			# right_col = np.asarray(img[-5:,:,i]).reshape(-1)

			# top_row = np.asarray(img[5:-5, :5,i]).reshape(-1)
			# bottom_row = np.asarray(img[5:-5, -5:,i]).reshape(-1)
			left_col = img[10:15,   15:-15,i]
			right_col = img[-15:-10,15:-15,i]

			top_row = img[  15:-15,   10:15 ,i]
			bottom_row = img[15:-15, -15:-10,i]
			
			GRAD = 1
			# left_col_grad = left_col[:, GRAD:] - left_col[:,:-GRAD]
			# right_col_grad = right_col[:,GRAD:] - right_col[:,:-GRAD]
			# top_row_grad = top_row[GRAD:,:] - top_row[:-GRAD,:]
			# bottom_row_grad = bottom_row[GRAD:,:] - bottom_row[:-GRAD,:]


			# left_col_sum = left_col_grad.sum(axis=0)
			# right_col_sum = right_col_grad.sum(axis=0)
			# top_row_sum = top_row_grad.sum(axis=1)
			# bottom_row_sum = bottom_row_grad.sum(axis=1)
			

			# left_col_arr = abs(np.asarray(left_col_sum).reshape(-1))
			# right_col_arr = abs(np.asarray(right_col_sum).reshape(-1))
			# top_row_arr = abs(np.asarray(np.transpose(top_row_sum)).reshape(-1))
			# bottom_row_arr = abs(np.asarray(np.transpose(bottom_row_sum)).reshape(-1))

			left_col_grad = abs(left_col[:, GRAD:] - left_col[:,:-GRAD])
			right_col_grad = abs(right_col[:,GRAD:] - right_col[:,:-GRAD])
			top_row_grad = abs(top_row[GRAD:,:] - top_row[:-GRAD,:])
			bottom_row_grad = abs(bottom_row[GRAD:,:] - bottom_row[:-GRAD,:])


			left_col_sum = left_col_grad.sum(axis=0)
			right_col_sum = right_col_grad.sum(axis=0)
			top_row_sum = top_row_grad.sum(axis=1)
			bottom_row_sum = bottom_row_grad.sum(axis=1)
			

			left_col_arr = abs(np.asarray(left_col_sum).reshape(-1))
			right_col_arr = abs(np.asarray(right_col_sum).reshape(-1))
			top_row_arr = abs(np.asarray(np.transpose(top_row_sum)).reshape(-1))
			bottom_row_arr = abs(np.asarray(np.transpose(bottom_row_sum)).reshape(-1))		



			total = np.concatenate([left_col_arr, bottom_row_arr, top_row_arr, right_col_arr])
			grand_total = np.concatenate([grand_total, total])

			# plt.plot(range(len(total)), total)
			# plt.show()
			# if img_name == '10.jpg_rectified.jpg':
			# 	fig = plt.figure()
			# 	ax1 = fig.add_subplot(141)
			# 	ax1.plot(range(len(left_col_arr)), left_col_arr)
			# 	ax1.set_title("top row")
				
			# 	ax2 = fig.add_subplot(142)
			# 	ax2.plot(range(len(right_col_arr)), right_col_arr)
			# 	ax2.set_title("bottom row")

			# 	ax3 = fig.add_subplot(143)
			# 	ax3.plot(range(len(top_row_arr)), top_row_arr)
			# 	ax3.set_title("left col")

			# 	ax4 = fig.add_subplot(144)
			# 	ax4.plot(range(len(bottom_row_arr)), bottom_row_arr)
			# 	ax4.set_title("right col")
		if sum(i >= 80 for i in grand_total) > 2:
			print img_name, "HANDS"
		else:
			cv2.imwrite(pathname2 + img_name, original)


	# plt.show()
