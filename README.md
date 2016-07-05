# Video2PDF
We introduce a system which takes a video of a document to be scanned as its input and outputs a set of images that correspond to the document pages. These images could then be converted to a single pdf file. Unlike the earlier works, not only does the system recognize each page flips and rectifies the pages, it also applies a series of image enhancements to improve the output image quality. The system also contains a false positive mechanism to remove any duplicate images. As taking a video takes significantly less time than scanning each page at a time using traditional methodologies, this greatly reduces the amount of time and mental pain that a user must go through when scanning documents.

## boundingbox.py
Takes in a segmented image as an input.<br />
Figures out the bounding box and rectifies the image.<br />
![Video2Scan](./imgs/222bounding.PNG)

## brightnessAdjust.py
Takes in a rectified image as an input. <br />
Guesses the brightness information of the input image, and tries to remove the uneven brightness distribution.<br />
![Video2Scan](./imgs/brightnessAdjust.PNG)

## finalScript.py
Automates the process of bounding box identification, image rectification, brightness identification, and Super Resolution.

## hand_detection.py
Takes in a segmented image as an input. <br />
Identifies frames with hands in them, and removes those frames.

## segmentation2.py
Uses estimateRigidTransform() to segment a video. If the affine transform of consecutive frames is "large," two frames are deemed to be of different pages.

## SR.py
Takes the brightness adjusted image as the input. <br />
Applies edge enhancement to the image.

## surf_matching.py
Uses SURF feature points to segment a video. If consecutive frames contain a few matching features, two frames are deemed to be of different pages.<br />
![Video2Scan](./imgs/kp_coreespondence_correct_small.jpg)

## Results
Results are shown in following images. From the left, the images are: original, brightness adjusted, and super resolutioned. <br />
![Video2Scan](./imgs/SR222.PNG) <br />
![Video2Scan](./imgs/SR222_2.PNG) <br />
![Video2Scan](./imgs/SR555.PNG)