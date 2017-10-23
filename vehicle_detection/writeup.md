
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines # through # of the file called `vehicle_detection.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I used the cv2 HOGDescriptor for extracting HOG features.
I applied the extraction on the car and notcar images after converting thme to grayscale colorspace.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I went for the parameters set out in the forums for cv2 HOG feature extraction.

winsize = (64,64)
blocksize = (16,16)
blockstride = (8,8)
cellsize = (8,8)
nbins = (9)
derivAperture = 1
winSigma = 4
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection =0
nlevels = 64

For the spatial binning I used 24 by 24 binning to save processing time.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a fully connected neural network using the HOG features plus spatially binned color for the HSV color-space and the binned colour values histogram for the RGB channels. The neural network is made up of an input and batchnormalization layer two dense layers with 128 neurons each an an ouput layer with a single neuron for binary classification (car noncar). 
Before training the network I trained and preprocessed the feature vectors using an SkLearn StandardScaler.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I went on a scaled window approach, where the window-size gets gradually bigger as the windows move to the bottom of the 
Image (as the Y value increases). The cars are bigger when nearer to the camera.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on the changing sized windows over the image. Resizing each windoe to the 64 by 64 size the Classifyer 
was trained on. using the same features as in the training: HOG (grayscale), Spatial binning (HSV) and histograms(RGB), which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions and supress false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. I used the weighted average of the heatmaps of prior frames in order to reduce wobbling and false positives.

Here's an example result showing the heatmap from a single frame, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image:

### Here are all the possible windows:

![alt text][image5]

### Here are all the positive windows:

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image7]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  After trying on the SVM for 9.29 Seconds with a test accuracy of 0.986 I tried the ANN approach. This got me an accuracy of 0.998 a full 1% improvement
I implemented an object oriented `VehicleTracker()` class for combining heatmaps over several frames.
If I'd had time I'd probably try different parameters for the HOG feature extraction and try it on the different color channels instead of just grayscale. Try some more ways to supress false positives, maybe 

