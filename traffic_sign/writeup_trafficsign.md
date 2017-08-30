# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization.jpg "Visualization"
[image2]: ./before_and_after.jpg
[image4]: ./signs/traffic_sign_0.jpg "Traffic Sign 1"
[image5]: ./signs/traffic_sign_1.jpg "Traffic Sign 2"
[image6]: ./signs/traffic_sign_2.jpg "Traffic Sign 3"
[image7]: ./signs/traffic_sign_3.jpg "Traffic Sign 4"
[image8]: ./signs/traffic_sign_4.jpg "Traffic Sign 5"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. 

You're reading it! and here is a link to my [project code](https://github.com/yochananscharf/carnd/blob/CarND-Traffic-Sign-Classifier-Project/traffic_sign/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is [32, 32].
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to 64 by 64 pixel images for better images. The I applied the OpenCV 
Normalization function. Normalizing the image does something similair to histogram streching and brightens the image.

Here is an example of a traffic sign image before and after resizing and normalizing..

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

[image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 64x64x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, no padding (valid) , outputs 60x60x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 30x30x16 				|
| Convolution 5x5     	| 1x1 stride, no padding (valid) , outputs 26x26x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Flatten  |  13x13x32 | output 5408.
| Fully connected		| 5408        									|outputs 240 
| Relu
| Fully connected		| 240        									|outputs 168 
| Relu
| Fully connected		| 168        									|outputs 43
| Logits				| etc.        									|

 


#### 3. Training the model. 

To train the model, I used an softmax_cross_entropy_with_logits cross entropy and Adam optimizer.
Learning rate of 0.001 and a batch size of 32.

#### 4. At first I didn't get to the 93% accuracy until I changed the Lenet architechture and preprocessed the images as oulined above. Training for 16 Epochs got the following results:

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.954 
* test set accuracy of 0.944


* I chose the Lenet architecture for it's simplicity. Changed the number of ouputs of the activation layer and the number of filters at each convolution layer and of course the input dimension to 64 by 64.
* The Model generalises pretty well to the validation and test sets.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because of the small porportion of the sign to the image.
The therd image might be difficult to classify because of the watermarks all over the image.


#### 2. Prediction results on the new traffic-signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only    		| Speed limit (80km/h)  									| 
| Turn right ahead    			|  Turn right ahead										|
| Pedestrians				| Yield										|
| Bumpy road	      		| Bumpy Road					 				|
| No entry			| No entry    							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This does no compare favorably to the accuracy on the test set of 94%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were



|sign_0	|sign_1	|sign_2	|sign_3	|sign_4|
|:----------:|:------------------:|:-----------------:|:----------------:|:-------------:|
|Speed limit (80km/h), 0.0)|	(Turn right ahead, 21.0)|	(Yield, 18.0)	|(Bumpy road, 33.0)|	(No entry, 37.0)|
|(Bicycles crossing, -0.0)|	(Ahead only, 9.0)	|(Priority road, 7.0)|	(Dangerous curve to the right, 8.0)	|(Speed limit(20km/h), 4.0)|
|(End of speed limit (80km/h), -0.0)	|(No passing for vehicles over 3.5 metric tons,6.0)|(End of all speed and passing limits, 5.0)|	(Bicycles crossing, 7.0)|	(Stop, 2.0)|
|	(No entry, -0.0)	|(Go straight or left, 6.0)|	(Ahead only, 1.0)	|(Speed limit (20km/h), 7.0)|	(Keep right, 0.0)|
|(Speed limit (30km/h), -2.0)|	(Speed limit (80km/h), 5.0)	|(No vehicles, -1.0)	|(Speed limit (60km/h), 5.0)|	(Traffic signals, -1.0)|


