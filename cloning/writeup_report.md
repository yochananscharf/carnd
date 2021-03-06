# **Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.jpg "Model Visualization"
[image3]: ./examples/center_2017_09_18_14_16_56_194.jpg "Recovery Image"
[image6]: ./examples/non_flip.png "Normal Image"
[image7]: ./examples/flip.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128. 

The model includes RELU layers to introduce nonlinearity (code line 40), and the data is normalized in the model using a Keras batch normalization layer (code line 39). 

#### 2. Attempts to reduce overfitting in the model

The model contains several dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets using a validation split of 20% to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an RMSprop optimizer with learning rate decay, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road in 4 runs. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the ResNet I thought this model might be appropriate because of it's simplicity and ease of parameter tuning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I spent hours and hours on tuning the model and hypermeters. What turned out to be really important was the learning rate. If it was to high, the model would overshoot and predict the same value for any image.

To combat the overfitting, I used dropuot rather aggresively. In the end the validation loss was lower than the train loss.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as where the wate and sky caused the lines not to contrast enough or at the bridge. to improve the driving behavior in these cases, I added some more training, focusing on these spots. Which in Data Science circles is called manual overfitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like starting from straddling the road edge :

![alt text][image3]


Then I repeated this process on track two in order to get more data points.

##### Augmenting the data set: 

After the collection process, I had X number of data points. I then preprocessed this data by resizing in order to save in training time. Flipping the image and tacking the inverse of the steering angle doubled the training data. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

### Enviroment and hardware.
The model was written using the Keras-2 version. The training and validation was done usin the Linux beta simulator.
The screen resolution and graphics quality were 960x720 and good respectively. When using higher resolution and quality, the car would leave the road. Used a Laptop powered with an Nvidia GPU, namely a GeForce 940mx.
