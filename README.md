# **Finding Lane Lines on the Road** 



---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images/solidWhiteRight.jpg "Original"
[image2]: ./test_images/laneLines_thirdPass.jpg "Annoitated"


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
The pipeline works on individual images. Taking an original image that looks lik this:

![Original Image][image1]

and output an annoitated version of the image showing the lane lines. 

![Annoitated Image][image2]

My pipeline consisted of 5 steps. 
1. First, convert the images to grayscale.
2. then apply Gaussian smoothing.
3. The next step is to apply the Canny edge detection  
4. Create a masked edges image for region of interest.
5. Run Hough on edge detected image
6. Combine original image with lines


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by using the code 

posted on the forums. I hope this is acceptable as I tried really hard doing my own and gave up for now.





### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the line curves. 

Another shortcoming could be when there are cars in the region of interest, it would cause the single line to go awry.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to extrapolate the left and right lines to top and bottom of the region of interest instead of the min and max of the detected lines.

