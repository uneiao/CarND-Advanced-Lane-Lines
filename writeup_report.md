## Writeup Report

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration4.jpg "Original"
[image2]: ./output_images/undistort_calibration4.jpg "Undistorted"
[image3]: ./output_images/undistort_test4.jpg "Undistorted Example"
[image4]: ./output_images/thresholded_image.jpg "Thresholded"
[image5]: ./output_images/pick_polygon_persp.jpg "Perspective"
[image6]: ./output_images/pick_polygon_birdview.jpg "Birdview"
[image7]: ./output_images/fit.jpg "Fit"
[image8]: ./output_images/final_result.jpg "Result"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The implemented methods for this step is contained in the file "calibration.py" from line 26 to line 65.

I start by preparing "object points" (in method `get_corners`), which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function (implemented in method `camera_calibration`).
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result(in method `distortion_correction`): 

Original:

![alt text][image1]

Undistorted:

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of absolute thresholds on lightness and saturation channel to generate a binary image (thresholding steps taken in the file `thresholds.py`).
Here's an example of my output for this step.  

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 68 through 90 in the file `calibration.py`.
The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.
I choose to hardcode the source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 295, 680      | 250, 710      | 
| 595, 450      | 250, 0        |
| 700, 450      | 1050, 0       |
| 1150, 680     | 1050, 710     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart
to verify that the lines appear parallel in the warped image.

![alt text][image5]
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use the histogram sliding window method provided in the lectures to pick the peaks in the histogram as the occurrence of lane-line pixels.
The codes is in the file `line_fit.py`, from line 9 to line 108.

Here I edit the method a little bit that I use the lane-line base position detected in previous frames to restrict the search area in a histogram (line 27 - 39).

Then the position of lane-line pixel are transformed into meter unit for the calculation of fitting 2nd-order-polynomial equations.
A plotted result is shown below:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated with the fitting polynomial result, using the formula provided in the lecture, as well as the position of lane-lines
are calculated with the bottom line of images as y positions.
The codes is in lines 125 through 157 in file `line_fit.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The related codes line from 123 through 142 in my code in the function `draw_lane()` of file `main.py` .  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To my concern, the sliding window method for detection is not robust enough against the noise in some dirty lanes.
I use the information of pixel position in the previous frames to narrow down search area in order to reduce the probability of taking noises as result. 
If the peak in the histogram is noise at the beginning frame, the later detected results will all probably go wrong.

For further improvement, my approach is to train some classifiers to filter out those noises. The histogram on binary warped image could be used to yield a region of interest.
Within this roi, I would use the flood-fill method to scan the pixels and get several contours. Each contour is a potential noise. Then I extract the patch
of a contour area from the image as training samples to train or predict by a trained classifiers. Those contours which stay after filtering step would be the accurate results.
