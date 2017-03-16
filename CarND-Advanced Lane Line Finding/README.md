
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

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/distorted.jpg "Road Transformed"
[image22]: ./output_images/distortion_correction.JPG "Road Transformed"
[image3]: ./output_images/color_gradient_combo.jpg "Binary Example"
[image33]: ./output_images/window_mask.jpg "Binary Example"
[image4]: ./output_images/warp.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image55]: ./output_images/curvature.jpg "Fit Visual"
[image6]: ./output_images/road_fit.jpg "Output"

** [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points **
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
**Writeup / README**

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

**Camera Calibration**

1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3rd code cell of the jupyter notebook located in "./Advanced Lane Finding.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

**Pipeline (single images)**

1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]


On the previous section I described how I calibrated the camera and got distortion coefficient and camera matrix. I use these two as parameters in the function `cv2.undistort()` and get back undistorted image like the following,

![alt text][image22]

2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears in 5h code cell in the jupyter notebook "./Advanced Lane Finding.ipynb". The `corners_unwarp()` function takes as inputs an image (`img`), as well as a prespective transform matrix `M`. We calculate this `M` with the function `get_perspective_transform_matrix` located on the same code cell. It takes an image `img` as input and returns the perspective transform matrix. In this function I calculate source (`src`) and destination (`dst`) points using following calculations:

```
    img_size = img.shape
    uy = 450
    ly = np.uint(img_size[0]) # lower y
    ulx = 560 # upper left x
    urx = 725 # upper right x
    llx = 150 # lower left x
    lrx = 1170 # lower right x
    offset = 150
    src = np.float32([[llx,ly],[lrx,ly],[ulx,uy],[urx,uy]])    
    dst = np.float32([[llx+offset,ly-10],[lrx-offset,ly-10],[llx+offset,0],[lrx-offset,0]])    

```
Following is the shape of the trapezoid on the image,

![alt text][image33]

The output of the perspective transformation after warping and unwarping is the following,

![alt text][image4]


3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (see the 6th code cell in the Jupyter notebook located in "./Advanced Lane Finding.ipynb"). I convert the image to HLS. Then I apply sobel operator on L channels to get the gradient. I have applied color threshold to S channel because it can detect lines better than other channels. Here's an example of my output for this step. Image with title 'color gradient combo binary' is the single channel image which combines color and gradient threshold. Image with 'Pipeline Result' also combines both color and gradient with 3 channels.

![alt text][image3]


4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used histogram and sliding window search to detect the lane lines. I summed the pixel column wise with histogram and searched for two peak in the first half and second half of the image. These two peaks are the starting points of the two lane lines. Then I use the sliding window technique to find lane lines in the vertically. Starting from the found peak positions we search that which non zero pixels fall in the sliding window area. After this we again do the sliding window search on top of the previous window. This is how we identify two lane pixels. We then use the `np.polyfit` function to fit a polynomial and get polynomial coefficients from these two lane line pixels. The described approach is implemented on the `histogram` function of the notebook at the 7th code cell.

This histogram technique is used only for the first frame. Because after that we know where the lane lines are. So after that we only search with the previously located lane positions with a margin because lanes will vary only a little from previous frame. This approach is coded on the `fit_draw_poly` function of the 7th code cell.

5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

We know the equation to calculate radius of curvature is following,

![alt text][image55]

We know the polynomial coefficients A and B from the previous section. From this we calculate the radius of curvature in the `calculate_carvature` function in 7th code cell of the "./Advanced Lane Finding.ipynb" notebook.

6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `process_frame` function on the 7th code cell.  Here is an example of my result on a test image:

![alt text][image6]

---

**Pipeline (video)**

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/eH8TgoFPdQ0)

---

**Discussion**

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took to solve this Computer vision problem is based on specific conditions. It depends on the curvature of the road, color of the lanes etc. My approach doesn't work well on the harder challenge video because it has a lot of curve in it and our trapezoid shaped window mask fails completely on that. Tuning parameters for different type of environment should not be an ideal solution. I think deep learning based lane finding approach would be a better solution.
