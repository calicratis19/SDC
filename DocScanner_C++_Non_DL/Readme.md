[//]: # (Image References)
[image1]: ./readmeImages/Good3.JPG
[image2]: ./readmeImages/Good4.JPG
[image3]: ./readmeImages/Good2.JPG
[image4]: ./readmeImages/Ok1.JPG
[image5]: ./readmeImages/Ok2.JPG
[image6]: ./readmeImages/Ok3.JPG
[image8]: ./readmeImages/Bad1.JPG
[image9]: ./readmeImages/Bad2.JPG
[image10]: ./readmeImages/Bad3.JPG
[image11]: ./readmeImages/Epochs.PNG
[image12]: ./readmeImages/result1.png
[image13]: ./readmeImages/result2.png
[image14]: ./readmeImages/result3.png

**Receipt Scanner**

The goals / steps of this project are the following:

In this project our goal is to create a pipeline to detect expense receipt in an image and remove its background. One way to achieve this is to use traditional computer vision techniques like canny edge detection, color/gradient threshold etc. The other way is to apply deep learning based approach for receipt detection.


**Overview**

We used traditional computer vision techniques to detect receipt on images. I have created a visual studio project and used C++ for implementation.

**Data Preparation**

I have load, processed, implemented the pipeline and tested the model in the ReceiptScanner.cpp file. We used opencv  to read the images.

**Implementation**

Steps:
On the *DetectReceipt* function I take the following steps:
1) After loading each image I converted the image to gray scale.
2) Then I applied Gaussian blur to remove noises.
3) Then I applied Canny edge detection algorithm on the blurred image.
4) By now canny has produced a binary image. On this image I apply opencv findContours function so that it can detect shapes and gives me the points.
5) Now I have many points which creates some structure on the image. From this I need to find an approximated rectangle. This is because the receipt shape is usually close to a rectangle. In the *OrderPoints* function I do this. The basic idea behind finding 4 points which defines the 4 corners of the rectangle is very simple.

    1) The top-left point x,y coordinate will have the smallest sum, whereas the bottom-right point will have the largest sum.
    2) The top-right point x,y coordinate will have the smallest difference, whereas the bottom-left will have the largest difference

   From these observations I have tried to approximate the 4 corners of the rectangle. This is a naive approach but it does a decent job.
6) The next step is to remove the background. Now that I have 4 corners I can find which points of the image is inside the quadrilateral. I implement this on *MaskBackGround* function. The idea to implement this is following,

    1) I pick all the points on the image one by one. For each point I take another point which has same Y coordinate but a very large X coordinate.
    2) Then I see the line created by these two points intersects with how many sides of the quadrilateral.
    3) If the number is odd then the point is inside the quadrilateral. Otherwise its outside.
    4) Set 0 on all the pixels that are outside of the quadrilateral.

Some Good output example are,

![alt text][image1]
![alt text][image2]
![alt text][image3]

Some not good but not very bad either examples are,

![alt text][image4]
![alt text][image5]
![alt text][image6]


Some Bad examples are,

![alt text][image8]
![alt text][image9]
![alt text][image10]


**Result**

This is a mixed bag. Some detection are very good and some are very bad and others are ok. The model can detect very good when there is a very good distinction between the white receipt and the background. If the background is close to the receipt color or there are lots of noises then the model performs poorly.

**Limitation**

This approach is limited on the specific scenario that I tweaked it on. For different lighting condition and different background or receipt color this model will perform very poorly. This is because the traditional computer vision approach needs a lot tweaking depending on the scenario. It doesn't generalize very well. For this reason now a days deep learning approach is being used widely and taking the place of traditional computer vision techniques.
