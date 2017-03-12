[//]: # (Image References)
[image1]: ./images/Read_Data.JPG
[image2]: ./images/Cropped.JPG
[image3]: ./images/Target_Mask.JPG
[image4]: ./images/Unet.PNG
[image5]: ./images/model.JPG
[image6]: ./images/IOU.JPG
[image7]: ./images/pred1.PNG
[image8]: ./images/pred2.PNG
[image9]: ./images/pred3.PNG
[image10]: ./images/almost_unet.PNG
[image11]: ./images/better_than_human.png

**Vehicle Detection Project**

The goals / steps of this project are the following:

In this project our goal is to create a pipeline to detect vehicles in a video stream. One way to achieve this is to extract HOG and spatial color features from each frame of the video and train a classifier to detect vehicle. Then implement a sliding window technique to detect vehicles on the window.


**Overview**

In my project I didn't use HOG feature. I used the spatial color features and trained a Convolutional Neural Network to directly draw bounding boxes on vehicles of a video frame. Its much more simpler and generalize much better for unseen data.

The model I trained to achieve the goal is inspired from an famous CNN model called U-net. This model directly predicts bounding box in a video frame. It has a special architecture. It first implement Convolutional layers which shrink the width and height and increases the depth of the feature map and then it implements merging layers which concatenates the Convolutional layers through an up-convolving process to finally achieve an image mask which contains the pixel wise prediction values to detect a car. This approach is inspired by a [prize winning submission on Kaggle ultrasound Nerve Segmentation competition challenge.](https://github.com/jocicmarko/ultrasound-nerve-segmentation)

**Data Preparation**

I have load, process and trained the model in the Vehicle_Detection.py file. I have performed the testing and applying the model to the video in the Vehicle_Detection_Tracking.ipynb Jupyter notebook.

I have used Udacity provided [4.5 GB](https://github.com/udacity/self-driving-car/tree/master/annotations) data set to train the model. This data has ~24K images with 1920x1200 resolution. This data set has the vehicles annotated in red rectangular bounding boxes. Also the csv file contains the upper left and lower right corners for the bounding boxes. Steps to prepare the data are following:

1) The csv file has error in the column names corresponding to the bounding boxes. We had to fix this problem to correctly read the csv file. Also the annotated rectangles were implemented through machine learning and human help. There are cases where images are not detected properly or wrongly detected. We could not do anything these cases.

2) The data set is separated into two separate folders. So we read them in two separate pandas DataFrame. The data set contains 3 classes of objects: Car, Track and Pedestrians. We removed the Pedestrian classes and merged car and track objects into one single class. Refer to the line 31 to 51 in the Vehicle_Detection.py file.

![alt text][image1]

3) We have resized the image to 960 x 630 resolution because my Geforce 980M card could not handle the full resolution image as an input.

4) The upper part of the image has sky and trees, we cropped the 2/5th part from the top of the image. And also the car dashboard was cropped which is 1/7th of the image. See the CropImage() function in the python file.

![alt text][image2]

5) IMO the data set contains much more darker images than sunny ones. That's why my model was detecting some shadows as cars. I implemented brightness augmentation to my data set. I scaled the V channel of the image by random value between 1 and 1.5. Please see the function RandomBrightness() function in the python file.

**Target Dataset**

From the csv file we know the bounding boxes coordinates in the images. We create a single channel image numpy array initialized with 0. We then fill up the entire region of those bounding boxes with 1 on the image array. This gives us our image target mask. Refer to the code line 104 to 106 of TrainDataGenerator() function in the python file.

![alt text][image3]


**Model**

I didn't implement the full Unet model as it takes a lot of time to train. With even a scaled down model with resized and cropped image input, it takes about 20 minutes to train on 10K image data. The original Unet model is the following,

![alt text][image4]

I had to play around with the model a lot to come up with a decent mini model. First I tried without the 512 and 256 filter layers. The model did ok with this one. It was detecting a lot of false positives. So I had to add the 256 filter feature extraction layer. With that layer I achieved the best model. Please refer to the function CreateModel() in the python file.
![alt text][image5]

But it had a lot of false positive. It was loving the shadows too much. Then I noticed the data set is imbalanced about lighting condition. So after running 40 iteration without any brightness augmentation I decided to load the weight of 40th iteration and apply image augmentation. After running 9 more iteration I achieved the best model. Still it detects some false positives but it also detects a lot of cars in the other lane.

**Training**

The hyper parameters are,

Training samples per epoch = 10000
Training Batch Size = 16
Resize Image Row = 630
Resize Image Column = 960
Cropped Image Row = 288
Cropped Image Column = 960
Learning Rate = .0001

We could not fit more than 16 batch size of images because of the limitation of GPU memory(8GB). With learning rate .00001 the model performed very poorly. We have used sigmoid activation function for the last layer because we wanted a pixel wise probability of being a car. Also we have used IOU(intersection of union) also known as [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) for loss function and accuracy matrix. This is simply taking 2 times intersection of two images meaning (pixel wise value multiplication of two images and summation)x2 divided by union of two images meaning summation of all the pixel values in two images. Please refer to the function dice_coef() in the python file. Visualization is given below,

![alt text][image6]

We have two data sets from Udacity in separate folders with csv file having separate column structures. So we have read the data into two pandas DataFrame and put them in a list. Then we randomly select one of them and then randomly select one of the images from that DataFrame to train the model. We have implemented a generator function that contains the above implementation. Few prediction examples are given below.

![alt text][image7]
![alt text][image8]
![alt text][image9]


**Result**

We reached IOU value of 93%. But that was not the final training step. We had to run 9 more iterations to achieve the best model.

![alt text][image10]

In a lot of cases the model could detect cars that were not annotated correctly in the data set. Below is an example,

![alt text][image11]

The model performs realtime prediction on the video. With my model I achieved 26FPS. For 50s video stream the model finishes it in 48 seconds.

Here is the link of the pipeline applied to a video. I have put a darker shade on the region of the video where I ran my prediction.


My model on project video.

https://youtu.be/m1iB_Pf3ORM

My model on challenge video.

https://youtu.be/fgAySk1MaDk

My model may not perform very well on the project videos. But check below its performance on a random video :-D

https://youtu.be/DXUVDZhQSKI

**Limitation**

My model doesn't do well with the shadowed regions. It tries to detect shadow as car. This biasness came from the data set. Heavy brightness augmentation should solve this problem.

**Further Improvement**

I would try to introduce much more augmentation like translation and stretching. I am also going to implement other models like YOLO, SSD etc.
