***Behavrioal Cloning Project***

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/ModelArchitecture.PNG "Model Visualization"
[image2]: ./images/center.jpg "Center camera image"
[image3]: ./images/left.jpg "Left camera image"
[image4]: ./images/right.jpg "Right camera image"
[image5]: ./images/distribution.png "Data distribution"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* Readme.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model employs a convolution neural network model with depth ranges from 24 to 128. And filter size ranges from 3x3 to 5x5.

The model includes ELU layers to introduce nonlinearity, and the data is normalized between -0.5 to +0.5. in the model using a Keras lambda layer (code line 226).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers on all fully connected layers in order to reduce overfitting. L2 regularization is also applied to all the fully connected layers.  

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 274-275). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so at first we didn't tune the learning rate. But after much experiment we saw that adam optimizer is making the model converge much faster to a local minima. So we changed the learning rate to .0001 instead and found the optimal solution (code line 271).

####4. Appropriate training data

I have used Udacity provided training data as it was not possible with a mouse and keyboard to generate suitable data with both the beta and main simulator provided by Udacity.

I have used images and angles from all the 3 (left, right, center) cameras. The details on this subject will be discussed in the later sections.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one employed by [Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). But with much data augmentation I could not make it generate an optimum solution. Seems like its always overfitting and can't generate a good model on the data. I kept tuning hyper paramters like image size, samples per epoch, translation range etc without success.

One problematic part of this project is that all the data generated from track 1 are quite similar. So if you want to generate your validation data from track1 then its useless. Because validation data should be unseen data. So the validation loss was always lower than the training loss and I could not get any wiser from that. Looking at MSE alone won't help and can only lead me to astray.

To compensate this I started to save weights of all the epochs. Then manually test for all the epochs to see which one performs best. For this reason I kept my number of samples per epoch pretty small (2560 samples per epoch). This gave me more weights.h5 files to test on. And I noticed that the models starts to overfit after seeing all the training data 1/2 times.  

After facing the above scenario I became confident that I need regularization. So I have added dropout on all the layers including the convolution layers as well. The performance was much better but still it was touching the yellow line in few places.

So after that I made the model more deeper, from 64 to 128. And removed dropouts from the convolution layers as I thought they might be hampering the feature extraction. But I kept regularization on all the fully connected layers. Also I increased the neurons sizes on the fully connected layers.

At the end of the process, the vehicle is able to drive autonomously around both of the track without leaving the road.

####2. Final Model Architecture

My model consists of a convolution neural network with 6 convolution layers and 5 fully connected layers. First 3 layers has 5x5 filter sizes and others has 3x3 filter sizes. It has a depths between 24 and 128 (model.py lines 219-267). It has ~567K trainable parameters.

Here is a visualization of the architecture.

![][image1]

####3. Creation of the Training Set & Training Process

The data set contains images from 3 cameras. Left, right and center cameras. Here is an example image of center camera:

![alt text][image2]

The other two cameras which are left and right camera, gives us recovery data. We can think them as center camera image when the car is close to the lane. Below are example of left and right cameras respectively,

![alt text][image3]
![alt text][image4]

To augment the data sat, I adjusted brightness of the image. I have randomly scaled the brightness between 0.3 to 1 range of the image.  I also flipped images and angles thinking that this would generate more data which is exactly as if driving from the opposite direction. I have also translated the images as much as 25 pixels on positive or negative side horizontally and compensate the driving angle by 0.2 on every pixel translated.

I tried to create a balanced training set which has equal number of positive and negative angle data. I also included zero angle data in the training set but only one tenth of the total number. Because there are much more zero angles than the positive or negative data. So if we include all zero angles then the data will be highly biased to zero angle prediction. Below is the histogram where I visualize my data distribution. All though its not perfectly balanced but it did the trick.

![alt text][image5]

After the collection process, I had around 100K number of data points. I then preprocessed this data by cropping 20% of the top portion of the image and 12.5% of bottom portion of the image to remove the hood of the car and blue sky or other artifacts. Then I resized the image to 64x64 size. I have also transformed the image to HSV format.

I finally randomly shuffled the data set which will be used for training only. Also I have another data set which were manually generated for validation purpose. The same preprocessing is done for this validation data but no augmentation.

I used this training data for training the model. Although I don't think validation set is doing anything useful in this project I kept it. The ideal number of epochs was 19 where training loss was .0650. The training loss goes down to .02 on 40 epochs but it had overfitted. The car was able to drive on both tracks successfully with usual and full speed (by changing throttle=.3 in drive.py). Below is the link of the video of the car running on two tracks.

https://youtu.be/F7v7l6YEFjo
