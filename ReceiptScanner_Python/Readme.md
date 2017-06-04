[//]: # (Image References)
[image1]: ./readmeImages/Augmentation.png
[image2]: ./readmeImages/Augmentation1.png
[image3]: ./readmeImages/Augmentation2.png
[image4]: ./readmeImages/Augmentation3.png
[image5]: ./readmeImages/Unet.png
[image6]: ./readmeImages/Model.PNG
[image7]: ./readmeImages/IOU.JPG
[image8]: ./readmeImages/PostProcess1.png
[image9]: ./readmeImages/PostProcess2.png
[image10]: ./readmeImages/PostProcess3.png
[image11]: ./readmeImages/Epochs.PNG
[image12]: ./readmeImages/result1.png
[image13]: ./readmeImages/result2.png
[image14]: ./readmeImages/result3.png

***Receipt Scanner***

The goals / steps of this project are the following:

In this project our goal is to create a pipeline to detect expense receipt in an image and remove its background. One way to achieve this is to use traditional computer vision techniques like canny edge detection, color/gradient threshold etc. But in this project we applied state of the art deep learning based approach for receipt detection.


***Overview***

We used spatial color features of a receipt image and trained a Convolutional Neural Network to do pixel wise prediction to detect the receipt and remove the background. This technique is much more robust than the traditional computer vision approach and generalize much better for unseen data.

The model I trained to achieve the goal is inspired from an famous CNN model called U-net. This model directly predicts a mask in an image. It has a special architecture. It first implements Convolutional layers which shrink the width and height and increases the depth of the feature map and then it implements merging layers which concatenates the Convolutional layers through an up-convolving process to finally achieve an image mask which contains the pixel wise prediction values to detect a receipt. This approach is inspired by a [prize winning submission on Kaggle ultrasound Nerve Segmentation competition challenge.](https://github.com/jocicmarko/ultrasound-nerve-segmentation)

***Data Preparation***

I have load, process, trained and tested the model in the Receipt_Scanner.ipynb jupyter notebook file.

In the data set each JPG image has a corresponding PNG image which contains the receipt shape masked on it. Steps to prepare the data are following:

1) I have used a very small data set of only 25 images to train the model. This images in the data set has 500x500 resolution. To fit on the model we had to resize each image to 512x512 resolution.

2) As the data set is very small, I had to augment it heavily to produce more image that resembles real life scenarios. I have employed randomly translation, rotation, shearing, stretching and brightness augmentation on the data set to create a very large additional data set. I have also used combination of the these augmentation applied on the image to maximize the chance of creating new training image. We apply same augmentation except brightness augmentation on the image mask as well. This is because the augmentation that changes the shape of the original image should be same for the mask as well. Otherwise the mask will not match the shape on the augmented image. The augmentation call flow starts from the function *augmentImage* written on the 7th cell of the notebook.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]


***Target Dataset***

From the mask images we know the shape of the receipt in the images. We load the mask as a numpy array of 512x512x1 dimension which gives us our image target mask. Refer to the *TrainDataGenerator* function at the 9th code cell of the Jupyter notebook.

***Model***

I didn't implement the full Unet model as it takes a lot of time to train. I have implemented a bit smaller size of it which also takes 20 minutes to  train on 15K images. The original Unet model architecture is the following,

![alt text][image5]

I had to play around with the model a lot to come up with a decent mini model. First I tried without the 512 and 256 filter layers. The model did ok with this one. It was detecting a lot of false positives. So I had to add the 256 filter feature extraction layer. With that layer I achieved the best model. Please refer to the function *CreateModel()* at the 10th cell of the notebook.

![alt text][image6]

But still the model had false positives. It was performing on the training set very well but not so well on the test set. So I decided to add regularization on it. After applying l2 regularization the prediction became a lot better.

***Training***

The hyper parameters are,

Training samples per epoch = 1500
Training Batch Size = 30
Resize Image Row = 512
Resize Image Column = 512
Learning Rate = .0001
EPOCH = 40

We could not fit more than 30 batch size of images because of the limitation of GPU memory(8GB). With learning rate .00001 the model performed very poorly. We have used sigmoid activation function for the last layer because we wanted a pixel wise probability of being a receipt. Also we have used IOU(intersection of union) also known as [dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) for loss function and accuracy matrix. This is simply taking 2 times intersection of two images meaning (pixel wise value multiplication of two images and summation)x2 divided by union of two images meaning summation of all the pixel values in two images. Please refer to the function *dice_coef()* in the notebook. Visualization is given below,

![alt text][image7]

***PostProcessing***

We have trained the model for 40 Epochs at first. Then we applied regularization on it and trained it on another 10 Epochs. We did post processing after we get the predicted mask. We saw that there are some small regions which were detected as receipt on images but are not. So to eliminate those regions we applied **erosion** and **dilation** morphing techniques on them. Using rectangular kernel after applying erosion many small regions gets disappear or get considerably small. But it also erodes the correctly detected receipt mask. So we applied dilation on it again. Please refer to the 14th cell of the notebook for implementation of the morphing techniques.

After that we applied another post processing technique. We saw that there are disconnected regions that are detected as receipt. The actual detected receipt is the largest connected region on the prediction. So we applied a flood fill algorithm using **Breadth First Search**. This way we detect the largest connected region on the predicted mask and eliminate others. Please refer to the 13th cell of the notebook. Some example of before post processing and after post processing are given below,

![alt text][image8]
![alt text][image9]
![alt text][image10]

***Result***

We reached IOU value of ~80% at 40 iterations. But that was not the final training step. We had to run 10 more iterations to achieve the best model which had 90% IOU. Below is the details of the last few iterations,

![alt text][image11]

Some final prediction result with the background removed are following,

![alt text][image12]
![alt text][image13]
![alt text][image14]

Without the PostProcessing the model takes few hundred milliseconds to complete the prediction and remove the background. After applying postprocessing it takes like 2-3 seconds to complete the process.

***Limitation***

Although I applied augmentation heavily still the augmented images are very related. For this reason there are false positive predictions exists with the model. To achieve much better accuracy more data set is needed. I believe with a large data set this model can achieve state of the art prediction accuracy on receipt detection.
