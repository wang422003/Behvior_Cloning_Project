
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

[image1]: ./images/cnn-architecture.png "Model Visualization"
[image2]: ./images/image2.jpg "Central Image"
[image3]: ./images/right1.jpg "Recovery Image"
[image4]: ./images/right2.jpg "Recovery Image"
[image5]: ./images/right3.jpg "Recovery Image"
[image6]: ./images/flip1.jpg "Normal Image"
[image7]: ./images/flip2.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* Save.py for driving the car in autonomous mode
* modeltry.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py modeltry.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network from Nvidia.(code line 75-84)

The data is normalized in the model using a Keras lambda layer (code line 72). 


#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And after that using flip to get double training data from the original one.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get accruate enough regression on steering angle to control the vehicle.

My first step was to use a convolution neural network model similar to the LeNET. I thought this model might be appropriate because the LeNet is widely spread.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had  pretty high errors on both training and validation sets. This implied that the model was underfitting. 

To combat the underfitting, I added the normalizing and flipping methods to the preprocessing part. But after the training, the vehicle seemed to be unstable for safe driving.

Then I tried the net from Nvidia, which is also introduced in the courses. And the mean square errors on both training and validation are small.

The final step was to run the simulator to see how well the car was driving around track one. Although the vehicle only ran at 9 mph, but it was able to hold the track without leaving the road. 


#### 2. Final Model Architecture

The final model architecture (model.py lines 75-84) consisted of a convolution neural network with the following layers and layer sizes:

Convolution2D(24,5,5)

Convolution2D(36,5,5)

Convolution2D(48,5,5)

Convolution2D(64, 3, 3)

Convolution2D(64, 3, 3)

Flatten()

Dense(100)

Dense(50)

Dense(10)

Dense(1)


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive to the center if it is close to the left or right sides. These images show what a recovery looks like starting from the right side :

![alt text][image3]

![alt text][image4]

![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would get anther training dataset from the opposite direction of the original training data. For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]


After the collection process, I had X number of data points. I then preprocessed this data by normalizing and cropping it to focus on the necessary part of the images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
