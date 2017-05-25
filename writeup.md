# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./before_flip.jpg "Image Before Flip"
[image9]: ./after_flip.jpg "Image After Flip"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* video.mp4 is a video recording autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 5 convolutional layers, followed by 4 fully connected layers (line 45 ~57 in model.py). The data is passed through the network after cropping and normalization.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 46).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 55, 57, 59).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 63). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used images from all three cameras provided in this project. For the left and right camera images, an artificial steering angle bias of +/-0.2 has been employed to steer the vehicle back to the track.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to leverage the code pieces from the lesson.

My first step was to use a convolution neural network model similar to the one provided in the lesson. I thought this model might be appropriate because everything looks fine on lesson videos.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding dropout layers after every fully connected layer so that the loss of training and validation set became closer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. It turns out again to be the overfitting issue, so I modified the dropout parameters several times until it can pass first track.

At the end of the process, the vehicle is able to drive autonomously around the first track without leaving the road.

However, no matter how hard I tried, the 2nd track never passed. The car can't even stay in center for long. I think it's because the training data of 2nd track is less than 1st track.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
* Normalization
* Cropping
* Convolution2D(24x5x5, with subsample 2x2)
* Convolution2D(36x5x5, with subsample 2x2)
* Convolution2D(48x5x5, with subsample 2x2)
* Convolution2D(64x3x3, with subsample 1x1)
* Convolution2D(64x3x3, with subsample 1x1)
* Dropout 0.5
* Fully connected layer 100
* Fully connected layer 50
* Fully connected layer 10
* Fully connected layer 1

#### 3. Creation of the Training Set & Training Process

I was trying to collect training data by playing with the simulator. However, I found myself is such a horrible gamer that I always drive into lake or hit the edges. I believe I can generate lots of useful data for recovery from car accidents - in the future.

As a result, I apply the training data provided by Udacity. As mentioned in lesson, I add more data by flipping the images horizontally and multiply all command data with -1 to balance left and right direction, make sure there won't be bias in training data.

<br>sample image before flip <br>
![alt text][image8]

<br>sample image after flip <br>
![alt text][image9]

On the other hand, I used images from all three cameras provided in this project. For the left and right camera images, an artificial steering angle bias of +/-0.2 has been employed to steer the vehicle back to the track.

After the above process, I had 38,568 of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by the loss starting to diverge between training and validation set after epoch=8. I used an adam optimizer so that manually training the learning rate wasn't necessary.
