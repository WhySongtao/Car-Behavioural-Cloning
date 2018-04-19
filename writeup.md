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

[image1]: ./Model_Summary.jpeg "Model Visualization"

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

### Model Architecture and hyperparameter

#### 1. The final model architecture

My model consists of 
1. A Convolution layer with 5x5 kernel size and depths of 6 and relu activation function.
2. A MaxPooling layer with strides 2 and 2x2 kernel size.
3. A Convolution layer with 5x5 kernel size and depths of 16 and relu activation function.
4. A MaxPooling layer with strides 2 and 2x2 kernel size.
5. A Convolution layer with 3x3 kernel size and depths of 32 and relu activation function.
6. A MaxPooling layer with strides 2 and 2x2 kernel size.
7. A Convolution layer with 3x3 kernel size and depths of 64 and relu activation function.
8. A MaxPooling layer with strides 2 and 2x2 kernel size.
9. A Dense layer with 1024 neurons and relu activation function.
10. A Dropout layer with 0.5 keep_prob
11. A Dense layer with 512 neurons and relu activation function.
12. A Dropout layer with 0.5 keep_prob
13. A Dense layer with 1 neurons to predict the steering angle.

![Model summary][image1]


#### 2. Attempts to reduce overfitting in the model

For the dataset, data augmentation has been used:
1. The left and right camera image are used to help model generalise better. 
2. More data are collected by running the car backwards in the simulator to get more different data.
3. The training data and their corresponding angel are also fliped to get more different data which can help the model to generalise.
For the model:
1. The model contains dropout layers following fully-connected layers in order to reduce overfitting. 
2. The training data are shuffled before fed into the model.

#### 3. Model parameter tuning

1. The model used an adam optimizer, so the learning rate was not tuned manually. 
2. The epoch has been set to 40 because the training error seems to plateau after 40 epochs. 
3. Batch size is 32.
4. The model used mean square error as loss funciton. I have experimented with mean absolute error as loss function but the result seems to be worse than using mean square error as loss function.

#### 4. Appropriate training data

The training data used are fairely simple. It contains images from keeping the vehicle driving on the center of the road, both by driving forwards and in the opposite direction. There are 10,924 images including both three cameras.

I then split the dataset so that training set contains 80% of the data and validation set contains 20% of the data. 


### Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design a simple model first, then based on the training error and validation error as well as the result of the test run in the simulator, iterate through different methods accordingly to address specific problems. 

My first step was to use a convolution neural network model similar to the AlexNet. I chose this because this is a simple and effective model to start with. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had both high mean squared error on the training set and a high mean squared error on the validation set. Also the car drove off the track at some sharp turns. To combat this problem, I recorded a recovery video teaching the car how to drive back to the centre when it is about to wander off the track. I also added one more convolutional layer and maxpooling layer to add more complexity to the model.   
On the second iteration, the error on training set was reduced but still had a high validation error. The car was able to drive through track 1. So to solve this overfitting problem, I tried batch normalisation layer and dropout layer. 

On the thrid iteration, I found that dropout layer seems to give a better results so I removed batch normalisation layer and added dropout layer between the fully-connected layer. However, the car started to drove off the track again. Being a bit confused with the results, I decided to re-take the recording again without any recovery recording. 

On the fourth iteration, surprsingly, the vehicle was able to drive autonomously around the track again. It seems that more training data doesn't guarantee a better results. My guess behind this is that because the car can only drive as good as the recording, the results will largely depend on the quality of the recording.

