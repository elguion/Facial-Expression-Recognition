**Facial Expression Recognition**

Used Convolutional neural networks (CNN) for facial expression recognition . The goal is to classify each facial image into one of the seven facial emotion categories considered .

Data :
We trained and tested our models on the data set from the Kaggle Facial Expression Recognition Challenge, which comprises 48-by-48-pixel grayscale images of human faces,each labeled with one of 7 emotion categories: anger, disgust, fear, happiness, sadness, surprise, and neutral .

Image set of 35,887 examples, with training-set : dev-set: test-set as 80 : 10 : 10 .

**Dependencies**
Python 2.7, sklearn, numpy, Keras.

Library Used:
Keras
Sklearn
numpy
Getting started
To run the code -

Download FER2013 dataset from Kaggle Facial Expression Recognition Challenge and extract in the main folder.

To run deep CNN model. Open terminal and navigate to the project folder and run cnn_major.py file

python cnn_major.py
No need to train the model , already trained weights saved in model4layer_2_2_pool.h5 file.
Want to train model yourself ?
Just change the statement

   is_model_saved = True
   // to
   is_model_saved = False
 
Shallow CNN Model
Code Link - cnn_major_shallow
Model Structure- Link
Saved model trained weights - Link

Deep CNN Model
Code Link - cnn_major
Model Structure- Link
Saved model trained weights - Link

Model Training:
Shallow Convolutional Neural Network
First we built a shallow CNN. This network had two convolutional layers and one FC layer.

First convolutional layer, we had 32 3×3 filters, along with batch normalization and dropout and max-pooling with a filter size 2×2.

Second convolutional layer, we had 64 3×3 filters, along with batch normalization and dropout and max-pooling with a filter size 2×2.

In the FC layer, we had a hidden layer with 512 neurons and Softmax as the loss function.

Deep Convolutional Neural Networks
To improve accuracy we used deeper CNN . This network had 4 convolutional layers and with 2 FC layer.

Model Evaluation:
Model predicts softmax output for 7 label for an image

[  4.99624775e-07   3.69855790e-08   9.91190791e-01   8.15907307e-03  2.62175627e-06   9.97206644e-06   1.02341000e-03]
which is converted to
[ 2 ]
label having highest value . For evaluation , categorial accuracy is used .

Some Experiment are done by changing number of layers and changing hyper-parameters.

Accuracy Achieved :
Shallow CNN -- 56.31%

Deep-CNN -- 65.55%
References
"Dataset: Facial Emotion Recognition (FER2013)" ICML 2013 Workshop in Challenges in Representation Learning, June 21 in Atlanta, GA.

"Convolutional Neural Networks for Facial Expression Recognition" Convolutional Neural Networks for Facial Expression Recognition Shima Alizadeh, Azar Fazel

"Andrej Karpathy's Convolutional Neural Networks (CNNs / ConvNets)" Convolutional Neural Networks for Visual Recognition (CS231n), Stanford University.
