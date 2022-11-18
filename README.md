# vehicle_classification
Binary classifier to accurately distinguish between cars and trucks using CIFAR10 data set. The CIFAR10 dataset consists of 60,000 images - 12,000 of which contains images of cars and trucks. 

Each picture is stored as a 32x32 image, represented in 3 colors (red, green, and blue). Considering each pixel in one image as a data point, a single image would have over 3000 features. While high dimensional data like this can be informative, using the images as is may increase model run time and lead to overfitting in the model. 

The two methods of dimension reduction explored in this project consists of Non-negative matrix factorization (NMF) and Principal Component Analysis (PCA). NMF highlights specific regions between images that could represent a grouping of features while PCA presents generic combinations of pixels that can be used to generalize patterns of information. 

To build a model that can predict whether an image is a car or truck, a Convolution Neural Network (CNN) model was built. I chose a CNN model as it is ideal for image recognition and the models tend to be flexible in what it can be trained to learn. Additionally, it's extendable, meaning it will be able to learn new classes should the need arise to expand the model, such as identifying motorcylces or any other vehicle type. 

The metrics being used to measure how the model performs is on accuracy, as I care about how well the model will classify both cars and trucks correctly. 

The first iteration of the CNN model, referenced as the base model, was nearly 79% accurate. This is fairly good for a model, but the loss was near 4, which would be considered high. With this base model in mind, my goals for optimizing the model were to maximize accuracy while also minimizing loss.

After including some hidden layers, pooling layers, and other features within the optimized model, the model was able to reach around 89% accuracy and only about a .3 loss, which is signifcantly better than the base model. When looking closer into the optimized CNN, the accuracy of the model classifying an image as a car is 80%, while the accuracy of the model classifying an image as a truck is 93.8%. 
