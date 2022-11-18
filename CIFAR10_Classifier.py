#!/usr/bin/env python
# coding: utf-8

# ### Optimized Toll Booths | Image Classifier
# > The purpose of this image classifier is to accurately distinquish between cars and trucks.   The below model is trained and test using the CIFAR10 dataset.

# 
# ##### Prepare Environment

# In[ ]:


###############################################################################
# Enable GPU (is applicable)
###############################################################################
gpu_info = get_ipython().getoutput('nvidia-smi')
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)


# In[ ]:


###############################################################################
# PIP Install 
###############################################################################
get_ipython().system('pip install -q -U keras-tuner')

###############################################################################
# Mount Google Drive (only run if using in colab)
###############################################################################
from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# ##### Import Packages

# In[ ]:


###############################################################################
# Imports for Data Analysis
###############################################################################
import numpy as np
from numpy.random import seed
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from tabulate import tabulate
###############################################################################
# Support of Plotting
###############################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import subplot_class_factory
###############################################################################
# Support of Deep Learning
###############################################################################
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.activations import relu, elu, softmax, tanh
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.engine.training import optimizers
from keras.layers import Dense, Dropout, ActivityRegularization
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPool2D, MaxPooling2D
from keras.layers.core.activation import Activation
from keras.layers.convolutional import Conv2D
from keras.regularizers import l1, l2, l1_l2
import keras_tuner
from keras_tuner import RandomSearch, HyperModel
from keras_tuner.tuners import Hyperband
from keras.utils import np_utils
###############################################################################
# Support of Supervised Learning, Tranformation, & Metrics
###############################################################################
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score,     precision_score, confusion_matrix, roc_auc_score, roc_curve


# #### Helper Functions

# In[ ]:


###############################################################################
# Load CIFAR10 Cars and Trucks
###############################################################################
def cifar10_cars_trucks():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    cars0 = 1; trucks1 = 9

    train_ind = np.where((y_train == cars0) | (y_train == trucks1))[0]
    test_ind = np.where((y_test == cars0) | (y_test == trucks1))[0]
    
    y_train = y_train[train_ind]
    X_train = X_train[train_ind]
    y_test = y_test[test_ind]
    X_test = X_test[test_ind]

    # Relabel car0 as 0 and truck1 as 1
    y_train[y_train == cars0] = 0
    y_train[y_train == trucks1] = 1
    y_test[y_test == cars0] = 0
    y_test[y_test == trucks1] = 1
    return( (X_train, y_train), (X_test, y_test) )

###############################################################################
# Min-Max Scaling
###############################################################################
def my_minmax(x):
    return( (x - np.min(x) / (np.max(x) - np.min(x))) )

###############################################################################
# Plot Image
###############################################################################
def plot_image(X: np.ndarray, idx: int=0):
    plt.imshow(X[idx].reshape(32, 32, 3));

###############################################################################
# Flattens tensor image for PCA and NMF processing
###############################################################################
def my_flattener(x):
    cnt, h, w, d = x.shape
    return x.reshape(cnt, h*w*d)

###############################################################################
# Dataclass for Processing Results
###############################################################################
@dataclass
class ClassifierResults:
  name: str
  y_true: np.ndarray = field(repr=False)
  preds: np.ndarray = field(repr=False)
  probs: np.ndarray = field(repr=False)

  def __repr__(self) -> None:
    rtn = f"Results for {self.name}\n" +           f"Accuracy:\t {self.accuracy()}\n"
    return(rtn)

  def accuracy(self) -> float:
    return (accuracy_score(self.y_true, self.preds))

###############################################################################
# Make Predictions
###############################################################################
def make_predictions(label, model, X_fitted, y_true):
  results = ClassifierResults(
      name = label,
      y_true = y_true,
      preds = model.predict(X_fitted),
      probs = model.predict_proba(X_fitted)
  )
  return (results)

###############################################################################
# Plot Feature Map
###############################################################################
def plot_feature_map(layer=0, n_col=8, n_row=4):
    plt.figure(figsize=(n_col, n_row))
    for j in range(n_row * n_col):
        plt.subplot(n_row, n_col, j + 1)
        plt.imshow(outputs[layer][0, :, :, j])
        plt.xticks(())
        plt.yticks(())
    plt.show;

###############################################################################
# Plot Image Map
###############################################################################
def plot_image_matrix(mod, n_col=10, n_row=10):
    plt.figure(figsize=(n_col, n_row)) 
    for j in range(n_col*n_row):
        plt.subplot(n_col, n_row, j + 1) 
        plt.imshow(mod[j].reshape(32, 32, 3)) 
        plt.xticks(())
        plt.yticks(())

###############################################################################
# Plot Model Results (loss) from History
###############################################################################
def plot_model_results(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show();

###############################################################################
# Plot Model Accuracy from History
###############################################################################
def plot_model_accuracy(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show();

###############################################################################
# Plot PCA Scree Plot
###############################################################################
def plt_pca_scree_plot(pca_mod: PCA):
    fig, ax = plt.subplots(1,2)
    fig.suptitle("Scree Plot")
    fig.set_size_inches(10, 5)
    fig.set_dpi(80)
    fig.tight_layout(pad=5.0)
    ax[0].plot(pca_mod.explained_variance_ratio_)
    ax[0].set_xlabel('number of components')
    ax[0].set_ylabel('ratio explained variance')
    ax[0].set_title("Ratio")
    ax[1].plot(np.cumsum(pca_mod.explained_variance_ratio_))
    ax[1].set_xlabel('number of components')
    ax[1].set_ylabel('cumulative explained variance')
    ax[1].set_title("Cumulative Sum ")
    fig.show();

###############################################################################
# Measure accuracy, recall, precision, fpr, fdr
###############################################################################
def get_metrics(y_true, y_preds):
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    accuracy = accuracy_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    roc = roc_curve(y_true,y_preds)
    fpr = fp / (fp + tn)
    fdr = fp / (fp + tp)
    return (accuracy, recall, precision, fpr, fdr)

###############################################################################
# Return Accuracy from Confusion Matrix
###############################################################################
def get_acc_from_conf(conf_mtx):
    acc = sum(conf_mtx.diagonal())/2000
    t_acc = conf_mtx.diagonal()[0]/1000
    c_acc = conf_mtx.diagonal()[1]/1000
    return acc, t_acc, c_acc


# ### Exploratory Data Analysis (EDA)

# #### Load Data
# > * Create a Training and Test using only the Cars and Trucks
# * Label Cars as 0 and Trucks as 1.

# In[ ]:


###############################################################################
# Load data into training and test
# Print Dimensions
###############################################################################
(X_train, y_train), (X_test, y_test) = cifar10_cars_trucks()
print(f"Dataset Dimensions:")
print(f"Train: X={X_train.shape}\ty={y_train.shape}")
print(f"Test : X={X_test.shape }\ty={y_test.shape}")


# **NOTE**:  The training set has 10,000 images of which 2,000 will be used as a validation set.  This will be done by setting validation_split=0.2.

# In[ ]:


###############################################################################
# Plot Car and Truck
###############################################################################
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(X_train[y_train.argmin()].reshape(32, 32, 3))
ax2.imshow(X_train[y_train.argmax()].reshape(32, 32, 3))
fig.show();


# #### EDA | Principal Component Analysis (PCA)
# > Explore the number of factors to achieve variance explained greater than 90%.

# In[ ]:


###############################################################################
# PCA Analysis
###############################################################################
X_train_pca = my_flattener(X_train.copy())
X_test_pca = my_flattener(X_test.copy())

scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
scaler.transform(X_test_pca)
# test validate mean == 0 and std == 1
assert np.isclose(0, X_train_pca.mean()) & np.isclose(1, X_train_pca.std())

pca = PCA(n_components=180, random_state=1842)
X_train_pca = pca.fit_transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)

plt_pca_scree_plot(pca)


# In[ ]:


###############################################################################
# PCA Model Results
###############################################################################
expl_diff = np.round(np.sum(pca.explained_variance_ratio_)*100, 2)
print(f"Top 180 components explain {expl_diff}% of the variance.")


# In[ ]:


###############################################################################
# Plot image matrix
###############################################################################
pca_loadings = minmax_scale(pca.components_, feature_range=(0,1), axis=1)
plot_image_matrix(pca_loadings)


# #### EDA | LDA-PCA Model
# > Using the PCA components, create an LDA model to test is ability to make predictions.

# In[ ]:


###############################################################################
# LDA-PCA Model 
###############################################################################
# set seed & create classifier
np.random.seed(1842)
lda_pca = LDA()

# tranform training and test
X_train_pca = np.dot(my_flattener(X_train), pca.components_.T)
X_test_pca = np.dot(my_flattener(X_test), pca.components_.T)

# fit model
lda_pca_fitted = lda_pca.fit(X_train_pca, y_train.reshape(10000))

## make predictions
train_pca_res = make_predictions("LDA-PCA | Train", lda_pca_fitted, X_train_pca, y_train)
test_pca_res = make_predictions("LDA-PCA | Test", lda_pca_fitted, X_test_pca, y_test)

# print results
print(train_pca_res)
print(test_pca_res)

print("Confusion Matrix | Test")
print(tabulate(confusion_matrix(test_pca_res.y_true, test_pca_res.preds), tablefmt='grid'))


# **NOTE:** The LDA-PCA model predicted better than 50% with an accuracy around 71%. 

# #### EDA | Non-Negative Matrix Factorization (NMF)
# > Explore and evaluate an NMF Transformation.

# In[ ]:


###############################################################################
# NMF Tranformation 
###############################################################################
X_train_nmf = my_flattener(X_train)
X_test_nmf = my_flattener(X_test)

nmf = NMF(n_components=100, random_state=1842, init='random', 
          max_iter=500, tol=5e-3).fit(X_train_nmf)
          
# nmf_W = nmf.fit_transform(X_train_nmf)
nmf_H = nmf.components_


# In[ ]:


###############################################################################
# Plot image matrix
###############################################################################
print(f"NMF Reconstruction Error:\t{round(nmf.reconstruction_err_, 2)}\n")
nmf_loadings = minmax_scale(nmf.components_, feature_range=(0,1), axis=1)
plot_image_matrix(nmf_loadings)


# #### EDA | LDA-NMF Model
# > Using the NMF loadings, create an LDA model to test predictive capabilities of reduced data.
# 

# In[ ]:


###############################################################################
# LDA-NMF Model for EDA
###############################################################################
# tranform training and test set of X
X_train_nmf = np.dot(my_flattener(X_train), nmf_H.T)
X_test_nmf = np.dot(my_flattener(X_test), nmf_H.T)

lda_nmf = LDA()
lda_nmf_fitted = lda_nmf.fit(X_train_nmf, y_train.reshape(10000))

train_nmf_res =     make_predictions("LDA-NMF | Train", lda_nmf_fitted, X_train_nmf, y_train)
test_nmf_res =     make_predictions("LDA-NMF | Test", lda_nmf_fitted, X_test_nmf, y_test)

print(train_nmf_res)
print(test_nmf_res)

print("Confusion Matrix | Test")
print(
    tabulate(confusion_matrix(test_nmf_res.y_true, test_nmf_res.preds), 
             tablefmt='grid'))


# **NOTE:** Performance of NMF model is approxiamately the same as the PCA model at 71%. 

# ##### EDA Summary / Conclusions
# The exploratory data analysis (EDA) showed the image prediction at 71% could be achieved using an LDA model along with dimension reduction technqiues such as PCA and NMF.  More specically, the PCA analysis showed that with 180 components that greater than 90% of the variance could be explained.  This insight will be useful in determining the initial design of the convolutional neural network (CNN) model.

# ### Modeling
# > 

# #### Modeling | Convolution Network (CNN)

# In[ ]:


###############################################################################
# CNN Modeling Preprocessing
###############################################################################
def preprocess_x(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    x = my_minmax(x)
    x = x.reshape(-1, 32, 32, 3)
    return(x)

def preprocess_y(y: np.ndarray) -> np.ndarray:
    return(np_utils.to_categorical(y))

X_train, X_test = [preprocess_x(X) for X in [X_train, X_test]]
y_train, y_test = [preprocess_y(y) for y in [y_train, y_test]]


# ##### Modeling | CNN | Model Prototype

# In[ ]:


###############################################################################
# CNN Hypermodel Prototype
###############################################################################
class CarTruckCNN(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()
        model.add(
            Conv2D(32, (3,3), activation="relu", input_shape=X_train.shape[1:]))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(rate=hp.Float(
            'dropout_1', min_value=0.0, max_value=0.5,default=0.25,step=0.05,)))
        model.add(Dropout(rate=0.05))
        model.add(Conv2D(32, (3,3)))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(rate=hp.Float(
            'dropout_2',min_value=0.0,max_value=0.5,default=0.25,step=0.05,)))
        model.add(Flatten())
        model.add(keras.layers.Dense(
            hp.Choice('units_1', [2048, 1024]), activation='relu'))
        model.add(Dense(hp.Choice('units_2', [512, 256]), activation="relu"))
        model.add(Dense(hp.Choice('units_3', [256, 128]), activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.compile(
            optimizer="adam", 
            loss="categorical_crossentropy", 
            metrics=["accuracy"])
        return(model)

###############################################################################
# CNN Hypermodel Prototype | Settings
###############################################################################
hband_dir = Path("/content/drive/MyDrive/Colab Notebooks/hyperband")
INPUT_SHAPE = X_train.shape[1:]
NUM_CLASSES = 2
HYPERBAND_MAX_EPOCHS = 30
SEED = 1842
EXECUTION_PER_TRIAL = 5

###############################################################################
# CNN Hypermodel Prototype | Tuning
###############################################################################
card_truck_hypermodel =     CarTruckCNN(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
tuner = Hyperband(
    card_truck_hypermodel,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    objective='accuracy',
    seed=SEED,
    executions_per_trial=EXECUTION_PER_TRIAL,
    directory=hband_dir,
    project_name='cifar10'
)
tuner.search_space_summary()


# In[ ]:


###############################################################################
# This code block is commented out on purpose, as it take about 2 1/2 hours
# to complete. However, the results of the search are outlined below.  
###############################################################################

# tuner.search(X_train, y_train, epochs=2, validation_split=0.2)

###############################################################################
# Best accuracy So Far: 0.9816499948501587
# Total elapsed time: 02h 29m 12s 
###############################################################################

# best_model = tuner.get_best_models(num_models=1)[0]
# best_model.evaluate(X_test, y_test)

###############################################################################
# loss: 0.5597 - accuracy: 0.8515
# [0.5596567988395691, 0.8514999747276306]
###############################################################################


# Hypermodel prototype trained for accuracy, produced and accurate but overfit model.   This model was scaled down the the base model, which was
# finally optimized.

# ##### Modeling | CNN | Base Model

# In[ ]:


###############################################################################
# CNN Base Model
###############################################################################
def build_base_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation="relu", 
                     input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=2, activation="sigmoid"))

    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"])
    return(model)

car_truck_model_base = build_base_model()
car_truck_model_base.summary()


# In[ ]:


###############################################################################
# CNN Base Model | Fit and Evaluate Resutls
###############################################################################
base_model_hist_base = car_truck_model_base.fit(
    x=X_train, 
    y=y_train, 
    epochs=100, 
    verbose="auto", 
    use_multiprocessing=True, 
    validation_split=0.2
    )

_, accuracy = car_truck_model_base.evaluate(x=X_test, y=y_test)
print(f"Base Model Accuracy:  {accuracy}")

plot_model_accuracy(base_model_hist_base)


# In[ ]:


plot_model_results(base_model_hist_base)


# In[ ]:


my_img = X_test[0].astype(int)
layer_outputs = [layer.output for layer in car_truck_model_base.layers]
layers_model = keras.Model(inputs=car_truck_model_base.input, outputs=layer_outputs)
outputs = layers_model.predict(my_img.reshape(1, 32, 32, 3))


# In[ ]:


print("Feature Map, Layer 0")
plot_feature_map(layer=0, n_col=8, n_row=4)


# ##### Modeling | CNN | Optimized Model

# In[ ]:


###############################################################################
# CNN Optimized Model
###############################################################################
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation="relu", 
                     input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.35000000000000003))
    model.add(Conv2D(16, (3,3) ))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.35000000000000003))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=2, activation="sigmoid"))

    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"])
    return(model)

car_truck_model = build_model()
car_truck_model.summary()


# In[ ]:


###############################################################################
# CNN Optimized Model | Fit and Evaluate Results
###############################################################################
seed(1842)
early_stop = EarlyStopping(patience=5, monitor='accuracy')

model_hist = car_truck_model.fit(
    x=X_train, 
    y=y_train, 
    epochs=100, 
    verbose="auto", 
    use_multiprocessing=True, 
    validation_split=0.2,
    callbacks=[early_stop]
    )


# In[ ]:


###############################################################################
# Print and plot optimized model accuracy
###############################################################################
opt_loss, accuracy = car_truck_model.evaluate(x=X_test, y=y_test)

print(f"\nOptimized Model Accuracy:  {np.round(accuracy, 4)}")
plot_model_accuracy(model_hist)


# In[ ]:


###############################################################################
# Plot optimized model loss
###############################################################################
print(f"\nOptimized Model Loss:  {np.round(opt_loss, 4)}")
plot_model_results(model_hist)


# #### Optimized Model | Image Layer Analysis
# > Plot feature maps across different layers with the intent to interpret the model results.

# In[ ]:


plot_image(X_test.astype(int), idx=0)


# In[ ]:


my_img = X_test[0].astype(int)
layer_outputs = [layer.output for layer in car_truck_model.layers]
layers_model = keras.Model(inputs=car_truck_model.input, outputs=layer_outputs)
outputs = layers_model.predict(my_img.reshape(1, 32, 32, 3))


# In[ ]:


print("Feature Map, Layer 0")
plot_feature_map(layer=0, n_col=8, n_row=4)


# In[ ]:


print("Feature Map, Layer 1")
plot_feature_map(layer=1, n_col=8, n_row=4)


# In[ ]:


print("Feature Map, Layer 2")
plot_feature_map(layer=2, n_col=8, n_row=4)


# In[ ]:


print("Feature Map, Layer 3")
plot_feature_map(layer=3, n_col=8, n_row=2)


# In[ ]:


print("Feature Map, Layer 4")
plot_feature_map(layer=4, n_col=8, n_row=2)


# Comparison of Models (Metrics)

# In[ ]:


###############################################################################
# Create Confusion Matrix
###############################################################################

# Base Model 
y_preds_base = car_truck_model_base.predict(X_test).argmax(axis=1)
y_test_base = y_test.argmax(axis=1)
car_truck_conf_mtx_base =     confusion_matrix(y_true=y_test_base, y_pred=y_preds_base)

# Optimized Model
y_preds_car_truck = car_truck_model.predict(X_test).argmax(axis=1)
y_test_car_truck = y_test.argmax(axis=1)
car_truck_conf_mtx =     confusion_matrix(y_true=y_test_car_truck, y_pred=y_preds_car_truck)


print("\nConfusion Matrix")
print("Base Model")
print(tabulate(car_truck_conf_mtx_base, tablefmt='grid'))
print("")
print("Optimized Model")
print(tabulate(car_truck_conf_mtx, tablefmt='grid'))


# In[ ]:


###############################################################################
# Accuracy
###############################################################################
acc_b = get_acc_from_conf(car_truck_conf_mtx_base)
acc_o = get_acc_from_conf(car_truck_conf_mtx)

print(f"Accuracy:")
print(f"Base:       Overall: {acc_b[0]}\tTruck: {acc_b[1]}\tCar: {acc_b[2]}")
print(f"Optimized:  Overall: {acc_o[0]}\tTruck: {acc_o[1]}\tCar: {acc_o[2]}")


# In[ ]:


###############################################################################
# Accuracy, Recall, Precision, FPR, FDR
###############################################################################
model_metrics_dict = {
    'Metrics': ['Accuracy', 'Recall', 'Precision','FPR', 'FDR'],
    'Base Model': 
        np.round(get_metrics(y_true=y_test_base, y_preds=y_preds_base), 4),
    'Optimized Model': 
        np.round(
            get_metrics(y_true=y_test_car_truck, y_preds=y_preds_car_truck), 4)
    }

model_metric_tbl = tabulate(
    model_metrics_dict,
     tablefmt='grid',
     headers=["Metrics","Base Model", "Optimized Model"]
    )

print(model_metric_tbl)

