## Corn Disease Detection ##

### 1. Problem Statement - Build an image classifier for corn diseases. ###
 Steps:-
* Data loading 
* Data Visualisation, exploration, data preparation for modeling
* Model training and Evaluation


### 2. Process
### 2.1. Data
- Corn Disease Detection -  https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

### 2.2. Data Visualisation, exploration and prep for modeling
* Loading data using tensorflow functions into train and val sets, converting to batches
* Visualising data and understanding the nuances( diff sizes, quality,etc.)
* Investing the raw format of data and applying necessary fucntions to normalise it as required for model building.

### 2.3 Model training and Evaluation
* Model 1 - Using convolutional layers, max pool, dense layers
* Model 2 - Transfer Learning via Feature Selection method.
* Checkpointing using callback.
