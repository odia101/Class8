from sklearn.datasets import load_diabetes
diabetes_data = load_diabetes()

dir(diabetes_data)
### Now, I want to list the properties of the dataset, and what they each "mean
# TODO:
# 'DESCR': this is a dataset description
# data: numpy array, in the shape row x columns
#feature_names: feature labels for dataset
#target: numpy array, 1x rows, containing the thing we are trying to predict
#target_names: name of the class we are predicting in the categorical case

#TODO:
# 1. First, visualize your data from loading it *this way*
#	This can include converting it to pandas, and running your script from before
# 2. train and apply either a KNN-classifier or KNN-regressor on your dataset
# 3. print the output "score" or "performanve" of the classifier/regressor
