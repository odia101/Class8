This work involves loading diabetes dataset from scikit-learn, visualizing features in histogram, pairplot, and 
scatterplot. Several regression/classification algorithms are also performed.

#*******************
#Running the script
#*******************

python newscript.py

#********************
#Question Asked?
#********************

What features are most important for diabetes patients disease progression (outcome or target feature)?
To answer the above question, the following classification/refression algorithms were performed on the diabetes. 

*Linear Regression

*Decision Tree

*Random Forest

*Gradient Boosting

#***************************
#PSEUDOCODE
#***************************


Load Data

Train Regression/Classification Model

	(X=Features and y=target)

Train Test Split

	(X_train, X_test, y_train, y_test = train_test_split)

Create and Train Model

def plot_feature_imortance_diabetes(model)
	plot(bar chart)

	For Linear Regression
		Print(Mean Square Error)
		Print(Score)
		Scatter Plot(y_test, prediction)
	
	For Decision Tree
		Print(Accuracy on Training set
		Print(Accuracy on test set)
		Plot(Feature Importance)

	For Random Forest
		Print(Accuracy on Training set
		Print(Accuracy on test set)
		Plot(Feature Importance)
		
	For Gradient Boosting
		Print(Accuracy on Training set
		Print(Accuracy on test set)
		Plot(Feature Importance)
	





