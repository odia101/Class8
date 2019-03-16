#!/user/bin/env python


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier

#******************************
#Loading Dataset from sklearn
#******************************

from sklearn.datasets import load_diabetes
#data = load_diabetes()
#df = pd.DataFrame(data=data["data"], columns=data["feature_names"])
diabetes = load_diabetes()
df = pd.DataFrame(data = np.c_[diabetes['data'], 
diabetes['target']],columns=diabetes['feature_names']+['target'])
#print (df)

#*****************************************
# This session  Accesses Rows and Columns
#*****************************************

#print("\n","Rows 3 and 4","\n",df.iloc[3:5,:]) #Access Row
#print("\n", "Column 3" "\n", df.iloc[:,3]) #Access Column

#************************************************************************
# This session computes Summary Statistics (Mean and Standard Deviation)
#************************************************************************

#print("\n","Mean of Features","\n",np.mean(df,axis=0))
#print("\n")
#print("Standard Deviation of Features", "\n", np.std(df,axis=0))

# This session plots Histogram of each feature (column)

#for i, column in enumerate(df.columns):
#       plt.figure(i)
#       sns.distplot(df[column])
#       plt.savefig('histogram_''{0}.pdf'.format(column))
#plt.show()
#plt.close()

#***********************************************************************
# This session plot scatterplot for every pair combinantion of features
#***********************************************************************

#for i, column1 in enumerate(df.columns):
#	for j, column2 in enumerate (df.columns[i+1:]):
               #print (column1, column2)
#		data1 = df[column1]
#		data2 = df[column2]
#		plt.figure()
#		sns.scatterplot(data1, data2)
#		plt.title("Scatter Plot Between {} and {}".format(column1,column2))
#		plt.savefig("Scatterplot_{}_{}.pdf".format(column1, column2))
#		plt.clf()
#plt.show()
#plt.close()

#********************************************************************
# This session plots scatterplot for every 4 combination of features
#********************************************************************

#for pair in itertools.combinations((df.columns), 4): #Computes Columns Combination 4
#	data1 = df[list(pair)]
#	data2 = pd.DataFrame(data1)
#	print (data2)
#	sns.pairplot(data2, kind='reg')
#	plt.title("PairPlot between {}".format(pair))
#	plt.savefig("Pairplot_{}.pdf".format(pair))
#plt.show()
#plt.close()

#*************************************************************
# Perform Classification/Regression and print the performance
#*************************************************************

#Plotting Pairplot for all features
sns.pairplot(df)
plt.title("Pairplot for all features")
plt.savefig("Pair_plot_Diabetes.pdf")
plt.show()
plt.close()

#Plotting Histogram of Target feature
sns.distplot(df['target'])
plt.title("Histogram for Target Feature(Y)")
plt.savefig("Histogram_Target.pdf")
plt.show()


#Plotting Heatmap for all features
df1=df.corr() #To get correlation from data
sns.heatmap (df1)
plt.title("Heatmap for Diabetes Dataset")
plt.savefig("Heatmap_Diabetes.pdf")
plt.show()

#*********************************
#Training a Linear Regression Model
#*********************************
X = df[['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']] # Split up data into X
y = df['target']						# and y

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=101)
from sklearn import linear_model
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                 normalize=False)
print("\n", "Linear Regression Coefficients","\n",lm.coef_)
# The mean square error
print ("Mean Square Error")
print (np.mean((lm.predict(X_test) - y_test)**2))
print("The score")
print(lm.score(X_test, y_test))
#Score of 1: Perfect Prediction
#Score of 0: No linear relationship

sns.regplot(y_test,predictions)
plt.xlabel('Expected')
plt.ylabel('Predicted')
plt.title("Linear Regression Model: Prediction Plot")
plt.savefig("Linear_Regression_plot.pdf")
plt.show()

#**************************************************
# Classification/Regression for Features Importance
#**************************************************
def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 10
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

#*************************
# 1. Decision Tree  Model
#*************************
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("\n","Decision Tree Results")
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
print("Feature importances:\n{}".format(tree.feature_importances_))
plot_feature_importances_diabetes(tree)
plt.title("Decision Tree: Feature Importance Plot") 
plt.savefig("Decision_Tree.pdf")
plt.show()

#****************************
# 2. Random Forest Model
#****************************

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("\n","Random Forest Results")
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))
print("Feature importances:\n{}".format(rf.feature_importances_))
plot_feature_importances_diabetes(rf)
plt.title("Random Forest: Feature Importance Plot")
plt.savefig("Random_Forest.pdf")
plt.show()

#***************************
# 3. Gradient Boosting Model
#***************************

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gb.fit(X_train, y_train)
print("\n","Gradient Boosting Results")
print("Accuracy on training set: {:.3f}".format(gb.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gb.score(X_test, y_test)))
print("Feature importances:\n{}".format(gb.feature_importances_))
plot_feature_importances_diabetes(gb)
plt.title("Gradient Boosting: Feature Importance Plot")
plt.savefig("Gradient_Boosting.pdf")
plt.show()
