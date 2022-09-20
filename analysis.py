# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation

data = pd.read_csv("data/data.csv",header=0)# here header 0 means the 0 th row is our coloumn 
                                                # header in data
"""
# have a look at the data
print(data.head(2))# as u can see our data have imported and having 33 columns
# head is used for to see top 5 by default I used 2 so it will print 2 rows
# If we will use print(data.tail(2))# it will print last 2 rows in data

data.info()

# now we can drop this column Unnamed: 32
data.drop("Unnamed: 32",axis=1,inplace=True) # in this process this will change in our data itself 
# if you want to save your old data then you can use below code
# data1=data.drop("Unnamed:32",axis=1)
# here axis 1 means we are droping the column

# here you can check the column has been droped
data.columns # this gives the column name which are persent in our data no Unnamed: 32 is not now there

# like this we also don't want the Id column for our analysis
data.drop("id",axis=1,inplace=True)

# As I said above the data can be divided into three parts.lets divied the features according to their category
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)

# lets now start with features_mean 
# now as ou know our diagnosis column is a object type so we can map it to integer value
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

print(data.describe()) # this will describe the all statistical function of our data
"""

# Assign y to the diagnosis column
y = data.diagnosis

# Assigning our index_col to be the column 'id' shifted our data over, leaving a column with all NaN entries.
# We drop that here
X = data.drop(columns=['Unnamed: 32'])

# Show all values whenever we call head.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# If we run .dtypes on our data frame, we notice that all columns, aside from the diagnosis being a string, our integers.

# We replace a malignant diagnosis with 1, and benign with 0
X['diagnosis'].replace('M', 1, inplace=True)
X['diagnosis'].replace('B', 0, inplace=True)
y.replace('M', 1, inplace=True)
y.replace('B', 0, inplace=True)

# Here, we use the seaborn correlation heatmap to visualize the correlatons of features in our dataset on one another.
# Using the filter method, we will drop features which have an absolute value of less than 0.5 on the feature 'diagnosis'


"""
# Setting up and displaying our heatmap correlation
plt.figure(figsize=(20,20))
cor = X.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')
plt.show()
"""

# First exploratory hypothesis: big and asymetric tumors have a greater probability to be malign
sns.relplot(data=data, x="symmetry_mean", y="area_mean", hue="diagnosis")
# plt.savefig('images/breast_cancer05.png')
plt.show()

