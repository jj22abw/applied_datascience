
# Importing essential libraries
import numpy as num 
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

# Ignore the warnings
import warnings
warnings.filterwarnings('ignore')

# import the dataset
df_1=pd.read_csv("C://3930cf78-211b-44ff-a4c0-deaeab2d2f2a_Data.csv")
df_1.dropna(inplace=True)

# Null value checking
print(df_1.isnull().sum())
df_1 = df_1.fillna(0)
df_1.shape
df_1.describe()
df_1.info()
from sklearn.preprocessing import LabelEncoder
X = df_1
y = df_1['Series Name']

# using lebel encoder convert object into numeric
le = LabelEncoder()
X['Series Name'] = le.fit_transform(X['Series Name'])
X
y
# Initiating K-means clustering
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot
X, y = make_blobs(n_samples=400, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
kmeans_clustering = KMeans(n_clusters=4, init='k-means++', max_iter=400, n_init=10, random_state=0)
pred_y = kmeans_clustering.fit_predict(X)
plot.scatter(X[:,0], X[:,1], c=pred_y, cmap='plasma')
plot.scatter(kmeans_clustering.cluster_centers_[:, 0], kmeans_clustering.cluster_centers_[:, 1], s=300, c='red')
plot.xlabel('X')
plot.ylabel('Y')
plot.title('Curve between predictions and number of clusters')
plot.show()

# taking sample data
data_sample = [51230, 49950, 49770, 50119, 50210, 50295.2, 50300, 50370, 50470]
keys=['2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']

# Plotting the pyplot
palette_color = sns.color_palette('hsv')
plot.pie(data_sample, labels=keys, colors=palette_color,
        autopct='%.0f%%')
plot.show()

years= [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
values = [47.8, 46.5, 45.2, 43.9, 42.6, 41.3, 40.1, 40.1, 40.1, 40.1]
plot.bar(years, values, color='cyan')
plot.xlabel('Years')
plot.ylabel('Values')
plot.title('Values by 10-year Intervals')
plot.show()

data_sample_1 = {'x' : ['Afganisthan','Colombia', 'Argentina', 'Austraila', "Belgium"],
                 'y' : [380100, 482428.255, 1083817.597, 3624770, 13646.093]}
df_2= pd.DataFrame(data_sample_1)
plot.figure(figsize=(20, 10))
plot.bar(data_sample_1['x'], data_sample_1['y'], color = 'magenta')
plot.yscale("log")
plot.title("comparing CO2 emissions between 5 countries in the year 2019")
plot.show()

# Initiating the curve_fit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
def exp_growth(x, a, b):
    return a * np.exp(b * x)
X_1 = np.linspace(0, 10, 50)
y_1 = exp_growth(X_1, 2, 0.5) + np.random.normal(0, 0.2, 50)

def err_ranges(params, covariance, sigma): 
    stdevs = np.sqrt(np.diag(covariance))
    lower = params - sigma * stdevs
    upper = params + sigma * stdevs
    return lower, upper
params, covariance = curve_fit(exp_growth, X_1, y_1)
lower, upper = err_ranges(params, covariance, 2)
plt.scatter(X_1, y_1)
plt.plot(X_1, exp_growth(X_1, params[0], params[1]), 'r-')
plt.fill_between(X_1, exp_growth(X_1, lower[0], lower[1]), exp_growth(X_1, upper[0], upper[1]), color='red', alpha=1)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

columns_to_drop = ['Series Name', 'Series Code', 'Country Name', 'Country Code']
df_3 = df_1.drop(columns_to_drop, axis=1)
df_3 = df_3.replace(['..', 'nan'], [0, 0])
df_3 = df_3.fillna(0)
df_3.info()

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_3[['2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]']], df_1['2015 [YR2015]'], test_size=0.2)
# create the logistic regression model
clf = LogisticRegression()
# fit the model on the training data
clf.fit(X_train, y_train)
# make predictions on the test data
y_pred = clf.predict(X_test)
# calculate the accuracy of the model
accuracy = clf.score(X_test, y_test)
print('Accuracy: ', accuracy)

