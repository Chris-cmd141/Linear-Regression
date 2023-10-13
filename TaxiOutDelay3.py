import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from sklearn.metrics import accuracy_score  # Import accuracy_score


# Load the dataset
data = pd.read_csv(r'C:\BOOTCAMP\MACHINE LEARNING\M1_final.csv') ## I'm storing the data set from the CSV file M1_final into a Panda dataframe that I call "data"

# Explore the dataset
print(data.head())

# Exploratory data analysis
# Calculate the correlation matrix for numeric columns
corr_mat = data.select_dtypes(include=[np.number]).corr() ## data.select_dtypes - filters columns based on data type and the argument to include only numbers (including floats)

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(15, 12))
sns.heatmap(corr_mat, annot=True, fmt=".2f") # "annot=True" displays corelation values in the boxes with 2 decimals (fmt=".2f")
plt.show()

# Check the correlation values between variables 
correlation_AB = corr_mat.loc['TAXI_OUT', 'MONTH'] ## corr_mat is the matrix, the heatmap. ".loc" selects the specific corelation "box" that I want to display the value for
print(f'Correlation between Taxi Out and Month: {correlation_AB:.2f}')


# Select the independent (x) and dependent (y) variables
x = data['DAY_OF_WEEK'].values.reshape(-1, 1) ## VALUES - we need this atribute to convert the column in a NumPy array so the machine learning can understand it
y = data['TAXI_OUT'].values.reshape(-1, 1) ## RESHAPE (-1,1) - organizes these values so that each value gets its own row (vertically) in a table-like format. "Make a table with one column, and adjust the number of rows to fit the data"

# Create a Linear Regression model
LR = LinearRegression() ## we are telling Python to initialise the LinearRegression model

# Train the Linear Regression model
LR.fit(x, y) ## tells Python to learn a linear relationship between the independent variable(s) represented by x and the dependent variable represented by y

# Make predictions based on the trained model
y_pred = LR.predict(x) ## tells Python to make the predictions and store them in the variable y_pred

# Create a dataframe for predicted and actual values
df = pd.DataFrame({'Actual': y.flatten(), 'Predicted': y_pred.flatten()})
## we first converted  the Panda data frame into a NumPy array by using .values.reshape(-1,1) in order for the machine learning to be able to compute and then we've flatten it back to Panda so we can visualise the results 
print(df)

# Plot a graph
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.xlabel('Day of the Week')
plt.ylabel('Taxi Out Delay')
plt.show()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32) ##test_size=0.20: This specifies that you want to allocate 20% of your data to the testing set and the remaining 80% to the training set.
## "random_state=32" it's telling Python the starting point every time you run the train_test_split code. Like always starting from lvl 32 in a game. *or for ex. shufling cards 32 times, needs to be consistent. If you change this number to 70 for ex. it can influence to consistency fo the model 

# Standardize the training data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# Train the model with standardized data
LR.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = LR.predict(x_test)

# Create a dataframe for prediction
prediction_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(prediction_df)

# New data point to make predictions
new_data_point = np.array([[4]])  # 4 is the day of the week, for ex for Thursday. If I want Wednesday prediction I would put 3.
# Make predictions using Linear Regression
predicted_taxi_out = LR.predict(new_data_point)
print(f"Predicted TAXI_OUT for the new data point: {predicted_taxi_out[0]}") ## 0 is an index here. There's only one index here, one value

# Evaluate the model
print('MAE', metrics.mean_absolute_error(y_test, y_pred))
print('MSE', metrics.mean_squared_error(y_test, y_pred))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2', metrics.r2_score(y_test, y_pred))

# Define models with regularization
model_Lasso = Lasso(alpha=0.10) ## low Alpha allows the model to use more of the variables that can lead to a more complex model. (like salt in cooking, if I lower Alpha I tell the model use as much as you want, if I increase I tell him to use only a little)
model_ridge = Ridge(alpha=10) ## this is set to high, simple model with less complexity 

# Train the Ridge and Lasso models
model_ridge.fit(x, y)
model_Lasso.fit(x, y)

# Make predictions using Ridge and Lasso models
lasso_prediction = model_Lasso.predict(new_data_point)
ridge_prediction = model_ridge.predict(new_data_point)

print('Lasso Prediction:', lasso_prediction)
print('Ridge Prediction:', ridge_prediction)

# Tune alpha for Ridge model
cv = RepeatedKFold(n_splits=2, n_repeats=1, random_state=32)
model = RidgeCV(alphas=(0.5, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
model.fit(x, y)

# Print the chosen alpha
print('Chosen Ridge Alpha:', model.alpha_)

# Create Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(x_train, y_train.ravel())  # Gaussian Naive Bayes does not accept 2D arrays
y_pred_gnb = gnb.predict(x_test)

# Calculate accuracy for Gaussian Naive Bayes
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print('Gaussian Naive Bayes Accuracy:', accuracy_gnb)