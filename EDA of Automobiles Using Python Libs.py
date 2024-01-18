"""During the course of this case study, I will be covering a wide range of topics including Data Exploration, Cleaning, Investigation and forming inferences that'll help you gain a deeper understanding of this fascinating dataset. 

The automobile dataset contains information on various attributes of cars, such as their make, model, fuel type, engine displacement,
horsepower, and price. The dataset includes a total of 205 instances, with 26 different attributes."""

# Here is the dataset that we will be using:
# https://drive.google.com/file/d/13pCKXi7EAbXUWTwvi6xWYS9V2AfsCQHK/view?usp=sharing
# download this file and save it on google drive

import pandas as pd
import numpy as np

#mounting your drive, so that you can access the files there
#you'll receive a authentication prompt. Complete it.
from google.colab import drive
drive.mount('/content/drive')

automobile="drive/My Drive/automobile.csv"
auto_df=pd.read_csv(automobile)


"""To get a list of all the column names in the automobile dataset, you can use the command auto_df.columns.
This would provide you with the names of all the columns in the dataset and enable you to carry out various
data manipulation and analysis operations."""
print(auto_df.columns)


"""Dataset Cleaning 🧹
Handling Missing Values
When data is missing from a dataset, it may be represented by other values such as '?' question marks, '-' dashes or blank spaces.
These values can create problems during data analysis as they cannot be recognized as missing data by most software packages and can 
interfere with calculations and statistics.
By converting these values to NaN (Not a Number), we can standardize the representation of missing data and make it easier to handle 
and process during data analysis. NaN values are recognized by most software packages as missing data and can be easily removed, 
imputed or handled in other ways during data analysis. NaN stands for "Not a Number". It is a special floating-point value in 
Python that represents an undefined or unrepresentable value. In the context of datasets, NaN values typically represent 
missing or undefined values."""
#  Handle missing values
auto_df.replace('?', np.nan, inplace=True) # replace '?' with NaN

# drop rows with missing values in the price column
auto_df.dropna(subset=['price'], axis=0, inplace=True)

# replace missing values in normalized-losses column with mean value
auto_df['normalized-losses'].fillna(auto_df['normalized-losses'].astype(float).mean(), inplace=True)

print(auto_df.dtypes)

# Convert data types from object to float
auto_df[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']] = auto_df[['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']].astype(float)
print(auto_df.dtypes)

# Removes duplicate rows from the auto_df dataframe object
auto_df.drop_duplicates(inplace=True)


"""OUTLIERS: Outliers are like those weirdos who always stand out in a crowd.
They are data points that are significantly different from other points in the dataset,
and they can really mess up your analysis if you don't handle them properly."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a box plot of the price feature
sns.boxplot(x="price", data=auto_df)
plt.show()
"""The outliers are points you see on either sides of the whiskers."""

# Dealing with outliers
auto_df = auto_df[(auto_df['price'] >= auto_df['price'].quantile(0.05)) & (auto_df['price'] <= auto_df['price'].quantile(0.95))]

"""The auto_df dataframe is filtered to remove observations whose price is below the 5th percentile or above the 95th percentile 
of all price values in the dataset. This means that values that are considered as extreme or unusual are removed from the dataset."""


"""The plt.hist() function generates a histogram by taking in an array of values and dividing them into a specified number of 
bins (in this case, 20 bins) based on the range of the data. The resulting histogram shows the distribution of the data,
with the x-axis representing the range of prices and the y-axis representing the frequency of prices falling into each bin."""

# Histogram of prices
plt.hist(auto_df['price'], bins=20)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Car Prices')
plt.show()

"""The plt.scatter() function generates a scatter plot by taking in two arrays of data, one for each axis of the plot.
In this example, the x-axis represents engine size and the y-axis represents horsepower. Each data point is plotted as a point on the graph,
with the x and y values determining its position."""
# Scatterplot of engine size vs. horsepower
plt.scatter(auto_df['engine-size'], auto_df['horsepower'])
plt.xlabel('Engine Size')
plt.ylabel('Horsepower')
plt.title('Engine Size vs. Horsepower')
plt.show()


"""The sns.boxplot() function generates a box plot by taking in a DataFrame (data) and specifying the variables to be plotted on 
the x-axis (x) and y-axis (y). In this example, the x-axis represents body-style and the y-axis represents price. The resulting 
plot shows the distribution of prices for each body style, with a box representing the interquartile range (IQR) of the data, 
whiskers extending to the most extreme data points within 1.5 times the IQR, and any data points
outside this range plotted as individual points or outliers."""
# Boxplot of price vs. body-style:
sns.boxplot(x = 'body-style', y='price', data=auto_df)
plt.show()


"""The sns.scatterplot() function generates a scatter plot by taking in a DataFrame (data) and specifying the variables 
to be plotted on the x-axis (x) and y-axis (y). In this example, the x-axis represents horsepower and the y-axis represents price. 
The hue parameter is used to specify a categorical variable (fuel-type) to be used for coloring the data points.
Each unique value of the fuel-type variable is assigned a different color, allowing for easy visual comparison of the data across
different categories."""
# Scatterplot of horsepower vs. price colored by fuel-type:
sns.scatterplot(x='horsepower', y='price', hue = 'fuel-type', data= auto_df)
plt.show()


"""The sns.histplot() function generates a histogram by taking in a DataFrame (data) and specifying the variable to be plotted 
on the x-axis (x). In this case, the x-axis represents city-mpg."""
# Histogram of city-mpg:
sns.histplot(x='city-mpg',data=auto_df)
plt.show()


"""Data Inspection and Visualization:"""
"""1. How does the fuel economy (city-mpg and highway-mpg) vary between different body styles and fuel types? Which body style
and fuel type have the highest and lowest fuel economy?"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create boxplots of city-mpg and highway-mpg for each body style and fuel type
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
sns.boxplot(x='body-style', y='city-mpg', hue='fuel-type', data=auto_df)
plt.title('City MPG by Body Style and Fuel Type')

plt.subplot(2,1,2)
sns.boxplot(x='body-style', y='highway-mpg', hue='fuel-type', data=auto_df)
plt.title("Highway MPG by Body Style and Fuel Type")
plt.tight_layout()
plt.show()


"""2. Is there a correlation between engine size and horsepower? How does engine size and horsepower vary across different makes and models?"""
# Create a scatter plot of engine size vs. horsepower, with different colors for different makes
plt.figure(figsize=(10,6))
sns.scatterplot(x='engine-size', y='horsepower', hue='make', data=auto_df)
plt.title('Engine Size vs. Horsepower by Make')
plt.xlabel('Engine Size')
plt.ylabel('Horsepower')
plt.show()


"""3. How does the length, width, and height of the car vary between different body styles and makes? Which make and body style
has the largest and smallest dimensions?"""
# Calculate the average dimensions for each body style and make
dims= ['length','width','height']
body_style_means = auto_df.groupby('body-style')[dims].mean()
make_means = auto_df.groupby('make')[dims].mean()

# Create bar plots of the average dimensions for each body style and make
body_style_means.plot(kind='bar', rot=0)
plt.title('Average Dimensions by Body Style')
plt.xlabel('Body Style')
plt.ylabel('Average Dimension')
plt.show()

make_means.plot(kind='bar', rot=90)
plt.title('Average Dimension by Make')
plt.xlabel('Make')
plt.ylabel('Average Dimension')
plt.show()


"""4.How does the price of the car vary with respect to its drivetrain (4WD or 2WD)? Is there a difference in price between cars with 4WD and 2WD?"""
# Create a box plot of car prices by drivetrain type
sns.boxplot(x='drive-wheels', y='price', data=auto_df)
plt.title('Car Prices by Drivetrain Type')
plt.xlabel('Drivetrain Type')
plt.ylabel('Price')
plt.show()

"""Following code calculates the mean and standard deviation of car prices for each drivetrain type using the mean() and std() functions from pandas."""
# Calculate the mean and standard deviation of car prices for each drivetrain type
price_4wd = auto_df[auto_df['drive-wheels']=='4wd']['price']
price_2wd = auto_df[auto_df['drive-wheels']=='2wd']['price']
mean_4wd = price_4wd.mean()
mean_2wd = price_2wd.mean()
std_4wd = price_4wd.std()
std_2wd = price_2wd.std()
# Print the mean and standard deviation of car prices for each drivetrain type
print('Mean price (4WD):', mean_4wd)
print('Standard deviation (4wd):', std_4wd)
print('Mean price (2WD):', mean_2wd)
print('Standard deviation (2WD):',std_2wd)


"""5. Is there a correlation between curb weight and fuel economy (city-mpg and highway-mpg)? How does curb weight vary across 
different body styles and makes?"""
# Create a scatterplot of curb weight vs. city-mpg and curb weight vs. highway-mpg
sns.scatterplot(x = 'curb-weight', y= 'city-mpg', data=auto_df)
plt.title('Curb Weight vs. City MPG')
plt.xlabel('Curb Weight')
plt.ylabel('City MPG')
plt.show()

sns.scatterplot(x='curb-weight',y='highway-mpg', data=auto_df)
plt.title('Curb Weight vs. Highway MPG')
plt.xlabel('Curb Weight')
plt.ylabel('Highway MPG')
plt.show()

# Calculate the correlation coefficients
corr_city = auto_df['curb-weight'].corr(auto_df['city-mpg'])
corr_highway = auto_df['curb-weight'].corr(auto_df['highway-mpg'])
print('Correlation coefficient (Curb Weight vs. City MPG):', corr_city)
print('Correlation coefficient (Curb Weight vs. Highway MPG):', corr_highway)


"""To explore how curb weight varies across different body styles and makes, we can use descriptive statistics and box plots.
Here is an example code:"""
# Create a box plot of curb weight by body style and make
sns.boxplot(x = 'body-style', y='curb-weight',data= auto_df)
plt.title('Curb Weight by Body Style')
plt.xlabel('Body Style')
plt.ylabel('Curb Weight')
plt.show()

sns.boxplot(x='make',y='curb-weight',data= auto_df)
plt.title('Curb Weight by Make')
plt.xlabel('Make')
plt.ylabel('Curb Weight')
plt.xticks(rotation=90)
plt.show()

# Calculate the mean and standard deviation of curb weight for each body style and make
curb_weight_by_body_style =auto_df.groupby('body-style')['curb-weight']
curb_weight_by_make = auto_df.groupby('make')['curb-weight']
mean_curb_weight_by_body_style = curb_weight_by_body_style.mean()