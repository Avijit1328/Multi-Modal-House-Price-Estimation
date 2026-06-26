import os
import cv2
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import concatenate, Input, Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
sns.set(font_scale=1.3)

table_dir = 'data/HousesInfo.txt'
table = pd.read_csv(table_dir, header=None, sep=' ', names=['bedrooms', 'bathrooms', 'area', 'zipcode', 'price'])

table.shape

table.head()

info = pd.DataFrame(index=table.columns, columns=['Dtype', 'Unique Numbers'])
info['Dtype'] = table.dtypes
info['Unique Numbers'] = table.nunique()
info['Description'] = ['Number of bedrooms', 'Number of bathrooms', 'Area of the house', 'Zipcode', 'House price']

info

print('Duplicated Values:', table.duplicated().sum())

print('Null Values:', table.isna().sum().sum())

info['Dtype'].value_counts()

images_dir = 'data/Houses Dataset'
image_paths = sorted(list(glob.glob(os.path.sep.join([images_dir, "{}_*".format(1)]))))
fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'hspace': -0.3})
for i, ax in enumerate(axes.flatten()):
    if i < len(image_paths):
        image = cv2.imread(image_paths[i])[..., ::-1]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(os.path.basename(image_paths[i]).replace('.jpg', '').replace('1_', '').title())
plt.show()

def img_read(size=256):
    images = []
    for i in table.index.values:
        basepath = os.path.sep.join(['data/Houses Dataset', "{}_*".format(i + 1)]).replace('\\', '/')
        imgPaths = sorted(list(glob.glob(basepath)))
        inputImages = []
        for imgPath in imgPaths:
            image = cv2.resize(cv2.imread(imgPath), (size // 2, size // 2))
            inputImages.append(image)
        outputImage = np.zeros((size, size, 3), dtype='uint8')
        outputImage[0:size // 2, 0:size // 2] = inputImages[0]
        outputImage[0:size // 2, size // 2:size] = inputImages[1]
        outputImage[size // 2:size, size // 2:size] = inputImages[2]
        outputImage[size // 2:size, 0:size // 2] = inputImages[3]
        images.append(outputImage)
    images = np.array(images)
    return images

images = img_read()

images.shape

plt.imshow(images[0][..., ::-1])
plt.axis('off')
plt.show()

table['total rooms'] = table['bedrooms'] + table['bathrooms']

table['total rooms'] = table['total rooms'] + 1

table['avg_room_area'] = table['area'] // table['total rooms']

bins = [0, 2000, 4000, 6000, 8000, float('inf')]
labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
table['area category'] = pd.cut(table['area'], bins=bins, labels=labels, right=False)

table.drop('zipcode', axis=1, inplace=True)

table.head()

table.shape

def SIFT(image, plot=True):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    if plot:
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Image with Keypoints')
        plt.axis('off')
        plt.show()
    else:
        mean_array = np.mean(descriptors, axis=0)
        mean_array = mean_array.ravel()
        return mean_array

SIFT(images[0])

features = []
for img in images:
    feature = SIFT(img, plot=False)
    features.append(feature)

len(features)

features[0].shape

sift_features = np.vstack(features)

sift_features.shape

table.describe().T.drop(['count', 'std'], axis=1)

table['price'].hist(bins=20)
plt.title('Price (Target)')
plt.show()

table[['area']].hist()
plt.show()

table['total rooms'].hist()
plt.title('Total Rooms')
plt.show()

table['avg_room_area'].hist()
plt.title('Average Room Area')
plt.show()

plt.figure(figsize=(10, 6))
table['bathrooms'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Number of Bathrooms')
plt.xlabel('Num')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
table['bedrooms'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Number of Bedrooms')
plt.xlabel('Num')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 4))
table['area category'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Area Categories Distribution')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()

def price_vs_col(col):
    plt.scatter(table[col], table['price'])
    plt.title('Price vs ' + col.title().replace('_', ' '))
    plt.xlabel(col.title().replace('_', ' '))
    plt.ylabel('Price')
    plt.show()

price_vs_col('bedrooms')

price_vs_col('bathrooms')

price_vs_col('total rooms')

price_vs_col('area')

price_vs_col('avg_room_area')

plt.figure(figsize=(8, 6))
sns.boxplot(x='area category', y='price', data=table)
plt.title('Price vs Area Category')
plt.xlabel('Area Category')
plt.ylabel('Price')
plt.show()

correlation_matrix = table.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

correlation_matrix['price'].sort_values(ascending=False)[1:]

table['area category'] = table['area category'].map({'Very Small': 1, 'Small': 2, 'Medium': 3,
                                                     'Large': 4, 'Very Large': 5})

X = table.drop('price', axis=1)
y = table['price']

X[['area', 'avg_room_area']] = np.log1p(X[['area', 'avg_room_area']])

y = np.log1p(y)

print(X.shape)
print(y.shape)

sift_features.shape

X = np.hstack([sift_features, X])
X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classic_results = {}

def linear_regression():
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mae, r2)

classic_results['Linear Regression'] = linear_regression()

classic_results

def poly_regression():
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mae, r2)

classic_results['Polynomial Regression'] = poly_regression()

classic_results

def ridge_regression():
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mae, r2)

classic_results['Ridge Regression'] = ridge_regression()

classic_results

def dtree_regression():
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mae, r2)

classic_results['DTree Regression'] = dtree_regression()

classic_results

models = list(classic_results.keys())
first_values = [value[0] for value in classic_results.values()]
second_values = [value[1] for value in classic_results.values()]
sorted_models_first = [model for _, model in sorted(zip(first_values, models))]
sorted_models_second = [model for _, model in sorted(zip(second_values, models))]

plt.figure(figsize=(7, 4))
plt.bar(sorted_models_first, sorted(first_values), color='skyblue', label='MAE')
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error')
plt.title('Comparison of MAE for Classic Models')
plt.xticks(rotation=45)
plt.legend()
plt.show()

plt.figure(figsize=(7, 4))
plt.bar(sorted_models_second, sorted(second_values), color='skyblue', label='R2')
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('Comparison of R2 Score for Classic Models')
plt.xticks(rotation=45)
plt.legend()
plt.show()

advanced_results = {}

def rf_regression():
    model = RandomForestRegressor(n_estimators=50, random_state=90)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mae, r2)

advanced_results['Random Forest Regressor'] = rf_regression()

advanced_results

def svr():
    model = SVR(kernel='rbf', C=20, epsilon=0.5, degree=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mae, r2)

advanced_results['Support Vector Regressor'] = svr()

advanced_results

def xgboost():
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=8,
                             learning_rate=0.01, random_state=14, eval_metric='mae')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mae, r2)

advanced_results['XGBoost Regressor'] = xgboost()

advanced_results

def catboost():
    model = CatBoostRegressor(iterations=100, depth=8, eval_metric='MAE', random_state=10, verbose=0, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mae, r2)

advanced_results['CatBoost Regressor'] = catboost()

advanced_results

models = list(advanced_results.keys())
first_values = [value[0] for value in advanced_results.values()]
second_values = [value[1] for value in advanced_results.values()]
sorted_models_first = [model for _, model in sorted(zip(first_values, models))]
sorted_models_second = [model for _, model in sorted(zip(second_values, models))]

plt.figure(figsize=(7, 4))
plt.bar(sorted_models_first, sorted(first_values), color='skyblue', label='MAE')
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error')
plt.title('Comparison of MAE for Advanced Models')
plt.xticks(rotation=45)
plt.legend()
plt.show()

plt.figure(figsize=(7, 4))
plt.bar(sorted_models_second, sorted(second_values), color='skyblue', label='R2')
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('Comparison of R2 Score for Advanced Models')
plt.xticks(rotation=45)
plt.legend()
plt.show()

data = table.copy()

data.head()

images.shape

data = data[['bedrooms', 'bathrooms', 'area', 'price']]

nn_images = img_read(size=64)

X_train, X_test, IX_train, IX_test = train_test_split(data, nn_images, random_state=5)

max_price = data['price'].max()
y_train = (X_train['price'] / max_price).values
y_test = (X_test['price'] / max_price).values

ss = StandardScaler()
X_train = ss.fit_transform(X_train[['bedrooms', 'bathrooms', 'area']])
X_test = ss.transform(X_test[['bedrooms', 'bathrooms', 'area']])

print(IX_train.shape)
print(y_train.shape)

print(IX_test.shape)
print(y_test.shape)

# CNN
input_shape = (64, 64, 3)
input1 = Input(shape=input_shape)
conv1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input1)
batchnorm1 = BatchNormalization(axis=-1)(conv1)
maxpool1 = MaxPooling2D(pool_size=(2, 2))(batchnorm1)

conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(maxpool1)
batchnorm2 = BatchNormalization(axis=-1)(conv2)
maxpool2 = MaxPooling2D(pool_size=(2, 2))(batchnorm2)

conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')(maxpool2)
batchnorm3 = BatchNormalization(axis=-1)(conv3)
maxpool3 = MaxPooling2D(pool_size=(2, 2))(batchnorm3)

flatten1 = Flatten()(maxpool3)
dense1 = Dense(32, activation='relu')(flatten1)
batchnorm4 = BatchNormalization(axis=-1)(dense1)
dropout1 = Dropout(0.5)(batchnorm4)
dense2 = Dense(16, activation='relu')(dropout1)
m1 = Model(input1, dense2)

# MLP
m2 = Sequential()
m2.add(Dense(16, activation='relu', input_dim=3))
m2.add(Dropout(0.5))
m2.add(Dense(8, activation='relu'))
m2.add(Dense(4, activation='relu'))

# Combination
combinedModel = concatenate([m2.output, m1.output])

# Last layers
densef1 = Dense(8, activation='relu')(combinedModel)
densef2 = Dense(1, activation='linear')(densef1)
model = Model(inputs=[m2.input, m1.input], outputs=densef2)

model.summary()

plot_model(model, show_shapes=True)

model.compile(loss='mean_absolute_error', optimizer=Adam(0.01))

history = model.fit([X_train, IX_train], y_train,
                    validation_data=([X_test, IX_test], y_test), epochs=84, batch_size=10)

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='val_Loss')
plt.title('Loss Function Evolution')
plt.legend()
plt.show()

predicts = model.predict([X_test, IX_test]) * max_price

r2_score(y_pred=predicts.flatten(), y_true=(y_test * max_price))

