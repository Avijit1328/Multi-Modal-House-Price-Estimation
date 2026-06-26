import os
import cv2
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def SIFT(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    if descriptors is not None:
        mean_array = np.mean(descriptors, axis=0)
        return mean_array.ravel()
    else:
        return np.zeros(128)

def img_read(table, size=256):
    images = []
    for i in table.index.values:
        basepath = f"Houses Dataset/{i + 1}_*"
        imgPaths = sorted(list(glob.glob(basepath)))
        inputImages = []
        for imgPath in imgPaths:
            image = cv2.resize(cv2.imread(imgPath), (size // 2, size // 2))
            # OpenCV reads in BGR, let's keep BGR for SIFT extraction like the notebook
            inputImages.append(image)
            
        outputImage = np.zeros((size, size, 3), dtype='uint8')
        if len(inputImages) == 4:
            outputImage[0:size // 2, 0:size // 2] = inputImages[0]
            outputImage[0:size // 2, size // 2:size] = inputImages[1]
            outputImage[size // 2:size, size // 2:size] = inputImages[2]
            outputImage[size // 2:size, 0:size // 2] = inputImages[3]
        images.append(outputImage)
    return np.array(images)

def main():
    print("Loading textual dataset...")
    table = pd.read_csv('Houses Dataset/HousesInfo.txt', header=None, sep=' ', names=['bedrooms', 'bathrooms', 'area', 'zipcode', 'price'])
    
    # We will use the base 3 textual features to keep the UI simple
    X_text = table[['bedrooms', 'bathrooms', 'area']].values
    y = table['price'].values
    
    print("Loading and processing images...")
    images = img_read(table, size=256)
    
    print("Extracting SIFT Visual Features...")
    sift_features = []
    for img in images:
        feature = SIFT(img)
        sift_features.append(feature)
    sift_features = np.vstack(sift_features)
    
    # Combine SIFT + Textual Features
    X_final = np.hstack([sift_features, X_text])
    
    # Split data into 80% training and 20% testing just to see how good it is
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    
    print("Evaluating Random Forest Regressor Model...")
    # Create the model with 50 trees
    model = RandomForestRegressor(n_estimators=50, random_state=90, n_jobs=-1)
    
    # Train the model on the training data for validation
    model.fit(X_train, y_train)
    
    # Test the model to see how well it did
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Validation Mean Absolute Error (80% split): ${mae:,.2f}")
    
    print("Training FINAL model on 100% of the dataset for deployment...")
    final_model = RandomForestRegressor(n_estimators=50, random_state=90, n_jobs=-1)
    final_model.fit(X_final, y)
    
    # Save the final trained model to a folder so app.py can use it later
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    joblib.dump(final_model, 'saved_models/rf_model.pkl')
    print("Successfully trained on FULL dataset and saved model to saved_models/rf_model.pkl")

if __name__ == '__main__':
    main()
