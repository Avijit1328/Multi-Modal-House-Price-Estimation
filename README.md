# Multi-Modal House Price Estimation 🏡

This project is a machine learning application designed to estimate the price of a house using both **Textual Features** (like the number of bedrooms, bathrooms, and area) and **Visual Features** (four images of the house: Front, Bedroom, Bathroom, and Kitchen).

This was developed as a college project to demonstrate multi-modal data processing and machine learning deployment using a modern web interface.

## How It Works

Instead of just relying on numerical data, this project combines computer vision and tabular data:
1. **Visual Features**: We use the **Scale-Invariant Feature Transform (SIFT)** algorithm via OpenCV to extract visual keypoints and descriptors from the 4 house images. 
2. **Textual Features**: We take the numerical inputs (Bedrooms, Bathrooms, Area in sq ft).
3. **Machine Learning**: We combine both sets of features and feed them into a **Random Forest Regressor** model to predict the final house price in USD (and converts it to INR).

## Project Structure
- `app.py`: The main Streamlit web application containing the user interface and inference logic.
- `train_rf.py`: The training script that reads the dataset, extracts SIFT features, trains the Random Forest model on 100% of the data, and saves it.
- `requirements.txt`: The list of Python libraries needed to run the project.
- `saved_models/`: Contains the trained `rf_model.pkl` machine learning model file.
- `Houses Dataset/`: The folder containing the images and the `HousesInfo.txt` dataset.

## Installation

1. Make sure you have Python installed on your system.
2. Open your terminal or command prompt in this folder.
3. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Web App

To launch the web interface, run the following command in your terminal:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser (usually at `http://localhost:8501`). You can type in the house details, upload the 4 images, and click "Estimate Price" to see the prediction!

## How to Retrain the Model

If you add new data or images to the `Houses Dataset` folder, you can easily retrain the model from scratch by running:
```bash
python train_rf.py
```
This script will automatically extract the new features, train a new Random Forest model, and save it to the `saved_models/` folder for the web app to use.
