# Movie-Rating-Prediction
A machine learning model to predict IMDb movie ratings using feature engineering, preprocessing, and regression models. 

#  Movie Rating Prediction  

##  Overview  
This project aims to **predict IMDb movie ratings** based on various attributes such as **genre, director, budget, runtime, and past performance metrics**.  
By leveraging **machine learning techniques**, we preprocess data, engineer relevant features, train models, and evaluate their performance.  

---

##  Task Objectives  
- Build a **regression model** to predict IMDb movie ratings.  
- Perform **data preprocessing** (handle missing values, encode categorical features).  
- Engineer new features (e.g., **director success rate, genre popularity**).  
- Train and evaluate models using metrics like **MAE, MSE, and R² score**.  
- Structure the project with proper **documentation and a clean GitHub repository**.  

---

##  Project Structure 
Movie-Rating-Prediction/  
│── data/                        # Folder for raw and processed datasets  
│   ├── IMDb_Movies_India.csv    # Raw dataset  
│   ├── processed_data.csv       # Preprocessed dataset  
│  
│── notebooks/                   # Jupyter notebooks for EDA and model training  
│   ├── 1_EDA.ipynb              # Exploratory Data Analysis  
│   ├── 2_Preprocessing.ipynb    # Data preprocessing and feature engineering  
│   ├── 3_Model_Training.ipynb   # Training different models  
│   ├── 4_Evaluation.ipynb       # Model evaluation and performance metrics  
│  
│── models/                      # Trained machine learning models  
│   ├── random_forest_model.pkl  # Saved Random Forest model  
│   ├── xgboost_model.pkl        # Saved XGBoost model  
│  
│── scripts/                     # Python scripts for automation  
│   ├── preprocess.py            # Data preprocessing script  
│   ├── train_model.py           # Model training script  
│   ├── evaluate.py              # Model evaluation script  
│  
│── visualizations/               # Plots and graphs generated  
│   ├── rating_distribution.png   # Histogram of IMDb ratings  
│   ├── feature_importance.png    # Feature importance bar chart  
│  
│── submission/                   # Folder for storing predictions  
│   ├── predictions.csv           # Predicted movie ratings  
│  
│── README.md                     # Project documentation  
│── requirements.txt               # Dependencies and libraries  
│── .gitignore                     # Files to ignore in Git  
│── LICENSE                        # Open-source license  


---

##  Dataset Information  
- **Dataset Source**: IMDb Movies India dataset from Kaggle  
- **Target Variable**: `imdb rating` (Regression task)  
- **Features**:
  - `title` - Movie title  
  - `genre` - Categories like Action, Comedy, Drama, etc.  
  - `director` - Name of the director  
  - `cast` - Lead actors in the movie  
  - `budget` - Total budget of the movie  
  - `revenue` - Total earnings  
  - `runtime` - Duration in minutes  
  - `release year` - Year of movie release  
  - `critic reviews` - Number of critic reviews  
  - `user votes` - Number of user ratings  

---

Why Random Forest?
1. Handles Missing Values Efficiently
The Titanic dataset contains missing values in Age, Cabin, and Embarked columns.

Random Forest can handle missing values by averaging predictions from multiple decision trees.

2. Works Well with Categorical Data
The dataset includes categorical features like Sex (Male/Female), Pclass (1st, 2nd, 3rd), and Embarked (C/Q/S).

Random Forest can handle categorical data without needing extensive preprocessing.

3. Reduces Overfitting Compared to Decision Trees
A single Decision Tree can overfit, capturing noise instead of patterns.

Random Forest uses multiple trees and averages their predictions, reducing overfitting.

4. Handles Non-Linear Relationships
Some features like Fare vs. Survival or Age vs. Survival have non-linear relationships.

Random Forest is flexible and captures such patterns better than linear models.

5. Feature Importance Analysis
Random Forest provides a feature importance ranking, helping us understand which features influence survival.

Example: Sex and Pclass are usually the most important predictors.

6. Robust and High Accuracy
In previous Titanic Kaggle competitions, Random Forest performed better than simpler models like Logistic Regression.

It achieves an accuracy of around 80-85% on Titanic datasets.


