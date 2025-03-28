# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import kagglehub

# Step 1: Download dataset from Kaggle
dataset_path = kagglehub.dataset_download("adrianmcmahon/imdb-india-movies")
print("Path to dataset files:", dataset_path)

# Step 2: List available files
available_files = os.listdir(dataset_path)
print("Available files in dataset:", available_files)

# Step 3: Detect correct file
file_name = "IMDb Movies India.csv"
data_path = os.path.join(dataset_path, file_name)

# Step 4: Load dataset with correct encoding and clean column names
def load_data(file_path):
    """Loads a dataset file into a Pandas DataFrame with proper encoding and cleans column names."""
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip().str.lower()  # Convert column names to lowercase and remove spaces
    return df

df = load_data(data_path)
print("\nDataset loaded successfully.")
print(df.head())

# Step 5: Check column names
print("\nColumns in the dataset:", df.columns)

# Step 6: Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 7: Encode categorical variables
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}

for column in categorical_features:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    label_encoders[column] = encoder

# Step 8: Feature Engineering (Ensure "director" and "imdb rating" exist)
if "director" in df.columns and "imdb rating" in df.columns:
    director_avg_rating = df.groupby("director")["imdb rating"].mean()
    df["director_avg_rating"] = df["director"].map(director_avg_rating)

# Step 9: Select features and target
target_column = "imdb rating" if "imdb rating" in df.columns else df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 10: Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Normalize numerical data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 12: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 13: Make predictions
y_pred = model.predict(X_val)

# Step 14: Evaluate model
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R Squared Score:", r2)

# Step 15: Save predictions
submission = pd.DataFrame({"Movie_ID": df.index[:len(y_pred)], "Predicted_Rating": y_pred})
submission_path = os.path.join(dataset_path, "submission.csv")
submission.to_csv(submission_path, index=False)

print("\nPredictions saved at:", submission_path)

# Step 16: Visualization

# Check if "imdb rating" exists before plotting
if "imdb rating" in df.columns:
    # Distribution of IMDB Ratings
    plt.figure(figsize=(8, 5))
    sns.histplot(df["imdb rating"], bins=20, kde=True, color="blue")
    plt.title("Distribution of IMDB Ratings")
    plt.xlabel("IMDB Rating")
    plt.ylabel("Count")
    plt.show()

    # Actual vs Predicted Ratings Scatter Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_val, y_pred, color="red", alpha=0.5)
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], linestyle="dashed", color="black")
    plt.title("Actual vs Predicted Ratings")
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.show()

# Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importance.nlargest(10).plot(kind="bar", color="green")
plt.title("Top 10 Important Features")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.show()
