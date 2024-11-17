import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

def retrain_model():
    # Load your dataset
    # Replace this with your actual dataset path
    df = pd.read_csv("C:/Users/gauth/Downloads/Disease-Prediction-from-Symptoms-master/dataset/training_data.csv")

    # Assuming 'prognosis' is your target variable and the rest are features
    X = df.drop("prognosis", axis=1)  # Features
    y = df["prognosis"]  # Target variable

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForest model (or any other classifier)
    clf = RandomForestClassifier(random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate model performance (optional)
    accuracy = clf.score(X_val, y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model to disk
    dump(clf, 'saved_model/random_forest.joblib')
    print("Model saved successfully!")

if __name__ == "__main__":
    retrain_model()
