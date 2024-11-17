# Import Dependencies
import yaml
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import yaml
import seaborn as sn
import matplotlib.pyplot as plt

class DiseasePrediction:
    # Initialize and Load the Config File
    def __init__(self, model_name=None):
        """Initialize the class by loading the configuration and datasets."""
        # Load Config File
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
                if not self.config:
                    raise ValueError("Config file is empty or malformed.")
        except Exception as e:
            print(f"Error reading Config file: {e}")
            self.config = {}

        # Verbose
        self.verbose = self.config.get('verbose', True)
        
        # Model definition
        self.model_name = model_name
        
        # Model Save Path
        self.model_save_path = self.config.get('model_save_path', './saved_model')  # Default save path
        
        # Load Training and Test Data
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()

        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_df, show_fig=False)

    def _load_train_dataset(self):
        """Load the training dataset from the provided path in the config file."""
        try:
            df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        except Exception as e:
            print(f"Error loading training data: {e}")
            return None, None, None
        
        # Validate columns and separate features and labels
        cols = df_train.columns.tolist()
        if len(cols) < 3:
            print("Invalid dataset columns. Expected at least three columns.")
            return None, None, None

        # Feature columns
        train_features = df_train[cols[:-2]]
        # Label column
        train_labels = df_train['prognosis']

        # Sanity check
        assert len(train_features.iloc[0]) == 132, "Unexpected number of features."
        assert len(train_labels) == train_features.shape[0], "Mismatch between features and labels."

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)

        return train_features, train_labels, df_train

    def _load_test_dataset(self):
        """Load the test dataset from the provided path in the config file."""
        try:
            df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        except Exception as e:
            print(f"Error loading test data: {e}")
            return None, None, None
        
        # Validate columns and separate features and labels
        cols = df_test.columns.tolist()
        if len(cols) < 2:
            print("Invalid dataset columns. Expected at least two columns.")
            return None, None, None

        # Feature columns
        test_features = df_test[cols[:-1]]
        # Label column
        test_labels = df_test['prognosis']

        # Sanity check
        assert len(test_features.iloc[0]) == 132, "Unexpected number of features."
        assert len(test_labels) == test_features.shape[0], "Mismatch between features and labels."

        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
            print("Test Labels: ", test_labels.shape)

        return test_features, test_labels, df_test

    def _feature_correlation(self, data_frame=None, show_fig=False):
        """
        Calculate and plot feature correlation for the given data frame.
        
        Args:
        - data_frame (pd.DataFrame): The dataframe containing the features.
        - show_fig (bool): If True, will display the plot. If False, will save the plot as a PNG file.
        """
        if data_frame is None:
            print("No data frame provided for feature correlation.")
            return
        
        # Select only numeric columns to avoid errors
        numeric_df = data_frame.select_dtypes(include=[np.number])

        # Calculate correlation on numeric columns
        corr = numeric_df.corr()

        # Plot the correlation matrix
        plt.figure(figsize=(10, 8))
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()

        # Show or save the plot as needed
        if show_fig:
            plt.show()
        else:
            plt.savefig('feature_correlation.png')
            print("Feature correlation plot saved as 'feature_correlation.png'.")


    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    # Model Selection
    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            self.clf = GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                                  criterion=self.config['model']['gradient_boost']['criterion'])
        return self.clf

    # ML Model
    def train_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        # Training the Model
        classifier = classifier.fit(X_train, y_train)
        # Trained Model Evaluation on Validation Dataset
        confidence = classifier.score(X_val, y_val)
        # Validation Data Prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', accuracy)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # Save Trained Model
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))

    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model
            clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")

        if test_data is not None:
            result = clf.predict(test_data)
            return result
        else:
            result = clf.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, result)
        clf_report = classification_report(self.test_labels, result)
        return accuracy, clf_report


if __name__ == "__main__":
    # Model Currently Training
    current_model_name = 'decision_tree'
    # Instantiate the Class
    dp = DiseasePrediction(model_name=current_model_name)
    # Train the Model
    dp.train_model()
    # Get Model Performance on Test Data
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    print("Model Test Accuracy: ", test_accuracy)
    print("Test Data Classification Report: \n", classification_report)