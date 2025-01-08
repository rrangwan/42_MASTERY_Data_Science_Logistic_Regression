import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

from utils import is_valid_csv

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def prepare_data(dataset_file):
    """Prepares the dataset for prediction (X: features)."""
    data = pd.read_csv(dataset_file)

    # print(f"before drop na dataset shape: {data.shape}")
    
    # print(f"Initial dataset shape: {data.shape} {data.columns}")
    data = data.drop(columns=['Care of Magical Creatures'], errors='ignore')
    data = data.drop(columns=['Potions'], errors='ignore')
    data = data.drop(columns=['Arithmancy'], errors='ignore')
    # print(f"Initial dataset shape: {data.shape} ")
    # Drop rows containing any NaN values
    # data = data.dropna()
    # Fill NaN values with a constant (e.g., 0)
    data = data.fillna(0)



    # Not care of magical creatures feature and arithmancy
    # Extract features (excluding non-numeric columns)
    feature_columns = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts','Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic','Transfiguration', 'Charms', 'Flying']

        # For test data, allow missing labels ('Hogwarts House') but drop NaN in features
    # data = data[feature_columns].dropna()

    # print(f"Dataset shape after processing: {data.shape}")
    # print(f"after drop na dataset shape: {data.shape}")

    X = data[feature_columns].values
    print(f"Missing values in feature columns after processing:\n{data[feature_columns].isna().sum()}")

    # Normalize the data using Z-score normalization (mean 0, std 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Add an intercept (bias) term to X
    X = np.c_[np.ones(X.shape[0]), X]
    
    return X

def load_weights(weight_file):
    """Loads the model weights from a CSV file."""
    return np.loadtxt(weight_file, delimiter=',')

def softmax(z):
    """Softmax function for multi-class classification."""
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stabilize the exp function
    return e_z / e_z.sum(axis=1, keepdims=True)

def predict(X, weights):
    """Makes predictions using the trained model weights."""
    logits = np.dot(X, weights)
    
    # Debugging: Check the shapes of X, weights, and logits
    print(f"Shape of X: {X.shape}")
    print(f"Shape of weights: {weights.shape}")
    print(f"Shape of logits: {logits.shape}")
    
    probabilities = softmax(logits)  # Apply softmax for multi-class classification
    
    # Return the index of the maximum probability (the predicted class)
    return np.argmax(probabilities, axis=1)


def save_predictions(predictions, output_file):
    """Saves the predictions to a CSV file."""
    house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    house_predictions = [house_names[pred] for pred in predictions]
    
    output = pd.DataFrame({'Index': range(len(predictions)), 'Hogwarts House': house_predictions})
    output.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <dataset_test.csv> <model_weights.csv>")
        sys.exit(1)

    dataset_file = sys.argv[1]
    weight_file = sys.argv[2]

    # Validate the CSV files
    if not is_valid_csv(weight_file) or not is_valid_csv(dataset_file):
        sys.exit(1)
    
 
    # Prepare the data for prediction
    X = prepare_data(dataset_file)
    
    # Load the trained weights
    weights = load_weights(weight_file)
    
    # Make predictions
    print("Making predictions...")
    predictions = predict(X, weights)
    
    # Save the predictions to a CSV file
    save_predictions(predictions, 'houses.csv')

if __name__ == "__main__":
    main()