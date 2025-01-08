import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import is_valid_csv

def softmax(z):
    """Softmax function for multi-class classification."""
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stabilize the exp function
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def predict(X, weights):
    """Makes predictions using the trained model weights (multi-class)."""
    logits = np.dot(X, weights)  # Logits (raw predictions) of shape (num_samples, num_classes)
    return softmax(logits)  # Apply softmax to convert logits to probabilities

def gradient_descent(X, y, weights, learning_rate=0.01, epochs=1000):
    """Performs gradient descent to find the optimal weights."""
    m = len(y)
    for epoch in range(epochs):
        # Calculate the predictions (probabilities) using softmax
        predictions = predict(X, weights)
        
        # Compute the gradient (for multi-class classification)
        gradient = np.dot(X.T, predictions - y) / m
        
        # Update the weights
        weights -= learning_rate * gradient
        
        # Optionally, print the loss every 100 epochs
        if epoch % 100 == 0:
            # Compute the cross-entropy loss for multi-class classification
            loss = -np.mean(np.sum(y * np.log(predictions + 1e-10), axis=1))  # Avoid log(0)
            print(f"Epoch {epoch}: Loss = {loss}")
    return weights

def prepare_data(dataset_file):
    """Prepares the dataset for training (X: features, y: labels)."""
    data = pd.read_csv(dataset_file)

    # Drop rows containing any NaN values
    print(f"Initial dataset shape: {data.shape}")
    # Fill NaN values with a constant (e.g., 0)
    data = data.fillna(0)

    # Not care of magical creatures feature, arithmancy, potions
    data = data.drop(columns=['Care of Magical Creatures'], errors='ignore')
    data = data.drop(columns=['Potions'], errors='ignore')
    data = data.drop(columns=['Arithmancy'], errors='ignore')
    print(f"after dropping na dataset shape: {data.shape}")

    # Updated feature columns excluding 'Care of Magical Creatures' and 'Arithmancy
    feature_columns = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
                       'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
                       'Transfiguration', 'Charms', 'Flying']

    X = data[feature_columns].values
    
    # Normalize the features to improve gradient descent performance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode Hogwarts House as one-hot categorical labels (e.g., Gryffindor -> [1, 0, 0, 0])
    house_mapping = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
    y = data['Hogwarts House'].map(house_mapping).values
    y_one_hot = np.eye(4)[y]  # Convert labels to one-hot encoding
    
    # Add an intercept (bias) term to X
    X = np.c_[np.ones(X.shape[0]), X]
    
    return X, y_one_hot

def save_weights(weights, weight_file):
    """Saves the model weights to a CSV file."""
    np.savetxt(weight_file, weights, delimiter=',')
    print(f"Weights saved to {weight_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
        sys.exit(1)

    dataset_file = sys.argv[1]
    
    # Validate the CSV files
    if is_valid_csv(dataset_file) == False:
        exit(1)

    # Prepare the data
    X, y = prepare_data(dataset_file)
    
    # Initialize weights with zeros for multi-class classification (num_features x num_classes)
    weights = np.zeros((X.shape[1], 4))  # 4 classes in your case
    
    # Train the model using gradient descent
    print("Training model using gradient descent...")
    weights = gradient_descent(X, y, weights)
    
    # Save the weights to a CSV file
    save_weights(weights, 'weights.csv')

if __name__ == "__main__":
    main()
