import tensorflow as tf
from cnn_model import create_model
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm

def load_dataset(base_path='data/train'):
    csv_path = os.path.join(base_path, '_classes.csv')
    df = pd.read_csv(csv_path)
    
    # Standardize column naming (remove leading/trailing spaces if any)
    df.columns = [c.strip() for c in df.columns]
    
    # Exclude 'filename' column for labels
    label_columns = [col for col in df.columns if col != 'filename']
    
    X = []
    y = []
    
    print(f"Loading images from {base_path}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_filename = row['filename']
        img_path = os.path.join(base_path, img_filename)
        
        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Ensure 28x28
        img = cv2.resize(img, (28, 28))
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        
        X.append(img)
        
        # Get labels as one-hot array
        labels = row[label_columns].values.astype('float32')
        y.append(labels)
        
    return np.array(X), np.array(y), label_columns

def train_model():
    # Load training data
    X_train, y_train, classes = load_dataset('data/train')
    
    # Load test data (as validation)
    X_val, y_val, _ = load_dataset('data/test')
    
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")
    
    # Create model
    model = create_model(num_classes=num_classes)
    
    # Summary
    model.summary()
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32
    )
    
    model.save('braille_cnn.h5')
    print("Training complete! Model saved as 'braille_cnn.h5'")
    
    # Save the labels mapping to a file for use in app.py
    with open('model_labels.txt', 'w') as f:
        f.write("\n".join(classes))
    print("Labels mapping saved to 'model_labels.txt'")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
