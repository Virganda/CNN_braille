import tensorflow as tf
from tensorflow.keras import layers, models
import os

def create_model(num_classes=26):
    """
    Creates a CNN model for Braille character recognition.
    Standard input size is 28x28 grayscale images.
    """
    #MEMBUAT ALUR BERURUTAN
    model = models.Sequential([
        # First Convolutional Block / MEMINDAI POLA DASAR
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten / meratakan data and Dense Layers / fully connected
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def load_or_init_model(model_path='braille_cnn.h5', labels_path='model_labels.txt'):
    """
    Loads weights from model_path if exists, otherwise returns a fresh model.
    """
    num_classes = 26 # Default fallback
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = f.read().splitlines()
            num_classes = len(labels)
            
    model = create_model(num_classes=num_classes)
    if os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            print(f"Loaded weights from {model_path} ({num_classes} classes)")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print(f"Model weights not found. Using uninitialized model ({num_classes} classes) for demo.")
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    model.summary()
