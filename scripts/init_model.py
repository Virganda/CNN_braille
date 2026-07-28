import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cnn_model import create_model

def init():
    model = create_model()
    model.save('braille_cnn.h5')
    print("Created dummy model 'braille_cnn.h5' with random weights.")

if __name__ == '__main__':
    init()
