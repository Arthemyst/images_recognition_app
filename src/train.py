import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
import argparse
import pickle
import os

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model_preparation.architecture import models

print(f"Tensorflow version: {tf.__version__}")

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', default=1, help='Choose count of epochs', type=int)
args = vars(ap.parse_args())

MODEL_NAME = 'LeNet5'
LEARNING_RATE = 0.001
EPOCHS = args['epochs']
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)
TRAIN_DIR = 'images/train'
VALID_DIR = 'images/valid'
