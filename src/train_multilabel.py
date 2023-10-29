from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
from imutils import paths
import numpy as np
import pandas as pd
import argparse
import pickle
import cv2
import os
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model_preparation.architecture import models
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def plot_hist(history, filename, model_name):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Accuracy", "Loss"))

    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["accuracy"],
            name="train_accuracy",
            mode="markers+lines",
            marker_color="#f29407",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["val_accuracy"],
            name="valid_accuracy",
            mode="markers+lines",
            marker_color="#0771f2",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["loss"],
            name="train_loss",
            mode="markers+lines",
            marker_color="#f29407",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["val_loss"],
            name="valid_loss",
            mode="markers+lines",
            marker_color="#0771f2",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Count of epochs", row=1, col=1)
    fig.update_xaxes(title_text="Count of epochs", row=2, col=1)
    fig.update_yaxes(title_text="Accuraccy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_layout(width=1400, height=1000, title=f"Metrics: {model_name}")

    po.plot(fig, filename=filename, auto_open=False)


# Example of execution:
# $ python train_multilabel.py -d images -e 1

np.random.seed(10)

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--images', required=True, help='Path to the data')
ap.add_argument('-e', '--epochs', default=1, type=int, help='Choose number of epochs')
args = vars(ap.parse_args())

MODEL_NAME = "VGGNetSmall"
EPOCHS = args['epochs']
LEARNING_RATE = 0.001
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)

print('[INFO] Data loading...')
image_paths = list(paths.list_images(args['images']))
np.random.shuffle(image_paths)

data = []
labels = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    image = img_to_array(image)
    data.append(image)

    label = image_path.split('\\')[-2].split('_')
    labels.append(label)

data = np.array(data, dtype='float') / 255.
labels = np.array(labels)

print(f'[INFO] {len(image_paths)} images with size: {data.nbytes / (1024 * 1000.0):.2f} MB')
print(f'[INFO] Shape of data: {data.shape}')

print(f'[INFO] Binarization of labels... ')
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
print(f'[INFO] Labels: {mlb.classes_}')

print(f'[INFO] Export labels to file...')
with open(r'output/mlb,pickle', 'wb') as file:
    file.write(pickle.dumps(mlb))

print('[INFO] Split to train and test datasets...')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)
print(f'[INFO] Shape of train dataset: {X_train.shape}')
print(f'[INFO] Shape of test dataset: {X_test.shape}')

print('[INFO] Generator building...')
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print('[INFO] Model building...')
architecture = models.VGGNetSmall(input_shape=INPUT_SHAPE, num_classes=len(mlb.classes_), final_activation='sigmoid')
model = architecture.build()
model.summary()

model.compile(optimizer=Adam(lr=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
filepath = os.path.join('output', 'multilabel_model_' + dt + '.hd5f')
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor="val_accuracy", save_best_only=True
)
early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

print("[INFO] Model training...")
history = model.fit_generator(
    generator=train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
)

filename = os.path.join('output', 'multilabel_report_' + dt + '.html')
print(f"[INFO] Exporting plot to file {filename}...")
plot_hist(history, filename, MODEL_NAME)

print("[INFO] END")


