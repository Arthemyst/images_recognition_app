from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# example of execution:
# $ python predict_binary.py -d images-dataset/test -m output/model_28_10_2023_22_00.hdf5

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='type of images: [train, valid, test')
ap.add_argument('-m', '--model', required=False, help='path to model')
args = vars(ap.parse_args())

INPUT_SHAPE = (150, 150, 3)

datagen = ImageDataGenerator(rescale=1. / 255.)

generator = datagen.flow_from_directory(
    directory=args['dataset'],
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

print('[INFO] Model loading...')
model = load_model(args['model'])

y_prob = model.predict_generator(generator)
y_prob = y_prob.ravel()

y_true = generator.classes

predictions = pd.DataFrame({'y_prob': y_prob, 'y_true': y_true}, index=generator.filenames)
predictions['y_pred'] = predictions['y_prob'].apply(lambda x: 1 if x > 0.5 else 0)
predictions['is_incorrect'] = (predictions['y_true'] != predictions['y_pred']) * 1
errors = list(predictions[predictions['is_incorrect'] == 1].index)

y_pred = predictions['y_pred'].values

print(f'[INFO] Confusion matrix: \n{confusion_matrix(y_true, y_pred)}')
print(f'[INFO] Classification report: \n{classification_report(y_true, y_pred, target_names=generator.class_indices.keys())}')
print(f'[INFO] Model accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}')

label_map = generator.class_indices
label_map = {k: v for v, k in label_map.items()}
predictions['class'] = predictions['y_pred'].apply(lambda x: label_map[x])

predictions.to_csv(r'output/predictions.csv')

print(f'[INFO] Clasification errors: {len(errors)}')
print(f'[INFO] Files names (errors):')
for error in errors:
    print(error)




