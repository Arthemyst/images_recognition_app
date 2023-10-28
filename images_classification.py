import os
import numpy as np
from typing import Dict
execution_path = os.getcwd()
base_dir = os.path.join(execution_path, 'scrape_google/images')

CLS_1 = 'horse'
CLS_2 = 'lion'
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
DATA_DIR = r'images-dataset'

raw_no_of_files = {}
classes = [CLS_1, CLS_2]

number_of_samples = [(class_dir, len(os.listdir(os.path.join(base_dir, class_dir)))) for
                     class_dir in classes]


def create_directory(class_name: str, train_dir: str, valid_dir: str, test_dir: str) -> Dict[str, str]:
    class_directories_dict = {}
    train_cls_dir = os.path.join(train_dir, class_name)
    class_directories_dict["train"] = train_cls_dir
    valid_cls_dir = os.path.join(valid_dir, class_name)
    class_directories_dict["valid"] = valid_cls_dir
    test_cls_dir = os.path.join(test_dir, class_name)
    class_directories_dict["test"] = test_cls_dir

    for directory in (train_cls_dir, valid_cls_dir, test_cls_dir):
        if not os.path.exists(directory):
            os.mkdir(directory)

    return class_directories_dict


if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

train_dir = os.path.join(DATA_DIR, 'train')
valid_dir = os.path.join(DATA_DIR, 'valid')
test_dir = os.path.join(DATA_DIR, 'test')

for directory in (train_dir, valid_dir, test_dir):
    if not os.path.exists(directory):
        os.mkdir(directory)

classes_list = []
for dataset_class in classes:
    class_dict = create_directory(dataset_class, train_dir, valid_dir, test_dir)
    classes_list.append(class_dict)
print(classes_list)
print('[INFO] Validation of correct names...')
list_of_cls_names = []
for dataset_class in classes:
    cls_names = os.listdir(os.path.join(base_dir, dataset_class))
    cls_names = [fname for fname in cls_names if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
    np.random.shuffle(cls_names)
    list_of_cls_names.append(cls_names)

for dataset_class, cls_names in zip(classes, list_of_cls_names):
    print(f'[INFO] Count of images in dataset {dataset_class}: {len(cls_names)}')

list_of_train_classes = []
list_of_valid_classes = []
for cls_names in list_of_cls_names:
    train_idx_cls = int(TRAIN_RATIO * len(cls_names))
    print(f"{train_idx_cls=}")
    list_of_train_classes.append(train_idx_cls)

    valid_idx_cls = train_idx_cls + int(VALID_RATIO * len(cls_names))
    print(f"{valid_idx_cls=}")
    list_of_valid_classes.append(valid_idx_cls)

print(f'[INFO] Copy files to directories...')
