import os
import shutil
from typing import Dict, List, Tuple

import numpy as np

NumList = List[int]
StringList = List[str]


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


def create_train_valid_test_directories(DATA_DIR: str) -> List[Dict[str, str]]:
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'valid')
    test_dir = os.path.join(DATA_DIR, 'test')

    for directory in (train_dir, valid_dir, test_dir):
        if not os.path.exists(directory):
            os.mkdir(directory)

    list_of_classes_dir_dicts = []
    for dataset_class in classes:
        class_dict = create_directory(dataset_class, train_dir, valid_dir, test_dir)
        list_of_classes_dir_dicts.append(class_dict)
    return list_of_classes_dir_dicts


def validation_of_correct_names(classes: List[str], base_dir: str) -> List[StringList]:
    print('[INFO] Validation of correct names...')
    list_of_cls_names = []
    for dataset_class in classes:
        cls_names = os.listdir(os.path.join(base_dir, dataset_class))
        cls_names = [fname for fname in cls_names if
                     fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
        np.random.shuffle(cls_names)
        list_of_cls_names.append(cls_names)

    for dataset_class, cls_names in zip(classes, list_of_cls_names):
        print(f'[INFO] Count of images in class "{dataset_class}": {len(cls_names)}')

    return list_of_cls_names


def create_lists_of_train_and_valid_count_of_classes(list_of_cls_names: List[StringList]) -> Tuple[NumList, NumList]:
    list_of_train_classes = []
    list_of_valid_classes = []
    for cls_names in list_of_cls_names:
        train_idx_cls = int(TRAIN_RATIO * len(cls_names))
        list_of_train_classes.append(train_idx_cls)

        valid_idx_cls = train_idx_cls + int(VALID_RATIO * len(cls_names))
        list_of_valid_classes.append(valid_idx_cls)
    return list_of_train_classes, list_of_valid_classes


def copy_files_to_directories(dataset_name: str, class_name: str, file_name: str, class_dir_dict: Dict) -> None:
    src = os.path.join(base_dir, class_name, file_name)
    dst = os.path.join(class_dir_dict[dataset_name], file_name)
    shutil.copyfile(src, dst)


def copy_classes_to_directories(
        list_of_cls_names: List[StringList], list_of_train_classes: List[int], list_of_valid_classes: List[int],
        list_of_classes_dir_dicts: List[Dict[str, str]], classes: List[str]) -> None:

    print(f'[INFO] Copy files to directories...')
    for cls_names, train_idx_cls, valid_idx_cls, class_dir_dict, class_name in zip(
            list_of_cls_names, list_of_train_classes, list_of_valid_classes,
            list_of_classes_dir_dicts, classes):
        for i, file_name in enumerate(cls_names):
            if i <= train_idx_cls:
                copy_files_to_directories("train", class_name, file_name, class_dir_dict)
            if train_idx_cls < i <= valid_idx_cls:
                copy_files_to_directories("valid", class_name, file_name, class_dir_dict)
            if valid_idx_cls < i <= len(cls_names):
                copy_files_to_directories("test", class_name, file_name, class_dir_dict)

    for class_name, class_dir_dict in zip(classes, list_of_classes_dir_dicts):
        for dataset in ["train", "valid", "test"]:
            count_of_images = len(os.listdir(class_dir_dict[dataset]))
            print(f'[INFO] Count of images for class "{class_name}" in {dataset} dataset: {count_of_images}')


def prepate_images(classes: List[str], DATA_DIR:str , base_dir: str) -> None:
    list_of_classes_dir_dicts = create_train_valid_test_directories(DATA_DIR)
    list_of_cls_names = validation_of_correct_names(classes, base_dir)
    list_of_train_classes, list_of_valid_classes = create_lists_of_train_and_valid_count_of_classes(list_of_cls_names)
    copy_classes_to_directories(list_of_cls_names, list_of_train_classes, list_of_valid_classes, list_of_classes_dir_dicts, classes)


if __name__ == '__main__':
    original_images_directory = 'scrape_google/images'
    execution_path = os.getcwd()
    base_dir = os.path.join(execution_path, original_images_directory)

    CLS_1 = 'horse'
    CLS_2 = 'lion'
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.2
    DATA_DIR = r'images-dataset'

    classes = [CLS_1, CLS_2]

    prepate_images(classes, DATA_DIR, base_dir)
