import os
from typing import List

import environ


class CustomEnvironment:
    env = environ.Env(
        TRAIN_RATIO=(float, 0.7),
        VALID_RATIO=(float, 0.2),
        DATA_DIR=(str, r"images-dataset/"),
        ORIGINAL_IMAGES_DIRECTORY=(str, r"images/"),
    )

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environ.Env.read_env(os.path.join(BASE_DIR, ".env"))

    _train_ratio = env.float("TRAIN_RATIO")
    _valid_ratio = env.float("VALID_RATIO")
    _data_dir = env.str("DATA_DIR")
    _original_images_directory = env.str("ORIGINAL_IMAGES_DIRECTORY")
    _images_classes = env.list("IMAGES_CLASSES")

    @classmethod
    def get_train_ratio(cls) -> float:
        return cls._train_ratio

    @classmethod
    def get_valid_ratio(cls) -> float:
        return cls._valid_ratio

    @classmethod
    def get_data_dir(cls) -> str:
        return cls._data_dir

    @classmethod
    def get_original_images_directory(cls) -> str:
        return cls._original_images_directory

    @classmethod
    def get_base_dir(cls) -> str:
        original_images_directory = cls.get_original_images_directory()
        execution_path = os.getcwd()
        return os.path.join(execution_path, original_images_directory)

    @classmethod
    def get_images_classes(cls) -> List[str]:
        return list(cls._images_classes)
