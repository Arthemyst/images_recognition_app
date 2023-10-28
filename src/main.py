from config import CustomEnvironment
from images_preparation.images_preparation import prepare_images

if __name__ == '__main__':
    base_dir = CustomEnvironment.get_base_dir()
    train_ratio = CustomEnvironment.get_train_ratio()
    valid_ratio = CustomEnvironment.get_valid_ratio()
    data_dir = CustomEnvironment.get_data_dir()
    classes = CustomEnvironment.get_images_classes()

    prepare_images(classes, data_dir, base_dir, train_ratio, valid_ratio)
