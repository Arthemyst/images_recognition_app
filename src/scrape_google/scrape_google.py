from typing import List
from src.config import CustomEnvironment
from simple_image_download import Downloader


def download_images(keywords: List[str], directory: str, images_limit: int):
    response = Downloader()
    response.directory = directory

    for keyword in keywords:
        response.download(keyword, images_limit)


if __name__ == '__main__':
    keywords = ["plane", "missile", "drone"]
    images_limit = 600
    original_images_directory = CustomEnvironment.get_original_images_directory()
    download_images(keywords, original_images_directory, images_limit)
