from typing import List

from simple_image_download import Downloader


def download_images(keywords: List[str], directory: str, images_limit: int):
    response = Downloader()
    response.directory = directory

    for keyword in keywords:
        response.download(keyword, images_limit)


if __name__ == '__main__':
    keywords = ["plane", "missile", "drone"]
    images_limit = 600
    directory = './scrape_google/images/'
    download_images(keywords, directory, images_limit)
