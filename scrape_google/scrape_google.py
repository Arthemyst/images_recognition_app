from simple_image_download import Downloader


keywords = ["plane", "missile", "drone"]
limit = 600
response = Downloader()
response.directory = './scrape_google/images/'

for keyword in keywords:
    response.download(keyword, limit)
