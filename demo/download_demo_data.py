import os
from pathlib import Path
import json
import requests
from loguru import logger

# GitHub repository details
owner = "haakonnese"
repo = "NeCT-data"
branch = "main"  # or the branch you want to download from

# GitHub API URL to get the repository contents
api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"


def download_file(file_url, file_path):
    response = requests.get(file_url)
    response.raise_for_status()
    content = json.loads(response.content.decode("utf-8"))
    response = requests.get(content["download_url"])
    response.raise_for_status()
    content = response.content
    with open(file_path, "wb") as file:
        file.write(content)


def fetch_and_download_contents(api_url, save_dir):
    response = requests.get(api_url)
    response.raise_for_status()
    contents = response.json()

    for item in contents:
        item_path = os.path.join(save_dir, item["path"])

        if item["type"] == "dir":
            # If the item is a directory, create the directory and recursively fetch its contents
            os.makedirs(item_path, exist_ok=True)
            fetch_and_download_contents(item["url"], item_path)
        elif item["type"] == "file":
            if not os.path.exists(item_path):
                print(item)
                print(f"Downloading {item['path']}...")
                download_file(item["url"], item_path)


def download_demo_data(mode: str, folder: str | None = None):
    base_path = Path(__file__).parent / "NeCT-data"
    path = Path(mode) / folder if folder is not None else Path(mode)
    (base_path / path).mkdir(exist_ok=True, parents=True)
    
    fetch_and_download_contents(api_url + f"/{path}" + "?ref=" + branch, base_path)


if __name__ == "__main__":
    download_demo_data("static")
