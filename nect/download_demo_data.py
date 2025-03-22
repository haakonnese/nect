import os
import pathlib
import requests
from typing import Literal
from tqdm import tqdm
import zipfile
import shutil

def download_folder(project_id: str, osf_folder: str, local_folder: pathlib.Path, force_download: bool = False):
    # print("Listing OSF files, this might take some time...")
    base_url = f"https://api.osf.io/v2/nodes/{project_id}/files/osfstorage/"
    params = {"page[size]": 100}
    files = []
    url = base_url
    while url:
        resp = requests.get(url, params=params)
        # print(resp.json())
        resp.raise_for_status()
        data = resp.json()
        # Check if the item is a folder and list its contents recursively
        for item in data.get("data", []):
            
            if item["attributes"]["kind"] == "file":
                file_id = item["id"]
                if osf_folder in item["attributes"]["materialized_path"]:
                    folder_url = f"https://osf.io/download/{file_id}/"
                    # Check if the folder already exists
                    if os.path.exists(local_folder):
                        if force_download:
                            shutil.rmtree(local_folder)
                        elif not (len(os.listdir(local_folder)) == 1 and f"{file_id}.zip" == os.listdir(local_folder)[0]):
                            print(f"Skipping {osf_folder}, already exists.")
                            continue
                    r = requests.get(folder_url, stream=True)
                    with tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True, desc=f"Downloading {osf_folder}") as pbar:
                        r.raise_for_status()
                        zip_path = local_folder / f"{file_id}.zip"
                        zip_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(zip_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=10*1024*1024):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    # Extract the zip file
                    local_folder.mkdir(parents=True, exist_ok=True)
                    print(f"Extracting {osf_folder}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

                        zip_ref.extractall(local_folder.parent)
                    # Remove the zip file after extraction
                    os.remove(zip_path)
        url = data.get("links", {}).get("next")

def download_demo_data(data = Literal["Carp-parallel", "Carp-cone", "SimulatedFluidInvasion"], force_download: bool = False):
    local_cache_dir = pathlib.Path(__file__).parent / "data"
    local_cache_dir.mkdir(exist_ok=True)
    project_id = "2w8xc"
    osf_folder = f"/{data}"
    local_folder = local_cache_dir / data
    download_folder(project_id, osf_folder, local_folder, force_download)


def get_demo_data_path(data = Literal["Carp-parallel", "Carp-cone", "SimulatedFluidInvasion"]):
    local_cache_dir = pathlib.Path(__file__).parent / "data"
    return local_cache_dir / data

# Example usage
# download_folder("your_project_id", "/data", "./local_data")
if __name__ == "__main__":
    download_demo_data("Carp-cone")