import os
import pathlib
import requests
from typing import Literal
from tqdm import tqdm
import zipfile
import shutil
import hashlib

def verify_checksum(file_path, expected_checksum):
    hash_algo, hash_value = expected_checksum.split(':', 1)
    h = hashlib.new(hash_algo)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == hash_value


def download_file(project_id: str, data: str, local_folder: pathlib.Path, force_download: bool = False):
    url = f"https://zenodo.org/api/records/{project_id}/files/{data}.zip"
    if os.path.exists(local_folder):
        if force_download:
            shutil.rmtree(local_folder)
        elif not (len(os.listdir(local_folder)) == 1 and f"{data}.zip" == os.listdir(local_folder)[0]):
            print(f"Skipping {data}, already exists.")
            return local_folder
    r = requests.get(url)
    r_json = r.json()
    checksum = r_json.get('checksum', None)
    size = r_json.get('size', 0)
    content_path = r_json.get('links', {}).get('content', None)
    r_content = requests.get(content_path, stream=True)
    zip_path = local_folder / f"{data}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tqdm(total=int(size), unit='B', unit_scale=True, desc=f"Downloading {data}") as pbar:
            r_content.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r_content.iter_content(chunk_size=10*1024*1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        if checksum:
            if not verify_checksum(zip_path, checksum):
                os.remove(zip_path)
                raise ValueError(f"Checksum mismatch for {zip_path}")
        local_folder.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {data}... at {local_folder}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check the contents of the zip file
            names = zip_ref.namelist()
            top_level = {n.split('/')[0] for n in names if n.strip('/')}
            if len(top_level) == 1 and list(top_level)[0] == data:
                extract_path = local_folder.parent
            else:
                extract_path = local_folder
            zip_ref.extractall(extract_path)
        os.remove(zip_path)
        return local_folder
    except KeyboardInterrupt:
        print("Download interrupted. Cleaning up...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise KeyboardInterrupt("Download was interrupted by the user.")

def download_demo_data(data = Literal["Carp-parallel", "Carp-cone", "SimulatedFluidInvasion", "Bentheimer"], force_download: bool = False):
    local_cache_dir = pathlib.Path(__file__).parent / "data"
    local_cache_dir.mkdir(exist_ok=True)
    project_id = "16448474"
    local_folder = local_cache_dir / data
    return download_file(project_id, data, local_folder, force_download)


def get_demo_data_path(data = Literal["Carp-parallel", "Carp-cone", "SimulatedFluidInvasion", "Bentheimer"]):
    local_cache_dir = pathlib.Path(__file__).parent / "data"
    return local_cache_dir / data

if __name__ == "__main__":
    download_demo_data("Bentheimer")