"""
Demo 07: reconstruct from a configuration file.
An example of this is the Bentheimer experiment. 
"""
from nect.download_demo_data import download_demo_data
from nect import reconstruct_from_config_file
import yaml

config_file = download_demo_data("Bentheimer") / "config.yaml"

# need to change the img_path to point to the path of the projections
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

config["img_path"] = str(config_file.parent / "projections")

with open(config_file, "w") as f:
    yaml.safe_dump(config, f)

reconstruct_from_config_file(config_file)
