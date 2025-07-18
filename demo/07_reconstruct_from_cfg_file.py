"""
Demo 07: reconstruct from a configuration file.
An example of this is the Bentheimer experiment. 
Download the data, fill in the missing file paths in the config file, and run this script. 
"""

from nect import reconstruct_from_config_file


config_file = "<path_to_config.yaml>"

reconstruct_from_config_file(config_file)
