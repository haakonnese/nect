import argparse
from pathlib import Path


def txt_key_to_inr_key(key: str) -> str:
    if key == "noPixel":
        return "nDetector"
    elif key == "dimPixel":
        return "dDetector"
    elif key == "totSizeDetector":
        return "sDetector"
    elif key == "noVoxel":
        return "nVoxel"
    elif key == "dimVoxel":
        return "dVoxel"
    elif key == "totSizeVoxel":
        return "sVoxel"
    else:
        return key


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Convert NSI geometry to TIGRE geometry")
    parser.add_argument("nsi_geometry", help="Path to the NSI geometry txt file")
    args = parser.parse_args()

    # Read the NSI geometry txt file
    geom_dict = {}
    nsi_geometry_path = Path(args.nsi_geometry)
    with open(nsi_geometry_path, "r") as file:
        # read the lines separately
        lines = file.readlines()
        for line in lines:
            # Split the line into key and value
            key, values = line.split("=")
            key = key.strip()
            key = txt_key_to_inr_key(key)
            values = values.strip()
            if key == "mode":
                key = key
            elif len(values.split(" ")) > 1:
                if key == "nVoxel" or key == "nDetector":
                    values = [int(value) for value in values.split(" ")]
                    # reverse the list
                    values = values[::-1]
                else:
                    values = [float(value) for value in values.split(" ")]
            else:
                if key == "nProjections":
                    values = int(values)
                else:
                    values = float(values)
            if key in ["sDetector", "sVoxel", "offOrigin", "offDetector"]:
                # reverse the list
                values = values[::-1]
            geom_dict[key] = values
        geom_dict["radians"] = True

    # Dump the geometry to a yaml file, save the lists as list and not as indents
    geometry_path = nsi_geometry_path.with_suffix(".yaml")
    with open(geometry_path, "w") as file:
        for key, value in geom_dict.items():
            # Write each key-value pair without indentation
            file.write(f"{key}: ")
            if isinstance(value, list):
                # If value is a list, write it in square brackets without indentation
                file.write("[")
                file.write(", ".join([str(item) for item in value]))
                file.write("]\n")
            else:
                # If value is not a list, write it directly followed by a newline
                file.write(f"{value}\n")
