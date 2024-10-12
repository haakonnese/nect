import numpy as np
import scipy.constants


def equidistant(nprojs: int, nrevs: int, radians: bool = True) -> np.ndarray:
    """Generates a set of angles using the equidistant method

    Args:
        nprojs (int): The number of projections per revolution.
        nrevs (int): The number of revolutions.
        radians (bool, optional): Whether to return angles in radians. Defaults to True.

    Returns:
        np.ndarray: The equiditant angles.
    """
    start = 0
    end = 360 * nrevs
    angles = np.linspace(start, end, nprojs * nrevs, endpoint=False)

    if radians:
        return angles * np.pi / 180
    else:
        return angles


def golden_angle(nprojs: int, radians: bool = True) -> np.ndarray:
    """Generates a set of angles using the golden angle method

    Args:
        nprojs (int): The number of projections per revolution.
        radians (bool, optional): Whether to return angles in radians. Defaults to True.

    Returns:
        np.ndarray: The angles sampled by the golden angle method.
    """
    angles = np.arange(nprojs) * 360 / scipy.constants.golden_ratio
    if radians:
        return angles * np.pi / 180
    else:
        return angles


def golden_angle_v3(nprojs, nrevs, radians=True, starting=0):
    """Copied from Ruben's code 4D_CT/Golden Angle Analysis/analysisV3.ipynb"""
    golden_angle_sampling = lambda n, inc: np.mod((n * 1 / ((np.sqrt(5) - 1) / 2) * inc), inc)

    startings = golden_angle_sampling(np.arange(starting, nrevs), 360 / nprojs)

    linear_sampling = lambda s: np.linspace(s, s + 360, nprojs, endpoint=False)

    angles = linear_sampling(startings).T
    if radians:
        return angles * np.pi / 180

    # Reorder angles such that they appear in increasing/decreasing order per revolution
    angles[1::2, ...] = angles[
        1::2, ::-1
    ]  # Every other sublist/revolution is reversed to simulate the CT moving clockwise then anticlockwise
    angles = np.array(angles).flatten()
    return angles


if __name__ == "__main__":
    angles = golden_angle_v3(100, 56, radians=False)
    print(angles)
    print(angles[100:])
    # print(np.diff(angles))
    # print(np.sum(np.abs(np.diff(angles))))
    # # np.savetxt("angles.txt", angles, delimiter=",")
    # # angles = equidistant(25, 50, radians=False)

    # arr_list = angles.tolist()
    # print(len(arr_list))
    # # Convert the list to a string with commas separating the numbers
    # arr_str = ",".join(map(str, arr_list))

    # # Save the string to a file
    # file_path = "array_numbers.txt"
    # with open(file_path, "w") as file:
    #     file.write(arr_str)

    # print(f"Array saved to {file_path}")
