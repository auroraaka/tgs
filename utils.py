import os
import re

import numpy as np

def read_data(file_path: str, header_length: int = 15) -> np.ndarray:
    """
    Read TGS signal data file and return time and amplitude data casted as a numpy array.

    Parameters:
        file_path (str): path to the TGS data file
        header_length (int): number of lines to skip in the header
    Returns:
        np.ndarray: array of shape (N, 2) containing [time, amplitude]
    """
    time_data = []
    amplitude_data = []
    with open(file_path, 'r') as file:
        for _ in range(header_length):
            next(file)
        next(file)
        for line in file:
            time, amplitude = map(float, line.strip().split('\t'))
            time_data.append(time)
            amplitude_data.append(amplitude)
    
    return np.column_stack((time_data, amplitude_data))

def get_num_signals(path: str) -> int:
    """
    Get the number of positive signal files in the given path.

    Parameters:
        path (str): path to the directory containing the positive signal files
    Returns:
        int: number of positive signal files
    """
    pattern = re.compile(r'POS-(\d+)\.txt')
    return max((int(match.group(1)) for filename in os.listdir(path)
                if (match := pattern.search(filename))), default=0)

def get_file_prefix(path: str, i: int) -> str:
    """
    Get the file prefix of the positive signal file with the given index.

    Parameters:
        path (str): path to the directory containing the positive signal files
        i (int): index of the positive signal file
    Returns:
        str: file prefix
    """
    pattern = re.compile(rf'(.+)-POS-{i}\.txt')
    for filename in os.listdir(path):
        if match := pattern.match(filename):
            return match.group(1)
    return None