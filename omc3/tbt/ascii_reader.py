import datetime
import numpy as np
import pandas as pd


_ACQ_DATE_PREFIX = "#Acquisition date: "


def read_ascii_file(file_path):
    bpm_names = {"0": [], "1": []}
    matrix = {"0": [], "1": []}
    date = None
    with open(file_path, "r") as file_data:
        for line in file_data:
            line = line.strip()
            # Empty lines and comments:
            if line == "" or "#" in line:
                continue
            if _ACQ_DATE_PREFIX in line:
                date = _parse_date(line)
                continue
            # Samples:
            parts = line.split()
            bpm_plane = parts.pop(0)
            bpm_name = parts.pop(0)
            parts.pop(0)
            bpm_samples = _get_samples_array(parts)
            try:
                bpm_names[bpm_plane].append(bpm_name)
                matrix[bpm_plane].append(bpm_samples)
            except KeyError:
                raise ValueError(f"Wrong plane found in: {file_path}")
    matrix_x = pd.DataFrame(index=bpm_names["0"], data=np.array(matrix["0"]))
    matrix_y = pd.DataFrame(index=bpm_names["1"], data=np.array(matrix["1"]))
    return (bpm_names["0"], matrix_x,
            bpm_names["1"], matrix_y, date)


def is_ascii_file(file_path):
    """
    Returns true only if the file looks like a redable tbt ASCII file.
    """
    with open(file_path, "r") as file_data:
        for line in file_data:
            if line.strip() == "":
                continue
            return line.startswith("#")
    return False


def _parse_date(line):
    date_str = line.replace(_ACQ_DATE_PREFIX, "")
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d at %H:%M:%S")
    except:
        return datetime.datetime.today()


def _get_samples_array(samples_strings):
    return np.array([float(sample) for sample in samples_strings])
