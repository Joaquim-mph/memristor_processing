import os
import re
import warnings
import logging
from typing import Dict, List, Tuple, Type
import pandas as pd
from kedro.framework.session import KedroSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data_key(path):
    """
    Extracts a key from a file path, consisting of the folder name and the file name (without extension).

    Args:
        path (str): The full file path to process.

    Returns:
        str: A string combining the folder name and the file name (without extension),
             separated by the operating system's path separator.

    Example:
        Input:
            '/home/user/documents/nanolab_processing_base/data/01_raw/project_name/2024-11-29/ITt2024-11-29_1.csv'
        Output:
            '2024-11-29/ITt2024-11-29_1'
    """
    # Split the path into components based on the operating system's path separator
    splitted_path = path.split("/")

    # Extract the file name (without the extension)
    file_name = splitted_path[-1].split(".")[0]

    # Extract the folder name (immediate parent of the file)
    folder = splitted_path[-2]

    # Join the folder and file name using the OS path separator
    return os.path.join(folder, file_name)


def load_catalog_item(dataset_name: str) -> Dict:
    """
    Load a dataset from Kedro's catalog.
    """
    with KedroSession.create() as session:
        context = session.load_context()
        catalog = context.catalog
        try:
            dataset = catalog.load(dataset_name)
            logger.info(f"Loaded dataset: {dataset_name}")
            return dataset
        except KeyError:
            logger.error(f"Dataset '{dataset_name}' not found in catalog.")
            raise


_cached_procedures = None


def get_procedures() -> Dict:
    """
    Lazily load the 'procedures' catalog item.
    """
    global _cached_procedures
    if _cached_procedures is None:
        _cached_procedures = load_catalog_item("params:procedures")
    return _cached_procedures


def unix_time_to_datetime(unix_time_str: str) -> pd.Timestamp:
    """
    Converts a Unix timestamp string into a pandas datetime object.

    Args:
        unix_time_str (str): Unix timestamp as a string (e.g., "1731364225.4285064").

    Returns:
        pd.Timestamp: Corresponding pandas datetime object.
    """
    try:
        unix_time = float(unix_time_str)
        return pd.to_datetime(unix_time, unit='s')
    except ValueError as e:
        raise ValueError(f"Invalid Unix timestamp: {unix_time_str}") from e


def parse_info_line(info_line: str) -> Dict[str, str]:
    """
    Parses a line containing key-value pairs separated by ': '.

    Args:
        info_line (str): The line to parse.

    Returns:
        Dict[str, str]: A dictionary containing the key and the value. If the value is empty, returns 'None'.
    """
    if ": " in info_line:
        key, value = info_line.split(": ", 1)
        value = value if value.strip() else "None"
    elif ":" in info_line:
        key, value = info_line.split(":", 1)
        value = value.strip() if value.strip() else "None"
    else:
        return {}

    return {key.strip(): value.strip()}


def make_dict_from_parsed_data(list_of_lines: List[str]) -> Dict[str, str]:
    """
    Converts a list of lines with key-value pairs separated by ': ' into a dictionary.
    Handles the line starting with 'Information' separately using `parse_info_line`.

    Args:
        list_of_lines (List[str]): List of strings, each containing a key-value pair.

    Returns:
        Dict[str, str]: A dictionary with all parsed key-value pairs.
    """
    # Separate the 'Information' line from the rest
    info_line = next((line for line in list_of_lines if line.startswith("Information")), None)
    rest_of_lines = [line for line in list_of_lines if not line.startswith("Information")]

    # Parse the 'Information' line if it exists
    dict_of_info = parse_info_line(info_line) if info_line else {}

    # Parse remaining lines, skipping any that don't contain a valid separator
    dict_rest_of_lines = {}
    for line in rest_of_lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            dict_rest_of_lines[key.strip()] = value.strip()

    # Combine both dictionaries
    return {**dict_of_info, **dict_rest_of_lines}


def read_comment_lines(file_path: str) -> Tuple[Dict[str, str], int]:
    """
    Reads and processes comment lines from the beginning of a file.
    """
    comment_lines = []
    current_line = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().startswith('#'):
                    current_line += 1
                if line.strip().startswith('#\t'):
                    comment_lines.append(line.strip().replace("#\t", ""))
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while processing the file: {e}")

    if comment_lines:
        return make_dict_from_parsed_data(comment_lines), current_line
    else:
        return {}, 0


def determine_procedure(path: str) -> str:
    """
    Extracts and processes content enclosed within angle brackets ("<" and ">")
    from the first line of a CSV file corresponding to an experiment.
    """
    if not path.endswith(".csv"):
        raise ValueError(f"The file '{path}' is not a CSV. Please provide a valid .csv file.")

    with open(path, "r") as file:
        first_line = file.readline().strip()

    match = re.search(r"<(.*?)>", first_line)
    if match:
        content = match.group(1)
        return content.split(".")[-1]
    else:
        warnings.warn("No content found between < and >", UserWarning)
        return ""


def string_to_bool(string_bool: str) -> bool:
    return string_bool == "True"


def parse_metadata(key, value, how_to_process):
    """
    Parses a metadata value based on the specified processing type.
    """
    if how_to_process == "float":
        return float(value.split(" ")[0])
    if how_to_process == "int":
        return int(value)
    if how_to_process == "bool":
        return string_to_bool(value)
    if how_to_process == "str":
        return value
    if how_to_process == "datetime":
        return unix_time_to_datetime(value)
    if how_to_process == "float_no_unit":
        return float(value)

    raise ValueError(f"Unhandled metadata key: {key}")


def get_keys(procedure: str) -> List[str]:
    """
    Retrieves a list of keys associated with a specific procedure.
    """
    procedures = get_procedures()
    if procedure not in procedures:
        raise KeyError(f"Procedure '{procedure}' not found in procedures.")

    parameters = list(procedures[procedure]["Parameters"].keys())
    metadata = list(procedures[procedure]["Metadata"].keys())

    if not isinstance(parameters, list) or not isinstance(metadata, list):
        raise TypeError(f"Expected 'Parameters' and 'Metadata' to be lists in procedure '{procedure}'.")

    return parameters + metadata


def make_props_data(path: str) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Parses properties and reads data from a file based on the procedure definition.
    """
    dictionary_found_properties, header = read_comment_lines(path)

    procedure = determine_procedure(path)
    procedures = get_procedures()
    keys = get_keys(procedure)
    procedure_dict = procedures[procedure]["Parameters"] | procedures[procedure]["Metadata"]
    props_dict = {}

    for key in dictionary_found_properties:
        if key in keys:
            props_dict[key] = parse_metadata(key, dictionary_found_properties[key], procedure_dict[key])
        else:
            raise KeyError(f"Key '{key}' is missing in the configuration file of {procedure}.")

    props_dict["data_key"] = get_data_key(path)
    props_dict["Procedure type"] = procedure
    props_series = pd.Series(props_dict)

    dtype_mapping = procedures[procedure]["Data"]
    try:
        data = pd.read_csv(path, header=header, dtype=dtype_mapping)
    except ValueError as e:
        raise ValueError(f"Error in reading {path} with dtype mapping: {dtype_mapping}") from e

    return props_series, data


