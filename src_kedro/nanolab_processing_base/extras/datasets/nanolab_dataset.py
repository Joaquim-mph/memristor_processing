from typing import Any, Dict
from nanolab_processing_base.extras.datasets.nanolab_dataframe import make_props_data, get_procedures
import pickle
from kedro.io.core import AbstractDataset


class NanoLabDataSet(AbstractDataset):
    def __init__(self, filepath: str, catalog: Any = None):
        """
        Custom dataset for NanoLab data.

        Args:
            filepath (str): The path to the data file.
            catalog (Any): Optional Kedro catalog object or similar placeholder.
        """
        super().__init__()
        self.filepath = filepath
        self.catalog = catalog
        self.props = None
        self.data = None

    def _load(self) -> Any:
        """
        Load the dataset by parsing the file and loading properties and data.

        Returns:
            tuple: A tuple of (properties, data), where properties is a pandas Series
            and data is a pandas DataFrame.
        """
        procedures = get_procedures()  # Ensure procedures are loaded
        self.props, self.data = make_props_data(self.filepath)
        return self.props, self.data

    def _save(self, data: Any) -> None:
        """
        Save the dataset to a pickle file.

        Args:
            data (Any): The data to save.
        """
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)

    def _describe(self) -> Dict[str, Any]:
        """
        Describe the dataset for catalog purposes.

        Returns:
            dict: A dictionary with dataset details.
        """
        return {
            "type": "NanoLab Data Frame",
            "filepath": self.filepath,
        }
