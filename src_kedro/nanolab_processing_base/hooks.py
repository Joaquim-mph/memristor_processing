from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro_datasets.pandas import CSVDataset
from kedro_datasets.partitions import PartitionedDataset
from nanolab_processing_base.hooks_utils import separate_nanolab_dataset

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicDatasetHook:
    def __init__(self):
        self.projects = None
        self.is_processing = False  # Flag to prevent recursion

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog, **kwargs) -> None:
        """
        Register dynamic datasets without triggering infinite recursion.
        """
        self.projects = [key for key in catalog.list() if key.startswith("project_")]

        for project in self.projects:
            # Define paths for the datasets
            properties_path = f"data/03_primary/properties_{project}.csv"
            data_path = f"data/03_primary/data_{project}"

            # Register dynamic datasets
            catalog.add(
                f"properties_{project}",
                CSVDataset(filepath=properties_path),
            )

            catalog.add(
                f"data_{project}",
                PartitionedDataset(
                    path=data_path,
                    dataset=CSVDataset,
                    filename_suffix=".csv",
                ),
            )

        if self.is_processing:  # Prevent recursive calls
            logger.info("Already processing datasets. Skipping this execution.")
            return

        self.is_processing = True  # Set flag to avoid recursion

        try:
            for project in self.projects:
                # Define paths for the datasets
                properties_path = f"data/03_primary/properties_{project}.csv"
                data_path = f"data/03_primary/data_{project}"

                logger.info(f"Processing dataset: {project}")
                dataset = catalog.load(project)  # This might trigger catalog hooks
                consolidated_props, indexed_data = separate_nanolab_dataset(dataset)

                # Save properties
                catalog.save(f"properties_{project}", consolidated_props)
                logger.info(f"Saved properties for {project} to {properties_path}")

                # Save partitioned dataset
                catalog.save(f"data_{project}", indexed_data)
                logger.info(f"Saved data partitions for {project} to {data_path}")
        except Exception as e:
            logger.error(f"Error processing datasets: {e}")
        finally:
            self.is_processing = False

