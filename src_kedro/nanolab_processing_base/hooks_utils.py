from typing import Dict, Callable, Tuple
import pandas as pd
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def separate_nanolab_dataset(experiments: Dict[str, Callable]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Separates a NanoLab dataset into a consolidated properties DataFrame and indexed data.
    """
    logger.info(f"Starting separation of NanoLab dataset with {len(experiments)} experiments.")
    props_list = []
    indexed_data = {}
    for key, experiment_callable in experiments.items():
        try:
            prop, data = experiment_callable()
            props_list.append(prop)
            indexed_data[key] = data
            logger.info(f"Processed experiment: {key}")
        except Exception as e:
            logger.error(f"Error processing experiment {key}: {e}")

    consolidated_props = pd.DataFrame(props_list)

    # we sort by date and reset the index
    consolidated_props.sort_values(by="Start time", inplace=True)
    consolidated_props.reset_index(drop=True, inplace=True)

    logger.info("Finished dataset separation.")
    return consolidated_props, indexed_data
