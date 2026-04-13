
"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import process_partitioned_hysteresis, get_maximums_hysteresis_partitioned



def create_pipeline(**kwargs) -> Pipeline:
    process_hysteresis_tomas = Pipeline(
        [
            # 1 ─ raw → tomas's hysteresis
            node(
                func=process_partitioned_hysteresis,
                inputs=["properties_project_joaco", "data_project_joaco"],
                outputs="tomas_hysteresis",
                name="Calculation_of_IV_raw_hysteresis",
            ),
            # 2 - tomas's hysteresis → max tomas's hysteresis
            node(
                func = get_maximums_hysteresis_partitioned,
                inputs=["properties_project_joaco", "tomas_hysteresis"],
                outputs="props_with_maximums_tomas_hysteresis",
                name="Calculation_of_max_IV_hysteresis"
            ),
        ]
    )
    return process_hysteresis_tomas