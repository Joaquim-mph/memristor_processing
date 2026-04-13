"""
This is a boilerplate pipeline 'SnS_preprocessing'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import process_partitioned_bg_corrected, get_maximums_bg_partitioned

def create_pipeline(**kwargs) -> Pipeline:
    process_hysteresis_michael = Pipeline(
        [
            # 1 ─ raw → michael's hysteresis
            node(
                func=process_partitioned_bg_corrected,
                inputs=["properties_project_joaco", "data_project_joaco"],
                outputs="michael_hysteresis",
                name="Calculation_of_background_corrected_hysteresis",
            ),
            # 2 - michael's hysteresis → max michael's hysteresis, hysteresis field
            node(
                func = get_maximums_bg_partitioned,
                inputs=["properties_project_joaco", "data_project_joaco"],
                outputs="props_with_hyst_field_and_max_hysteresis",
                name="Calculation_of_hyst_field_and_max_hysteresis"
            ),
        ]
    )

    return process_hysteresis_michael