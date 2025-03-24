"""This module defines and runs a local TFX pipeline using BeamDagRunner."""

import os
from typing import Text
from absl import logging
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration import pipeline
from tfx.orchestration.metadata import sqlite_metadata_connection_config

# Pipeline name
PIPELINE_NAME = 'masdewa-pipeline'

# Path configurations
DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/transform.py'
TRAINER_MODULE_FILE = 'modules/trainer.py'

OUTPUT_BASE = 'outputs'
SERVING_MODEL_DIR = os.path.join(OUTPUT_BASE, 'serving_model')
PIPELINE_ROOT = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata.sqlite')
METADATA_CONFIG = sqlite_metadata_connection_config(METADATA_PATH)


def initialize_local_pipeline(components_list,
                              root_path: Text) -> pipeline.Pipeline:
    """
    Initialize a local TFX pipeline.

    Args:
        components_list: A list of TFX components to be included in the pipeline.
        root_path: Root directory for pipeline output artifacts.

    Returns:
        A TFX pipeline object.
    """
    logging.info(f'Pipeline root set to: {root_path}')

    beam_args = [
        '--direct_running_mode=multi_processing',
        '--direct_num_workers=1',  # Reduce workers to save memory
        '--experiments=shuffle_mode=service',
        '--runner=DirectRunner',
        '--max_num_records=10000'  # Limit number of records processed at once
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=root_path,
        components=components_list,
        enable_cache=True,
        metadata_connection_config=METADATA_CONFIG,
        beam_pipeline_args=beam_args
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components
    from modules.components import PipelineConfig

    # Initialize pipeline components
    config = PipelineConfig(
        data_dir=DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=SERVING_MODEL_DIR,
    )

    pipeline_components = init_components(config)

    # Create the pipeline object
    local_pipeline = initialize_local_pipeline(
        components_list=pipeline_components,
        root_path=PIPELINE_ROOT
    )

    # Run the pipeline
    BeamDagRunner().run(pipeline=local_pipeline)
