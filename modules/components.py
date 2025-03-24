"""This module initializes TFX pipeline components for training and deploying ML models."""

import os
from dataclasses import dataclass
import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen,
    ExampleValidator, Transform, Trainer,
    Evaluator, Pusher
)
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.types import Channel
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)


@dataclass
class PipelineConfig:
    """Configuration dataclass to hold pipeline parameters."""
    data_dir: str
    transform_module: str
    training_module: str
    training_steps: int
    eval_steps: int
    serving_model_dir: str


def create_example_gen(data_dir):
    """Creates a CsvExampleGen component to ingest data from the specified directory."""
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
        ])
    )
    return CsvExampleGen(input_base=data_dir, output_config=output_config)


def create_transform(example_gen, schema_gen, transform_module):
    """Creates a Transform component to apply feature engineering based on 
    a preprocessing module."""
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(transform_module)
    )


def create_trainer(transform, schema_gen, training_module, steps, eval_steps):
    """Creates a Trainer component to train the model using provided preprocessing and schema."""
    return Trainer(
        module_file=os.path.abspath(training_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=steps),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=eval_steps)
    )


def create_evaluator(example_gen, trainer, model_resolver):
    """Creates an Evaluator component to evaluate the model and compare it with a baseline."""
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='labels')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='Precision'),
                tfma.MetricConfig(class_name='Recall'),
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.0001}
                        )
                    )
                )
            ])
        ]
    )
    return Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )


def init_components(config: PipelineConfig):
    """
    Initializes and returns a tuple of TFX pipeline components.

    Args:
        config (PipelineConfig): Configuration object for pipeline parameters.

    Returns:
        tuple: A list of initialized TFX components for the pipeline.
    """
    example_gen = create_example_gen(config.data_dir)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = create_transform(example_gen, schema_gen, config.transform_module)

    trainer = create_trainer(
        transform,
        schema_gen,
        config.training_module,
        config.training_steps,
        config.eval_steps
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    evaluator = create_evaluator(example_gen, trainer, model_resolver)

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config.serving_model_dir
            )
        )
    )

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
