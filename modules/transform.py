"""TFX Transform module for preprocessing input features."""

import tensorflow as tf

LABEL_KEY = "labels"
FEATURE_KEY = "text"

def preprocessing_fn(inputs):
    """
    Preprocessing function for TFX Transform component.

    Args:
        inputs (dict): Dictionary of input features.

    Returns:
        dict: Dictionary of transformed features.
    """
    print("DEBUG: Data masuk ke Transform:", inputs.keys())

    if FEATURE_KEY not in inputs:
        raise KeyError(
            f"ERROR: Kolom {FEATURE_KEY} tidak ditemukan! Kolom yang tersedia: {inputs.keys()}"
        )

    if inputs[FEATURE_KEY] is None:
        raise ValueError("ERROR: Nilai FEATURE_KEY adalah None!")

    if inputs[LABEL_KEY] is None:
        raise ValueError("ERROR: Nilai LABEL_KEY adalah None!")

    outputs = {}
    label_name = LABEL_KEY + "_xf"
    feature_name = FEATURE_KEY + "_xf"

    outputs[feature_name] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[label_name] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
