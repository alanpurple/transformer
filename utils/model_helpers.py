import numbers
import tensorflow as tf
from tensorflow_core.python.util import nest
from absl import logging

def past_stop_threshold(stop_threshold,eval_metric):
    """Return a boolean representing whether a model should be stopped.

    Args:
        stop_threshold: float, the threshold above which a model should stop trainng.
        eval_metric: float, the current value of the relevant metric to check.

    Returns:
        True if training should stop, False otherwise.

    Raises:
        ValueError: if either stop_threshold or eval_metric is not a number.
    """
    if stop_threshold is None:
        return False

    if not isinstance(stop_threshold,numbers.Number):
        raise ValueError('Threshold for checking stop conditions must be a number.')
    if not isinstance(eval_metric,numbers.Number):
        raise ValueError('Eval metric being checked against stop conditions must be a number')

    if eval_metric>=stop_threshold:
        logging.info(
            'Stop threshold of {} was passed with metric value {}.'.format(stop_threshold,eval_metric)
        )
        return True

    return False

def generate_synthetic_data(input_shape,input_value=0,input_dtype=None,label_shape=None,
                            label_value=0,label_dtype=None):
    """Create a repeating dataset with constant values.

    Args:
        input_shape: a tf.TensorShape object or nested tf.TensorShapes. The shape of the input data
        input_value: Value of each input element.
        input_dtype: Input dtype. If None, will be inferred by the input value.
        label_shape: a tf.TensorShape object or nested tf.TensorShapes. The shape of
                    the label data.
        label_value: Value of each input element.
        label_dtype: Input dtype. If None, will be inferred by the target value.

    Returns:
        Dataset of tensors or tuples of tensors ( if label_shape is set).
    """
    # TODO: Replace with SyntheticDataset once it is in contrib.
    element=input_element=nest.map_structure(lambda  s: tf.constant(input_value,input_dtype,s),input_shape)

    if label_shape:
        label_element=nest.map_structure(lambda s: tf.constant(label_value,label_dtype,s),label_shape)
        element=(input_element,label_element)

    return tf.data.Dataset.from_tensors(element).repeat()

def apply_clean(flags_obj):
    if flags_obj.clean and tf.io.gfile.exists(flags_obj.model_dir):
        logging.info('--clean flag set. Removing existing model dir:'
                     ' {}'.format(flags_obj.model_dir))
        tf.io.gfile.rmtree(flags_obj.model_dir)