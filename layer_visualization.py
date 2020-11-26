import pandas as pd
import pydot
import matplotlib.pyplot as plt
from tensorflow import keras as keras
#  from tensorflow.keras import preprocessing
from keras.models import load_model
from keras.metrics import Recall
# from tensorflow.keras.utils import plot_model
# from vis.utils import utils
# from vis.visualization import visualize_activation, get_num_filters
# from vis.input_modifiers import Jitter

dependencies = {
    'recall': Recall
}

# fn = './keras_models/tenor_cnn_brass_ok/tenor_brass_ok_cnn_G.h5'
fn = './keras_models/tenor_cnn/tenor_cnn_G.h5'
model = load_model(fn, custom_objects=dependencies, compile=False)

folder_name = 'filter_visualizations'

# dir(model)
# '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__',
# '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__',
# '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__',
# '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
# '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__',
# '__weakref__', '_add_inbound_node', '_add_unique_metric_name', '_base_init',
# '_build_input_shape', '_built', '_cache_output_metric_attributes',
# '_check_trainable_weights_consistency', '_compute_previous_mask',
# '_expects_training_arg', '_feed_input_names', '_feed_input_shapes',
# '_feed_inputs', '_get_callback_model', '_get_existing_metric',
# '_get_node_attribute_at_index', '_get_training_eval_metrics',
# '_handle_metrics', '_handle_per_output_metrics', '_inbound_nodes',
# '_init_graph_network', '_init_subclassed_network', '_initial_weights',
# '_input_coordinates', '_input_layers', '_is_compiled', '_is_graph_network',
# '_layers', '_layers_by_depth', '_losses', '_make_predict_function',
# '_make_test_function', '_make_train_function', '_metrics', '_network_nodes',
# '_node_key', '_nodes_by_depth', '_non_trainable_weights', '_outbound_nodes',
# '_output_coordinates', '_output_layers', '_output_mask_cache',
# '_output_shape_cache', '_output_tensor_cache', '_per_input_losses',
# '_per_input_updates', '_prepare_total_loss', '_set_inputs',
# '_set_metric_attributes', '_set_per_output_metric_attributes',
# '_set_sample_weight_attributes', '_standardize_user_data',
# '_trainable_weights', '_updated_config', '_updates',
# '_uses_dynamic_learning_phase', '_uses_inputs_arg',
# '_validate_or_infer_batch_size', 'add', 'add_loss', 'add_metric',
# 'add_update', 'add_weight', 'assert_input_compatibility', 'build', 'built', '
# call', 'compile', 'compute_mask', 'compute_output_shape', 'count_params',
# 'dtype', 'evaluate', 'evaluate_generator', 'fit', 'fit_generator',
# 'from_config', 'get_config', 'get_input_at', 'get_input_mask_at',
# 'get_input_shape_at', 'get_layer', 'get_losses_for', 'get_output_at',
# 'get_output_mask_at', 'get_output_shape_at', 'get_updates_for', 'get_weights',
# 'input', 'input_mask', 'input_names', 'input_shape', 'input_spec', 'inputs',
# 'layers', 'load_weights', 'losses', 'metrics', 'metrics_names', 'model',
# 'name', 'non_trainable_weights', 'optimizer', 'output', 'output_mask',
# 'output_names', 'output_shape', 'outputs', 'pop', 'predict',
# 'predict_classes', 'predict_generator', 'predict_on_batch', 'predict_proba',
# 'reset_metrics', 'reset_states', 'run_internal_graph', 'save', 'save_weights',
# 'set_weights', 'state_updates', 'stateful', 'summary', 'supports_masking',
# 'test_on_batch', 'to_json', 'to_yaml', 'train_on_batch', 'trainable',
# 'trainable_weights', 'updates', 'uses_learning_phase', 'weights']

# print(model.layers)
# print(dir(model.layers[0]))

# dir(model.layers[0])
# '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__',
# '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__',
# '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__',
# '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
# '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
# '_add_inbound_node', '_built', '_get_existing_metric',
# '_get_node_attribute_at_index', '_inbound_nodes', '_initial_weights',
# '_losses', '_metrics', '_node_key', '_non_trainable_weights',
# '_outbound_nodes', '_per_input_losses', '_per_input_updates',
# '_trainable_weights', '_updates', 'activation', 'activity_regularizer',
# 'add_loss', 'add_metric', 'add_update', 'add_weight',
# 'assert_input_compatibility', 'batch_input_shape', 'bias', 'bias_constraint',
# 'bias_initializer', 'bias_regularizer', 'build', 'built', 'call',
# 'compute_mask', 'compute_output_shape', 'count_params', 'data_format',
# 'dilation_rate', 'dtype', 'filters', 'from_config', 'get_config',
# 'get_input_at', 'get_input_mask_at', 'get_input_shape_at', 'get_losses_for',
# 'get_output_at', 'get_output_mask_at', 'get_output_shape_at',
# 'get_updates_for', 'get_weights', 'input', 'input_mask', 'input_shape',
# 'input_spec', 'kernel', 'kernel_constraint', 'kernel_initializer',
# 'kernel_regularizer', 'kernel_size', 'losses', 'metrics', 'name',
# 'non_trainable_weights', 'output', 'output_mask', 'output_shape', 'padding',
# 'rank', 'set_weights', 'stateful', 'strides', 'supports_masking', 'trainable',
# 'trainable_weights', 'updates', 'use_bias', 'weights']

conv_layer_names = ['conv2d_38', 'conv2d_39', 'conv2d_40', 'conv2d_41']
for layer in model.layers:
    if layer.name in conv_layer_names:
        print(layer.weights[0])
