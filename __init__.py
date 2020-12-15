import tensorflow as tf

from cvlab.diagram.elements import load_auto
from cvlab.view.widgets import OutputPreview
from cvlab_samples import add_samples_submenu

from .model_utils import set_model
from cvlab_keras.shared import SAMPLES_DIR

# limit gpu usage for tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# add preview for model class
OutputPreview.preview_callbacks.append((tf.keras.models.Model, set_model))

# load elements
load_auto(__file__)

# load example diagrams
add_samples_submenu('Keras', SAMPLES_DIR)