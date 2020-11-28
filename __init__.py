import tensorflow as tf

from cvlab.diagram.elements import load_auto
from cvlab.view.widgets import OutputPreview
from cvlab_samples import add_samples_submenu

from cvlab_keras.model_utils import set_model
from cvlab_keras.shared import SAMPLES_DIR

# limit gpu usage for tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# add preview for model class
OutputPreview.preview_callbacks.append((tf.keras.models.Model, set_model))

# load elements
load_auto(__file__)

# load example diagrams
add_samples_submenu('Keras', SAMPLES_DIR)
