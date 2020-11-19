import tensorflow as tf

from cvlab.diagram.elements import load_auto
from cvlab.view.widgets import OutputPreview
from cvlab_keras.model_utils import set_model


# limit gpu usage for tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# add preview for model class
OutputPreview.preview_callbacks.append((tf.keras.models.Model, set_model))

load_auto(__file__)
