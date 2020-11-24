import tensorflow as tf

from glob import glob

from cvlab.diagram.elements import load_auto
from cvlab.view.widgets import OutputPreview
from cvlab_samples import OpenExampleAction, get_menu
from cvlab.diagram.elements import add_plugin_callback

from cvlab_keras.model_utils import set_model
from cvlab_keras.shared import SAMPLES_DIR

# limit gpu usage for tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# add preview for model class
OutputPreview.preview_callbacks.append((tf.keras.models.Model, set_model))

# load elements
load_auto(__file__)


def add_samples(main_window):
    menu_name = "Examples"
    submenu_name = "Keras"
    menu_title = menu_name + "/" + submenu_name
    samples = glob(SAMPLES_DIR + "/*.cvlab")
    samples.sort()

    print("Adding {} sample diagrams to '{}' menu".format(len(samples), menu_title))

    menu = get_menu(main_window, menu_title)

    for sample in samples:
        menu.addAction(OpenExampleAction(main_window, sample))


add_plugin_callback(add_samples)
