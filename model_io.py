from tensorflow.keras import models, applications

from cvlab.diagram.elements.base import *

HDF5_EXT = ".h5"
HDF5_FILTER = "HDF5 (*" + HDF5_EXT + ")"


# TODO (cvlab_keras) set convention for input and output names


class _ModelFromDiskLoader(NormalElement):
    """
    Base class for elements that load models from disk.
    Derived classes must provide path parameter.
    """

    def __init__(self):
        super().__init__()
        self.path = None
        self.do_load = False

    def get_attributes_with_path(self, path_parameter):
        return [], [Output("output", name="output (model)")], \
               [path_parameter,
                ButtonParameter("load", self.load, "load")]

    def process_inputs(self, inputs, outputs, parameters):
        new_path = parameters["path"]

        if self.path != new_path:
            self.path = new_path

        if self.do_load and self.path != "":
            self.do_load = False
            self.may_interrupt()
            model: models.Model = models.load_model(self.path)
            outputs["output"] = Data(model)

    def load(self):
        self.do_load = True
        self.recalculate(True, False, True)


class ModelFromFileLoader(_ModelFromDiskLoader):
    name = 'Model file loader'
    comment = 'Loads the whole model from single file with '+HDF5_EXT+' extension'

    def get_attributes(self):
        path_parameter = PathParameter("path", name="file path (" + HDF5_EXT + ")", value="",
                                       extension_filter=HDF5_FILTER)
        return super().get_attributes_with_path(path_parameter)


class ModelFromDirectoryLoader(_ModelFromDiskLoader):
    name = 'Model directory loader'
    comment = 'Loads the whole model from directory (SavedModel format)'

    def get_attributes(self):
        path_parameter = DirectoryParameter("path", name="directory path", value="")
        return super().get_attributes_with_path(path_parameter)


class _ModelToDiskSaver(NormalElement):
    """
    Base class for elements that save models to disk.
    Derived classes must provide path parameter.
    """

    def __init__(self):
        super().__init__()
        self.path = None
        self.model = None
        self.do_save = False

    def get_attributes_with_path(self, path_parameter):
        return [Input("input", name="input (model)")], [], \
               [path_parameter,
                ButtonParameter("save", self.save, "save")]

    def process_inputs(self, inputs, outputs, parameters):
        new_path = parameters["path"]
        new_model = inputs["input"].value

        if self.path != new_path:
            self.path = new_path

        if self.model != new_model:
            self.model = new_model

        if self.do_save and self.model is not None and self.path != "":
            self.do_save = False
            self.may_interrupt()
            models.save_model(self.model, self.path)

    def save(self):
        self.do_save = True
        self.recalculate(True, False, True)


class ModelToFileSaver(_ModelToDiskSaver):
    name = 'Model file saver'
    comment = 'Saves the whole model to a single file with '+HDF5_EXT+' extension'

    def get_attributes(self):
        path_parameter = SavePathParameter("path", name="file path (" + HDF5_EXT + ")", value="",
                                           extension_filter=HDF5_FILTER)
        return super().get_attributes_with_path(path_parameter)


class ModelToDirectorySaver(_ModelToDiskSaver):
    name = 'Model directory saver'
    comment = 'Saves the whole model to a directory (SavedModel format)'

    def get_attributes(self):
        path_parameter = DirectoryParameter("path", name="directory path", value="")
        return super().get_attributes_with_path(path_parameter)


class PretrainedModelLoader(NormalElement):
    name = 'Pre-trained model loader'
    comment = \
        "Loads one of keras built in pre-trained models\n" + \
        "When top is included input width and height are omitted (dimensions compatible with classifier are used)\n" + \
        "For more information see https://www.tensorflow.org/api_docs/python/tf/keras/applications"

    model_constructor_dictionary = {
        # "name": constructor
        "DenseNet121": applications.DenseNet121,
        "DenseNet169": applications.DenseNet169,
        "DenseNet201": applications.DenseNet201,
        "InceptionResNetV2": applications.InceptionResNetV2(),
        "InceptionV3": applications.InceptionV3,
        "MobileNet": applications.MobileNet,
        "MobileNetV2": applications.MobileNetV2,
        # "NASNetLarge": applications.NASNetLarge,   # TODO (cvlab_keras) use NASNets when they will support False
        # "NASNetMobile": applications.NASNetMobile, # value for argument include_top in constructor
        "ResNet101V2": applications.ResNet101V2,
        "ResNet152V2": applications.ResNet152V2,
        "ResNet50V2": applications.ResNet50V2,
        "VGG16": applications.VGG16,
        "VGG19": applications.VGG19,
        "Xception": applications.Xception
    }

    def __init__(self):
        super().__init__()
        self.do_load = False
        self.model_key = None
        self.has_top = None
        self.weights = None
        self.input_width = None
        self.input_height = None

    def get_attributes(self):
        # because model constructors are not JSON serializable we use workaround dictionary with key:key
        duplicate_key_dictionary = {key: key for key in self.model_constructor_dictionary.keys()}
        return [], \
               [Output("output", name="model")], \
               [ComboboxParameter("model", duplicate_key_dictionary),
                ComboboxParameter("top", [("no", False), ("yes", True)], "include top"),
                ComboboxParameter("weights", [("pre-trained - ImageNet", 'imagenet'), ("random", None)]),
                IntParameter("height", name="input height", value=224, min_=32),
                IntParameter("width", name="input width", value=224, min_=32),
                ButtonParameter("load", self.load)]

    def process_inputs(self, inputs, outputs, parameters):
        if self.model_key != parameters["model"]:
            self.model_key = parameters["model"]
        if self.has_top != parameters["top"]:
            self.has_top = parameters["top"]
        if self.weights != parameters["weights"]:
            self.weights = parameters["weights"]
        if self.input_width != parameters["width"]:
            self.input_width = parameters["width"]
        if self.input_height != parameters["height"]:
            self.input_height = parameters["height"]

        if self.do_load:
            self.do_load = False
            self.may_interrupt()
            model_constructor = self.model_constructor_dictionary.get(self.model_key)
            # Passing shape as None, when top is included, results with required (for classifier) input shape being used
            shape = None if self.has_top else (self.input_height, self.input_width, 3)  # all models require 3 channels

            model = model_constructor(include_top=self.has_top, weights=self.weights, input_shape=shape)
            outputs["output"] = Data(model)

    def load(self):
        self.do_load = True
        self.recalculate(True, False, True)


elements = [
    ModelFromFileLoader,
    ModelFromDirectoryLoader,
    ModelToFileSaver,
    ModelToDirectorySaver,
    PretrainedModelLoader
]

register_elements("Model IO", elements, 1)
