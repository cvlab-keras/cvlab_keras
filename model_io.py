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
        self.do_load = False

    def get_attributes_with_path(self, path_parameter):
        return [], [Output("model", name="model")], \
               [path_parameter,
                ButtonParameter("load", self.load, "Load")]

    def process_inputs(self, inputs, outputs, parameters):
        path = parameters["path"]

        if self.do_load and path != "":
            self.do_load = False
            self.may_interrupt()
            model: models.Model = models.load_model(path)
            outputs["model"] = Data(model)

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
        self.do_save = False

    def get_attributes_with_path(self, path_parameter):
        return [Input("model", name="model")], [], \
               [path_parameter,
                ButtonParameter("save", self.save, "Save")]

    def process_inputs(self, inputs, outputs, parameters):
        path = parameters["path"]
        model = inputs["model"].value

        if self.do_save and model is not None and path != "":
            self.do_save = False
            self.may_interrupt()
            models.save_model(model, path)

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

    def get_attributes(self):
        # because model constructors are not JSON serializable we use workaround dictionary with key:key
        duplicate_key_dictionary = {key: key for key in self.model_constructor_dictionary.keys()}
        return [], \
               [Output("model", name="model")], \
               [ComboboxParameter("model", duplicate_key_dictionary),
                ComboboxParameter("top", [("no", False), ("yes", True)], "include top"),
                ComboboxParameter("weights", [("pre-trained - ImageNet", 'imagenet'), ("random", None)]),
                IntParameter("height", name="input height", value=224, min_=32),
                IntParameter("width", name="input width", value=224, min_=32),
                ButtonParameter("load", self.load, "Load")]

    def process_inputs(self, inputs, outputs, parameters):
        model_key = parameters["model"]
        has_top = parameters["top"]
        weights = parameters["weights"]
        input_width = parameters["width"]
        input_height = parameters["height"]

        if self.do_load:
            self.do_load = False
            self.may_interrupt()
            model_constructor = self.model_constructor_dictionary.get(model_key)
            # Passing shape as None, when top is included, results with required (for classifier) input shape being used
            shape = None if has_top else (input_height, input_width, 3)  # all models require 3 channels

            model = model_constructor(include_top=has_top, weights=weights, input_shape=shape)
            outputs["model"] = Data(model)

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
