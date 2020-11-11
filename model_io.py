from tensorflow.keras import models, applications

from cvlab.diagram.elements.base import *

HDF5_EXT = ".h5"
HDF5_FILTER = "HDF5 (*" + HDF5_EXT + ")"

# TODO (cvlab_keras) set convention for input and output names


class ModelFileLoader(NormalElement):
    name = 'Model file loader'
    comment = 'Loads the whole model from single file with .h5 extension (HDF5 format)'

    def get_attributes(self):
        return [], [Output("output", name="output (model)")],\
               [PathParameter("file path (.h5)", value="", extension_filter=HDF5_FILTER)]

    def process_inputs(self, inputs, outputs, parameters):
        path = parameters["file path (.h5)"]

        if path != "":
            try:
                model: models.Model = models.load_model(path)
                self.may_interrupt()
                outputs["output"] = Data(model)
            except OSError:
                raise IOError("Model file is either invalid or does not exists at: "+path)


class ModelFileSaver(NormalElement):
    name = 'Model file saver'
    comment = 'Saves the whole model to a single file with .h5 extension (HDF5 format)'

    def get_attributes(self):
        return [Input("input", name="input (model)")], [], \
               [SavePathParameter("file path (.h5)", value="", extension_filter=HDF5_FILTER)]

    def process_inputs(self, inputs, outputs, parameters):
        path = parameters["file path (.h5)"]
        model = inputs["input"].value

        if path != "":
            try:
                self.may_interrupt()
                models.save_model(model, path)
            except AttributeError:
                raise TypeError("Input type is invalid! Make sure to connect proper model input")


class ModelDirectoryLoader(NormalElement):
    name = 'Model directory loader'
    comment = 'Loads the whole model from directory (SavedModel format)'

    def get_attributes(self):
        return [], [Output("output", name="output (model)")],\
               [DirectoryParameter("directory path", value="")]

    def process_inputs(self, inputs, outputs, parameters):
        path = parameters["directory path"]

        if path != "":
            try:
                model: models.Model = models.load_model(path)
                self.may_interrupt()
                outputs["output"] = Data(model)
            except OSError:
                raise IOError("Model directory is either invalid or does not exists at: " + path)


class ModelDirectorySaver(NormalElement):
    name = 'Model directory saver'
    comment = 'Saves the whole model to a directory (SavedModel format)'

    def get_attributes(self):
        return [Input("input", name="input (model)")], [], \
               [DirectoryParameter("directory path", value="")]

    def process_inputs(self, inputs, outputs, parameters):
        path = parameters["directory path"]
        model = inputs["input"].value

        if path != "":
            try:
                self.may_interrupt()
                models.save_model(model, path)
            except AttributeError:
                raise TypeError("Input type is invalid! Make sure to connect proper model input")


class PretrainedModelLoader(NormalElement):
    name = 'Pre-trained model loader'
    comment = \
        "Loads one of keras built in pre-trained models\n"+\
        "When top is included input width and height are omitted (dimensions compatible with classifier are used)\n"+\
        "For more information see https://keras.io/api/applications/"

    model_constructor_dictionary = {  # full list of pre-trained models here https://keras.io/api/applications/
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

    def get_attributes(self):
        # because model constructors are not JSON serializable we use workaround dictionary with key:key
        duplicate_key_dictionary = {key: key for key in self.model_constructor_dictionary.keys()}
        return [], \
               [Output("output", name="model")],\
               [ComboboxParameter("model", duplicate_key_dictionary),
                ComboboxParameter("include top", [("no", False), ("yes", True)]),
                ComboboxParameter("weights", [("pre-trained - ImageNet", 'imagenet'), ("random", None)]),
                IntParameter("input height", value=224, min_=32),
                IntParameter("input width", value=224, min_=32)]

    def process_inputs(self, inputs, outputs, parameters):
        model_key = parameters["model"]
        model_constructor = self.model_constructor_dictionary.get(model_key)
        has_top = parameters["include top"]
        weights = parameters["weights"]
        width = parameters["input width"]
        height = parameters["input height"]

        # Passing shape as None, when top is included, results with required (for classifier) input shape being used
        shape = None if has_top else (height, width, 3)  # 3 is required number of color channels for all models

        model = model_constructor(include_top=has_top, weights=weights, input_shape=shape)
        outputs["output"] = Data(model)


register_elements_auto(__name__, locals(), "Model IO", 1)
