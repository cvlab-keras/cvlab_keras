from tensorflow.keras import models

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


register_elements_auto(__name__, locals(), "Model IO", 1)
