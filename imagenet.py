from tensorflow.keras.applications import imagenet_utils

from cvlab.diagram.elements.base import *
from cvlab_keras.model_operations import PredictionDecoder


class ImageNetPredictionDecoder(NormalElement):
    name = 'ImageNet prediction decoder'
    comment = 'Decodes ImageNet probabilistic prediction into class names'

    def get_attributes(self):
        return [Input("prediction", name="prediction")], \
               [Output("decoded", name="decoded prediction", preview_only=True)], \
               [IntParameter("top", "top n probabilities", value=5, min_=1, max_=1000)]

    def process_inputs(self, inputs, outputs, parameters):
        predictions = inputs["prediction"].value
        top_n = parameters["top"]

        if predictions is not None:
            decoded = imagenet_utils.decode_predictions(predictions, top_n)
            _, labels, prediction = zip(*decoded[0])
            formatted = PredictionDecoder.format_decoded_prediction(prediction, labels, top_n)
            outputs["decoded"] = Data(formatted)


class ImageNetInputPreprocessor(NormalElement):
    name = 'ImageNet input preprocessor'
    comment = \
        'Preprocesses image in one of 3 modes:\n'\
        ' - "tf": will scale pixels between -1 and 1, sample-wise\n'\
        ' - "caffe": will convert the images from RGB to BGR,\n'\
        '   then will zero-center each color channel with respect\n'\
        '   to the ImageNet dataset, without scaling\n'\
        ' - "torch": will scale pixels between 0 and 1 and then will\n'\
        '   normalize each channel with respect to the ImageNet dataset.'

    def get_attributes(self):
        return [Input("input", name="input")], \
               [Output("output", name="output")], \
               [ComboboxParameter("mode", [('tf', 'tf'), ('caffe', 'caffe'), ('torch', 'torch')])]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs["input"].value
        mode = parameters["mode"]

        if image is not None:
            preprocessed = imagenet_utils.preprocess_input(image, mode=mode)
            outputs["output"] = Data(preprocessed)


register_elements_auto(__name__, locals(), "ImageNet", 1)
