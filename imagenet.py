from tensorflow.keras.applications import imagenet_utils

from cvlab.diagram.elements.base import *


class ImageNetPredictionsDecoder(NormalElement):
    name = 'ImageNet predictions decoder'
    comment = 'Decodes ImageNet probabilistic predictions into class names'

    def get_attributes(self):
        return [Input("predictions", name="predictions")], \
               [Output("decoded", name="decoded predictions", preview_only=True)], \
               [IntParameter("top", "top n predictions", value=5, min_=1, max_=1000)]

    def process_inputs(self, inputs, outputs, parameters):
        predictions = inputs["predictions"].value
        top_number = parameters["top"]

        if predictions is not None:
            decoded = imagenet_utils.decode_predictions(predictions, top_number)
            layout_base = '{:4} {:16.16} '  # number column is 4 chars wide and name one is 16 (cropping to long names)
            header = layout_base+'{:6}'     # 6 characters for probabilities title
            layout = layout_base+'{:0.4f}'  # display 4 decimal places of probability value
            output_string = header.format('no.', 'name', 'prob.') + '\n'
            for i in range(0, top_number):
                _, name, probability = decoded[0][i]
                output_string += layout.format(str(i+1)+'.', name, round(probability, 4))
                output_string += '\n' if i != top_number-1 else ''  # don't add endline in the last line

            outputs["decoded"] = Data(output_string)


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