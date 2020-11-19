import tensorflow

from tensorflow.keras import models
from math import ceil

from cvlab.diagram.elements.base import *

# limit gpu usage for tensorflow
config = tensorflow.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tensorflow.compat.v1.Session(config=config)
tensorflow.compat.v1.keras.backend.set_session(sess)


class Predict(NormalElement):
    name = 'Predict'
    comment = \
        'Gets model prediction on an image\n' \
        'Visualises activation of a chosen layer on input image (all channels on one image)'
    model = None
    activation_model = None

    def get_attributes(self):
        return [Input("model", name="model"), Input("image", name="image")], \
               [Output("predictions", name="predictions", preview_enabled=False),
                Output("name", name="layer name", preview_only=True),
                Output("activation", name="activation images", preview_only=True)], \
               [IntParameter("layer", name="layer index", value=0, min_=0),
                IntParameter("row", name="channels in row", value=4, min_=1),
                IntParameter("spacing", name="spacing (px)", value=2, min_=0)]

    def process_inputs(self, inputs, outputs, parameters):
        model: models.Model = inputs["model"].value
        image: np.ndarray = inputs["image"].value

        if model != self.model:  # change activation model only if new model is given
            self.may_interrupt()
            self.model = model
            layer_outputs = [layer.output for layer in self.model.layers]
            self.activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)

        if model is not None and image is not None:
            self.may_interrupt()
            try:
                if image.ndim == 2:  # expand iamge dimensions for convolution layers
                    image = np.expand_dims(image, axis=2)
                image = np.expand_dims(image, axis=0)

                activation = self.activation_model.predict(image)
                layer_index = parameters["layer"]
                if len(activation) <= layer_index:
                    raise IndexError("Layer index: "+str(layer_index)+" is out of range<0;"+str(len(activation)-1)+">")

                imgs_in_row = parameters["row"]
                spacing = parameters["spacing"]
                final_image = self.create_activation_images(activation[layer_index], imgs_in_row, spacing)

                outputs["predictions"] = Data(activation[-1])  # predictions - output of the last layer
                outputs["name"] = Data("layer: " + self.activation_model.layers[layer_index].name)
                outputs["activation"] = Data(final_image)

            except ValueError as e:
                raise ValueError(self.msg_without_stacktrace(e.__str__(), "ValueError"))  # invalid shape for predict

    # Method of creating activation images taken from book:
    # "Deep Learning. Praca z językiem Python i biblioteką Keras" written by Francois Chollet
    @staticmethod
    def create_activation_images(channels: np.ndarray, imgs_in_row: int, spacing: int) -> np.ndarray:
        imgs_in_column = ceil(channels.shape[3] / imgs_in_row)
        img_px_width = channels.shape[1]
        img_px_height = channels.shape[2]
        row_px_total = imgs_in_column * img_px_height + (imgs_in_column-1)*spacing
        col_px_total = imgs_in_row * img_px_width + (imgs_in_row-1)*spacing
        img_container = np.full((row_px_total, col_px_total), 0)

        for index in range(channels.shape[3]):
            img: np.ndarray = channels[0, :, :, index]
            img = (img - img.mean())/(img.std()+0.001)  # add small value to std deviation to avoid dividing by 0
            img = np.clip(img*64 + 128, 0, 255).astype('uint8')

            col = index // imgs_in_row
            row = index % imgs_in_row
            col_start = col * (img_px_height+spacing)
            row_start = row * (img_px_width+spacing)

            img_container[col_start:col_start + img_px_height, row_start:row_start + img_px_width] = img

        return img_container

    @staticmethod
    def msg_without_stacktrace(error_msg: str, error_name: str):
        # to remove stack trace from the error message find first occurrence of error name
        # stack trace (find), error name (length of name) and colon with white space (+2 signs)
        cause_index = error_msg.find(error_name) + error_name.__len__() + 2
        final_msg = error_msg[cause_index:]  # only error cause message remains
        return final_msg


class PredictionDecoder(NormalElement):
    name = 'Prediction decoder'
    comment = 'Maps probabilistic prediction to labels'

    def get_attributes(self):
        return [Input("prediction", name="prediction"), Input("labels", name="labels")], \
               [Output("decoded", name="decoded predictions", preview_only=True)], \
               [IntParameter("top", "top n probabilities", value=5, min_=1)]

    def process_inputs(self, inputs, outputs, parameters):
        prediction = inputs["prediction"].value
        labels = inputs["labels"].value
        top_n = parameters["top"]

        if prediction is not None and labels is not None:
            labels_probabilities = sorted(zip(labels, prediction), key=lambda item: item[1], reverse=True)
            labels, prediction = zip(*labels_probabilities)
            formatted_prediction = PredictionDecoder.format_decoded_prediction(prediction, labels, top_n)
            outputs["decoded"] = Data(formatted_prediction)

    @staticmethod
    def format_decoded_prediction(prediction, labels, top_n):
        layout_base = '{:4} {:16.16} '  # number column is 4 chars wide and name one is 16 (cropping to long names)
        header = layout_base + '{:6}'  # 6 characters for probabilities title
        layout = layout_base + '{:0.4f}'  # display 4 decimal places of probability value
        formatted = header.format('no.', 'label', 'prob.') + '\n'

        top_lim = top_n if top_n <= len(prediction) else len(prediction)  # limit top_n to number of probabilities
        for i in range(0, top_lim):
            index = str(i + 1) + '.'
            label = labels[i]
            probability = prediction[i]
            formatted += layout.format(index, label, probability)
            formatted += '\n' if i != top_lim - 1 else ''  # don't add endline in the last line
        return formatted


register_elements("Model operations", [Predict, PredictionDecoder, LabelsGen, PredGen], 1)
