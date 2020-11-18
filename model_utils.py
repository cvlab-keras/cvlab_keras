import cv2

from tensorflow.keras.models import Model
from tensorflow.keras import utils
from io import StringIO

from cvlab.view.widgets import ActionImage
from cvlab_keras.shared import IMAGES_DIR


def model_to_image(model: Model):
    png_path = IMAGES_DIR + "/tmp_model.png"

    utils.plot_model(model, png_path, show_shapes=True)
    image = cv2.imread(png_path)
    return image


def model_to_string(model: Model, line_length=None, positions=None):
    stream = StringIO()
    model.summary(line_length=line_length, positions=positions, print_fn=lambda line: stream.write(line + '\n'))
    model_summary = stream.getvalue()
    stream.close()
    return model_summary


def set_model(action_image, model: Model):
    narrow_elem_lim = 300  # pixel upper limit for element to be considered narrow
    # number of layers (upper limit) for model to be considered :
    small_model_lim = 15  # small
    medium_model_lim = 30
    if model.layers.__len__() <= small_model_lim:  # small models - image preview
        image = model_to_image(model)
        action_image.data_type = ActionImage.DATA_TYPE_IMAGE
        action_image.set_image(image)
    else:  # larger models - text preview
        preview_px_width = action_image.previews_container.preview_size * 1.9  # multiply to fill whole container
        font_px_size = 6  # TODO (cvlab_keras) self.font.pixelSize() returns 10 even though font really is 6px
        line_len = int(preview_px_width / font_px_size)

        if preview_px_width <= narrow_elem_lim:  # display summary in 2 columns for narrow elements
            positions = [0.5, 1, 1, 1]
        else:  # and in 3 columns for wide elements
            positions = [0.5, 0.9, 1, 1]

        model_string = model_to_string(model, line_length=line_len, positions=positions)

        line_count = model_string.count('\n')
        if line_count > medium_model_lim:  # for largest models display specified number of first layers
            n_lines = medium_model_lim * 2  # multiply by 2 to take line separators into account
            separate_lines = model_string.split('\n')
            model_string = separate_lines[0] + " - " + str(model.layers.__len__() ) +" layers\n"
            model_string += '\n'.join(separate_lines[1:n_lines])  # join following n-1 lines separated by endline
            model_string += "\n(...)"

        action_image.data_type = ActionImage.DATA_TYPE_TEXT
        action_image.set_text(model_string)

