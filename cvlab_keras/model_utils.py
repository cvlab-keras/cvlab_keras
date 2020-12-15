import cv2 as cv

from tensorflow.keras.models import Model
from tensorflow.keras import utils
from io import StringIO

from cvlab.view.widgets import ActionImage
from cvlab_keras.cvlab_keras.shared import IMAGES_DIR


def model_to_image(model: Model):
    png_path = IMAGES_DIR + "/tmp_model.png"
    utils.plot_model(model, png_path, show_shapes=True)
    image = cv.imread(png_path)
    return image


def model_to_string(model: Model, line_length=None, positions=None):
    stream = StringIO()
    model.summary(line_length=line_length, positions=positions, print_fn=lambda line: stream.write(line + '\n'))
    model_summary = stream.getvalue()
    stream.close()
    return model_summary


def set_model(action_image, model: Model):
    narrow_elem_lim = 300  # pixel upper limit for element to be considered narrow
    model_size_lim = 30  # don't display all lines for model preview that have more lines than this value

    preview_px_width = action_image.previews_container.preview_size * 1.9  # multiply to fill whole container
    font_px_size = 6  # TODO (cvlab_keras) self.font.pixelSize() returns 10 even though font really is 6px
    line_len = int(preview_px_width / font_px_size)

    if preview_px_width <= narrow_elem_lim:  # display summary in 2 columns for narrow elements
        positions = [0.5, 1, 1, 1]
    else:  # and in 3 columns for wide elements
        positions = [0.5, 0.9, 1, 1]

    model_string = model_to_string(model, line_length=line_len, positions=positions)

    line_count = model_string.count('\n')
    # if view is not expanded display few first and last layers for bigger models
    if not action_image.text_preview_expanded and line_count > model_size_lim:
        end_lines_count = 10
        start_lines_index = model_size_lim - end_lines_count
        end_lines_index = line_count - end_lines_count  # last few lines
        remaining_lines_count = line_count - start_lines_index - end_lines_count

        separate_lines = model_string.split('\n')
        model_string = separate_lines[0] + " - " + str(model.layers.__len__()) + " layers\n"
        model_string += '\n'.join(separate_lines[1:start_lines_index])  # join following n-1 lines separated by endline
        model_string += "\n\n({} more lines...)\n\n".format(remaining_lines_count)
        model_string += '\n'.join(separate_lines[end_lines_index:])  # include last few lines

    action_image.data_type = ActionImage.DATA_TYPE_TEXT
    action_image.set_text(model_string)

