from cvlab.diagram.elements.base import *
from tensorflow import keras
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
from random import shuffle
import math

TXT_FILTER = "TXT (*.txt)"


def list_to_dict(lst):
    return {k: v for v, k in enumerate(lst)}


class _BatchLoader(InputElement):
    """Base element for loading train or test data for neural networks operations"""

    def __init__(self):
        super(_BatchLoader, self).__init__()
        self.batch_size = None
        self.epochs = None
        self.total_batches_sent = 0
        self.number_of_batches = None
        self.dataset = None
        self.path = None
        self.classes = None
        self.to_categorical = True
        self.batch_notifier = threading.Event()

    def get_attributes(self):
        """Returns common attributes for all batch loaders"""
        return [], \
               [Output("images"),
                Output("labels", name="batch labels", preview_enabled=False),
                Output("classes", name="all labels", preview_enabled=False)], \
               [IntParameter("epochs", value=1, min_=1, max_=100),
                IntParameter("batch_size", name="batch size", value=64, min_=1, max_=2048),
                ComboboxParameter("to_categorical", name="to categorical", values=[("Yes", True), ("No", False)]),
                ButtonParameter("restart", self.restart, "Restart")]

    def update_parameters(self):
        """Updates values of parameters and returns True if dataset needs to be reloaded"""
        epochs = self.parameters["epochs"].get()
        batch_size = self.parameters["batch_size"].get()
        to_categorical = self.parameters["to_categorical"].get()
        should_reload = False

        if self.batch_size != batch_size:
            self.batch_size = batch_size
            should_reload = True
        self.to_categorical = to_categorical
        self.epochs = epochs

        return should_reload

    def recalculate(self, refresh_parameters, refresh_structure, force_break, force_units_recalc=False):
        self.batch_notifier.set()
        ThreadedElement.recalculate(self, refresh_parameters, refresh_structure, force_break, force_units_recalc)

    def delete(self):
        self.batch_notifier.set()
        ThreadedElement.delete(self)

    def restart(self):
        """Starts sending batches from the beginning"""
        self.total_batches_sent = 0
        self.recalculate(True, False, True)

    def process(self):
        should_regenerate_dataset = self.update_parameters()
        if should_regenerate_dataset:
            self.total_batches_sent = 0
            self.generate_dataset()
            self.classes = list_to_dict(self.get_classes())   # classes are stored as pairs (class name: index)
            self.outputs["classes"].put(Data(self.classes.keys()))
        self.send_batches()

    def generate_dataset(self):
        """Generates new set of data"""
        pass

    def send_batches(self):
        while self.total_batches_sent < self.number_of_batches * self.epochs:
            self.set_state(Element.STATE_BUSY)
            self.may_interrupt()
            images, labels = self.get_next_batch()
            image_sequence = Sequence([Data(image) for image in images])
            labels = [self.classes[label] for label in labels]  # get index for each class from class dictionary
            if self.to_categorical:
                # converts a class vector to binary class matrix
                labels = keras.utils.to_categorical(labels, len(self.classes))
            label_sequence = Data(labels)

            self.set_state(Element.STATE_READY)
            self.notify_state_changed()
            if self.total_batches_sent != 0:
                self.batch_notifier.wait()  # wait while previous batch is being processed
            self.batch_notifier.clear()

            self.outputs["images"].put(image_sequence)
            self.outputs["labels"].put(label_sequence)
            self.total_batches_sent += 1

    def get_next_batch(self):
        """Returns single batch of data (images and labels) of given batch_size"""
        return

    def get_classes(self):
        """Returns sorted list of all possible classes for the dataset"""
        return


class _BatchLoaderFromDisk(_BatchLoader):
    """Base element for loading data from disk. Deriving elements should contain path parameter."""
    def update_parameters(self):
        path = self.parameters["path"].get()
        should_reload = False

        if self.path != path:
            self.path = path
            should_reload = True

        return super().update_parameters() or should_reload

    def get_next_batch(self):
        start = self.total_batches_sent % self.number_of_batches * self.batch_size
        end = start + self.batch_size
        labeled_image_paths = self.dataset[start:end]

        images = []
        labels = []
        for image_path, label in labeled_image_paths:
            try:
                images.append(plt.imread(image_path))
                labels.append(label)
            except (SyntaxError, UnidentifiedImageError):  # plt.imread() file exception (f.e. not a PNG file exception)
                continue
        return images, labels

    def get_classes(self):
        classes = list(set((image[1] for image in self.dataset)))
        classes.sort()
        return classes


class BatchLoaderFromFile(_BatchLoaderFromDisk):
    name = "File image batch loader"
    comment = "Loads batches of images with labels from text file.\n" \
              "File should contain one line per image in format <path_to_image image_class>." \
              "For the most efficient processing set the batch size to the power of 2."

    def get_attributes(self):
        base_inputs, base_outputs, base_parameters = super().get_attributes()
        new_parameters = [PathParameter("path", name="file (.txt)", value="", extension_filter=TXT_FILTER)]
        parameters = new_parameters + base_parameters
        return base_inputs, base_outputs, parameters

    def generate_dataset(self):
        with open(self.path) as file:
            labeled_image_paths = [line.rstrip().split(" ") for line in file]

        shuffle(labeled_image_paths)  # randomize order of images
        self.dataset = labeled_image_paths
        self.number_of_batches = math.ceil(len(self.dataset) / self.batch_size)


class BatchLoaderFromDirectory(_BatchLoaderFromDisk):
    name = "Directory image batch loader"
    comment = "Loads batches of images from directory.\n" \
              "Target directory should contain one subdirectory per class.\n" \
              "For the most efficient processing set the batch size to the power of 2."

    def get_attributes(self):
        base_inputs, base_outputs, base_parameters = super().get_attributes()
        new_parameters = [DirectoryParameter("path", name="directory path", value="")]
        parameters = new_parameters + base_parameters
        return base_inputs, base_outputs, parameters

    def generate_dataset(self):
        labeled_image_paths = []
        subdirectories_paths = [f.path for f in os.scandir(self.path) if f.is_dir()]
        for sub_path in subdirectories_paths:
            label = os.path.basename(sub_path)  # name of subdirectory indicates image's label
            for root, dirs, files in os.walk(sub_path):
                for file in files:
                    labeled_image_paths.append([os.path.join(root, file), label])

        shuffle(labeled_image_paths)  # randomize order of images
        self.dataset = labeled_image_paths
        self.number_of_batches = math.ceil(len(self.dataset) / self.batch_size)


class KerasDatasetBatchLoader(_BatchLoader):
    name = "Built-in datasets loader"
    comment = "Loads train or test dataset from keras built-in image datasets.\n" \
              "For the most efficient processing set the batch size to the power of 2.\n" \
              "For more information see https://www.tensorflow.org/api_docs/python/tf/keras/datasets"
    type = None
    dataset_name = None

    keras_datasets = {
        "MNIST": keras.datasets.mnist,
        "CIFAR10": keras.datasets.cifar10,
        "CIFAR100": keras.datasets.cifar100,
        "Fashion MNIST": keras.datasets.fashion_mnist
    }

    def get_attributes(self):
        dataset_names = {key: key for key in self.keras_datasets.keys()}
        base_inputs, base_outputs, base_parameters = super().get_attributes()
        new_parameters = [ComboboxParameter("dataset", dataset_names), ComboboxParameter("type", {"train": 0, "test": 1})]
        parameters = new_parameters + base_parameters
        return base_inputs, base_outputs, parameters

    def update_parameters(self):
        dataset_name = self.parameters["dataset"].get()
        type = self.parameters["type"].get()
        should_reload = False

        if self.dataset_name != dataset_name:
            self.dataset_name = dataset_name
            should_reload = True
        if self.type != type:
            self.type = type
            should_reload = True

        return super().update_parameters() or should_reload

    def generate_dataset(self):
        dataset = self.keras_datasets.get(self.dataset_name)
        self.dataset = dataset.load_data()[self.type]
        self.number_of_batches = math.ceil(len(self.dataset[0]) / self.batch_size)

    def get_next_batch(self):
        start = self.total_batches_sent % self.number_of_batches * self.batch_size
        end = start + self.batch_size
        images = self.dataset[0][start:end]
        labels = self.dataset[1][start:end]
        if labels.ndim != 1:
            labels = np.squeeze(labels, axis=1)
        return images, labels

    def get_classes(self):
        classes = self.dataset[1]
        if classes.ndim != 1:
            classes = np.squeeze(classes, axis=1)
        classes = list(set(list(classes)))  # get unique values
        classes.sort(key=lambda cl: str(cl))   # making sure that classes are in alphanumerical order
        return classes


register_elements("Image batch loading", [BatchLoaderFromFile, BatchLoaderFromDirectory, KerasDatasetBatchLoader], 20)
