import os


def get_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


# paths
MODULE_DIR = os.path.abspath(__file__ + "/..").replace("\\", "/")
IMAGES_DIR = get_dir(MODULE_DIR + "/images")
LOGS_DIR = get_dir(MODULE_DIR + "/logs")
SAMPLES_DIR = MODULE_DIR + "/samples"

PLUGIN_PRIORITY = 1000
