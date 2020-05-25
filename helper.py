import datetime
import logging
import math
import os
import sys
import time
from os import listdir

import numpy as np
import tensorflow as tf
from PIL import Image
from os.path import isfile, join
from scipy import misc
from skimage.measure import compare_psnr, compare_ssim


import configparser
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS

# Model
flags.DEFINE_integer("filters", 96, "Number of CNN Filters")
flags.DEFINE_integer("min_filters", 32, "Number of the last CNN Filters")
flags.DEFINE_integer("nin_filters", 64, "Number of Filters of A1 in Reconstruction network")
flags.DEFINE_integer("nin_filters2", 0, "Number of Filters of B1 and B2 in Reconstruction. If 0, it will be half of A1")
flags.DEFINE_integer("cnn_size", 3, "Size of CNN filters")
flags.DEFINE_integer("last_cnn_size", 1, "Size of Last CNN filters")
flags.DEFINE_integer("layers", 7, "Number of layers of CNNs")
flags.DEFINE_boolean("nin", True, "Use Network In Network")
flags.DEFINE_boolean("bicubic_init", True, "make bicubic interpolation values as initial input of x2")
flags.DEFINE_float("dropout", 0.8, "For dropout value for  value. Don't use if it's 1.0.")
flags.DEFINE_string("activator", "prelu", "Activator. can be [relu, leaky_relu, prelu, sigmoid, tanh]")
flags.DEFINE_float("filters_decay_gamma", 1.2, "Gamma")

# Training
flags.DEFINE_string("initializer", "he", "Initializer for weights can be [uniform, stddev, xavier, he, identity, zero]")
flags.DEFINE_float("weight_dev", 0.01, "Initial weight stddev (won't be used when you use he or xavier initializer)")
flags.DEFINE_float("l2_decay", 0.0001, "l2_decay")
flags.DEFINE_string("optimizer", "adam", "Optimizer can be [gd, momentum, adadelta, adagrad, adam, rmsprop]")
flags.DEFINE_float("beta1", 0.1, "Beta1 for adam optimizer")
flags.DEFINE_float("beta2", 0.1, "Beta2 for adam optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum for momentum optimizer and rmsprop optimizer")
flags.DEFINE_integer("batch_num", 40, "Number of mini-batch images for training")
flags.DEFINE_integer("batch_image_size", 32, "Image size for mini-batch")
flags.DEFINE_integer("stride_size", 0, "Stride size for mini-batch. If it is 0, use half of batch_image_size")

# Learning Rate Control for Training
flags.DEFINE_float("initial_lr", 0.001, "Initial learning rate")
flags.DEFINE_float("lr_decay", 0.5, "Learning rate decay rate when it does not reduced during specific epoch")
flags.DEFINE_integer("lr_decay_epoch", 5, "Decay learning rate when loss does not decrease (5)")
flags.DEFINE_float("end_lr", 2e-5, "Training end learning rate (2e-5")

# Dataset or Others
flags.DEFINE_string("test_dataset", "set5", "Directory of Test dataset [set5, set14, bsd100, urban100]")
flags.DEFINE_string("dataset", "yang91", "Training dataset dir. [yang91, general100, bsd200]")
flags.DEFINE_integer("tests", 1, "Number of training tests")

# Image Processing
flags.DEFINE_integer("scale", 2, "Scale factor for Super Resolution (can be 2 or more)")
flags.DEFINE_float("max_value", 255, "For normalize image pixel value")
flags.DEFINE_integer("channels", 1, "Number of image channels used. Use only Y of YCbCr when channels=1.")
flags.DEFINE_boolean("jpeg_mode", False, "Turn on or off jpeg mode when converting from rgb to ycbcr")

# Environment (all directory name should not contain '/' after )
flags.DEFINE_string("checkpoint_dir", "models", "Directory for checkpoints")
flags.DEFINE_string("graph_dir", "graphs", "Directory for graphs")
flags.DEFINE_string("data_dir", "data", "Directory for original images")
flags.DEFINE_string("batch_dir", "batch_data", "Directory for training batch images")
flags.DEFINE_string("output_dir", "output", "Directory for output test images")
flags.DEFINE_string("tf_log_dir", "tf_log", "Directory for tensorboard log")
flags.DEFINE_string("log_filename", "log.txt", "log filename")
flags.DEFINE_string("model_name", "", "model name for save files and tensorboard log")
flags.DEFINE_string("load_model_name", "", "Filename of model loading before start [filename or 'default']")

# Debugging or Logging
flags.DEFINE_boolean("debug", False, "Display each calculated MSE and weight variables")
flags.DEFINE_boolean("initialise_tf_log", True, "Clear all tensorboard log before start")
flags.DEFINE_boolean("save_loss", True, "Save loss")
flags.DEFINE_boolean("save_weights", False, "Save weights and biases")
flags.DEFINE_boolean("save_images", False, "Save CNN weights as images")
flags.DEFINE_integer("save_images_num", 10, "Number of CNN images saved")


def get():
    print("Python Interpreter version:%s" % sys.version[:3])
    print("tensorflow version:%s" % tf.__version__)
    return FLAGS


class Timer:
    def __init__(self, timer_count=100):
        self.times = np.zeros(timer_count)
        self.start_times = np.zeros(timer_count)
        self.counts = np.zeros(timer_count)
        self.timer_count = timer_count

    def start(self, timer_id):
        self.start_times[timer_id] = time.time()

    def end(self, timer_id):
        self.times[timer_id] += time.time() - self.start_times[timer_id]
        self.counts[timer_id] += 1

    def print(self):
        for i in range(self.timer_count):
            if self.counts[i] > 0:
                total = 0
                print("Average of %d: %s[ms]" % (i, "{:,}".format(self.times[i] * 1000 / self.counts[i])))
                total += self.times[i]
                print("Total of %d: %s" % (i, "{:,}".format(total)))


# utilities for save / load

class LoadError(Exception):
    def __init__(self, message):
        self.message = message


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_files_in_directory(path):
    if not path.endswith('/'):
        path = path + "/"
    file_list = [path + f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]
    return file_list


def remove_generic(path, __func__):
    try:
        __func__(path)
    except OSError as error:
        print("OS error: {0}".format(error))


def clean_dir(path):
    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    for x in files:
        full_path = os.path.join(path, x)
        if os.path.isfile(full_path):
            f = os.remove
            remove_generic(full_path, f)
        elif os.path.isdir(full_path):
            clean_dir(full_path)
            f = os.rmdir
            remove_generic(full_path, f)


def set_logging(filename, stream_log_level, file_log_level, tf_log_level):
    stream_log = logging.StreamHandler()
    stream_log.setLevel(stream_log_level)

    file_log = logging.FileHandler(filename=filename)
    file_log.setLevel(file_log_level)

    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(stream_log)
    logger.addHandler(file_log)
    logger.setLevel(min(stream_log_level, file_log_level))

    tf.logging.set_verbosity(tf_log_level)

    # optimizing logging
    logging._srcfile = None
    logging.logThreads = 0
    logging.logProcesses = 0


def save_image(filename, image, print_console=True):
    if len(image.shape) >= 3 and image.shape[2] == 1:
        image = image.reshape(image.shape[0], image.shape[1])

    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    image = misc.toimage(image, cmin=0, cmax=255)  # to avoid range rescaling
    misc.imsave(filename, image)

    if print_console:
        print("Saved [%s]" % filename)


def save_image_data(filename, image):
    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    np.save(filename, image)
    print("Saved [%s]" % filename)


def convert_rgb_to_y(image, jpeg_mode=True, max_value=255.0):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114]])
        y_image = image.dot(xform.T)
    else:
        xform = np.array([[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0]])
        y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

    return y_image


def convert_rgb_to_ycbcr(image, jpeg_mode=True, max_value=255):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, [1, 2]] += max_value / 2
    else:
        xform = np.array(
            [[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
             [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
        ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

    return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=True, max_value=255.0):
    if len(y_image.shape) <= 2:
        y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image[:, :, 0]
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=jpeg_mode, max_value=max_value)


def convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=True, max_value=255.0):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    if jpeg_mode:
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array([[1, 0, 1.402], [1, - 0.344, - 0.714], [1, 1.772, 0]])
        rgb_image = rgb_image.dot(xform.T)
    else:
        rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - (16.0 * max_value / 256.0)
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array(
            [[max_value / 219.0, 0, max_value * 0.701 / 112.0],
             [max_value / 219, - max_value * 0.886 * 0.114 / (112 * 0.587),
              - max_value * 0.701 * 0.299 / (112 * 0.587)],
             [max_value / 219.0, max_value * 0.886 / 112.0, 0]])
        rgb_image = rgb_image.dot(xform.T)

    return rgb_image


def set_image_alignment(image, alignment):
    alignment = int(alignment)
    width, height = image.shape[1], image.shape[0]
    width = (width // alignment) * alignment
    height = (height // alignment) * alignment

    if image.shape[1] != width or image.shape[0] != height:
        image = image[:height, :width, :]

    if len(image.shape) >= 3 and image.shape[2] >= 4:
        image = image[:, :, 0:3]

    return image


def resize_image_by_bicubic(image, scale):
    size = [int(image.shape[0] * scale), int(image.shape[1] * scale)]
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    tf_image = tf.image.resize_bicubic(image, size=size)
    image = tf_image.eval()
    return image.reshape(image.shape[1], image.shape[2], image.shape[3])


def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # the image may has an alpha channel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image


def load_image(filename, width=0, height=0, channels=0, alignment=0, print_console=True):
    if not os.path.isfile(filename):
        raise LoadError("File not found [%s]" % filename)
    image = misc.imread(filename)

    if len(image.shape) == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)
    if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
        raise LoadError("Attributes mismatch")
    if channels != 0 and image.shape[2] != channels:
        raise LoadError("Attributes mismatch")
    if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
        raise LoadError("Attributes mismatch")

    if print_console:
        print("Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2]))
    return image


def load_image_data(filename, width=0, height=0, channels=0, alignment=0, print_console=True):
    if not os.path.isfile(filename):
        raise LoadError("File not found")
    image = np.load(filename)

    if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
        raise LoadError("Attributes mismatch")
    if channels != 0 and image.shape[2] != channels:
        raise LoadError("Attributes mismatch")
    if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
        raise LoadError("Attributes mismatch")

    if print_console:
        print("Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2]))
    return image


def get_split_images(image, window_size, stride=None, enable_duplicate=True):
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.reshape(image.shape[0], image.shape[1])

    window_size = int(window_size)
    size = image.itemsize  # byte size of each value
    height, width = image.shape
    if stride is None:
        stride = window_size
    else:
        stride = int(stride)

    if height < window_size or width < window_size:
        return None

    new_height = 1 + (height - window_size) // stride
    new_width = 1 + (width - window_size) // stride

    shape = (new_height, new_width, window_size, window_size)
    strides = size * np.array([width * stride, stride, width, 1])
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    windows = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3], 1)

    if enable_duplicate:
        extra_windows = []
        if (height - window_size) % stride != 0:
            for x in range(0, width - window_size, stride):
                extra_windows.append(image[height - window_size - 1:height - 1, x:x + window_size:])

        if (width - window_size) % stride != 0:
            for y in range(0, height - window_size, stride):
                extra_windows.append(image[y: y + window_size, width - window_size - 1:width - 1])

        if len(extra_windows) > 0:
            org_size = windows.shape[0]
            windows = np.resize(windows,
                                [org_size + len(extra_windows), windows.shape[1], windows.shape[2], windows.shape[3]])
            for i in range(len(extra_windows)):
                extra_windows[i] = extra_windows[i].reshape([extra_windows[i].shape[0], extra_windows[i].shape[1], 1])
                windows[org_size + i] = extra_windows[i]

    return windows


def xavier_cnn_initializer(shape, uniform=True):
    fan_in = shape[0] * shape[1] * shape[2]
    fan_out = shape[0] * shape[1] * shape[3]
    n = fan_in + fan_out
    if uniform:
        init_range = math.sqrt(6.0 / n)
        return tf.random_uniform(shape, minval=-init_range, maxval=init_range)
    else:
        stddev = math.sqrt(3.0 / n)
        return tf.truncated_normal(shape=shape, stddev=stddev)


def he_initializer(shape):
    n = shape[0] * shape[1] * shape[2]
    stddev = math.sqrt(2.0 / n)
    return tf.truncated_normal(shape=shape, stddev=stddev)


def weight(shape, stddev=0.01, name="weight", uniform=False, initializer="stddev"):
    if initializer == "xavier":
        initial = xavier_cnn_initializer(shape, uniform=uniform)
    elif initializer == "he":
        initial = he_initializer(shape)
    elif initializer == "uniform":
        initial = tf.random_uniform(shape, minval=-2.0 * stddev, maxval=2.0 * stddev)
    elif initializer == "stddev":
        initial = tf.truncated_normal(shape=shape, stddev=stddev)
    elif initializer == "identity":
        initial = he_initializer(shape)
        if len(shape) == 4:
            initial = initial.eval()
            i = shape[0] // 2
            j = shape[1] // 2
            for k in range(min(shape[2], shape[3])):
                initial[i][j][k][k] = 1.0
    else:
        initial = tf.zeros(shape)

    return tf.Variable(initial, name=name)


def bias(shape, initial_value=0.0, name=None):
    initial = tf.constant(initial_value, shape=shape)

    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


# utilities for logging -----

def add_summaries(scope_name, model_name, var, save_stddev=True, save_mean=False, save_max=False, save_min=False):
    with tf.name_scope(scope_name):
        mean_var = tf.reduce_mean(var)
        if save_mean:
            tf.summary.scalar("mean/" + model_name, mean_var)

        if save_stddev:
            stddev_var = tf.sqrt(tf.reduce_mean(tf.square(var - mean_var)))
            tf.summary.scalar("stddev/" + model_name, stddev_var)

        if save_max:
            tf.summary.scalar("max/" + model_name, tf.reduce_max(var))

        if save_min:
            tf.summary.scalar("min/" + model_name, tf.reduce_min(var))
        tf.summary.histogram(model_name, var)


def get_now_date():
    d = datetime.datetime.today()
    return "%s/%s/%s %s:%s:%s" % (d.year, d.month, d.day, d.hour, d.minute, d.second)


def get_loss_image(image1, image2, scale=1.0, border_size=0):
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    if image1.dtype == np.uint8:
        image1 = image1.astype(np.double)
    if image2.dtype == np.uint8:
        image2 = image2.astype(np.double)

    loss_image = np.multiply(np.square(np.subtract(image1, image2)), scale)
    loss_image = np.minimum(loss_image, 255.0)
    loss_image = loss_image[border_size:-border_size, border_size:-border_size, :]

    return loss_image


def compute_mse(image1, image2, border_size=0):
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    if image1.dtype != np.uint8:
        image1 = image1.astype(np.int)
    image1 = image1.astype(np.double)

    if image2.dtype != np.uint8:
        image2 = image2.astype(np.int)
    image2 = image2.astype(np.double)

    mse = 0.0
    for i in range(border_size, image1.shape[0] - border_size):
        for j in range(border_size, image1.shape[1] - border_size):
            for k in range(image1.shape[2]):
                error = image1[i, j, k] - image2[i, j, k]
                mse += error * error

    return mse / ((image1.shape[0] - 2 * border_size) * (image1.shape[1] - 2 * border_size) * image1.shape[2])


def compute_psnr_and_ssim(image1, image2, border_size=0):
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    image1 = image1.astype(np.double)
    image2 = image2.astype(np.double)

    if border_size > 0:
        image1 = image1[border_size:-border_size, border_size:-border_size, :]
        image2 = image2[border_size:-border_size, border_size:-border_size, :]

    psnr = compare_psnr(image1, image2, data_range=255)
    ssim = compare_ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                        sigma=1.5, data_range=255)
    return psnr, ssim


def print_filter_weights(tensor):
    print("Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape())))
    weight = tensor.eval()
    for i in range(weight.shape[3]):
        values = ""
        for x in range(weight.shape[0]):
            for y in range(weight.shape[1]):
                for c in range(weight.shape[2]):
                    values += "%2.3f " % weight[y][x][c][i]
        print(values)
    print("\n")


def print_filter_biases(tensor):
    print("Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape())))
    bias = tensor.eval()
    values = ""
    for i in range(bias.shape[0]):
        values += "%2.3f " % bias[i]
    print(values + "\n")


def get_psnr(mse, max_value=255.0):
    if mse is None or mse == float('Inf') or mse == 0:
        psnr = 0
    else:
        psnr = 20 * math.log(max_value / math.sqrt(mse), 10)
    return psnr


def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if output_detail:
            logging.info(parameters_string)
        logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))



INPUT_IMAGE_DIR = "input"
INTERPOLATED_IMAGE_DIR = "interpolated"
TRUE_IMAGE_DIR = "true"


def convert_to_multi_channel_image(multi_channel_image, image, scale):
    height = multi_channel_image.shape[0]
    width = multi_channel_image.shape[1]

    for y in range(height):
        for x in range(width):
            for y2 in range(scale):
                for x2 in range(scale):
                    multi_channel_image[y, x, y2 * scale + x2] = image[y * scale + y2, x * scale + x2, 0]


def convert_from_multi_channel_image(image, multi_channel_image, scale):
    height = multi_channel_image.shape[0]
    width = multi_channel_image.shape[1]

    for y in range(height):
        for x in range(width):
            for y2 in range(scale):
                for x2 in range(scale):
                    image[y * scale + y2, x * scale + x2, 0] = multi_channel_image[y, x, y2 * scale + x2]


def load_input_image(filename, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True,
                     jpeg_mode=False, print_console=True):
    image = load_image(filename, print_console=print_console)
    return build_input_image(image, width, height, channels, scale, alignment, convert_ycbcr, jpeg_mode)


def build_input_image(image, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True, jpeg_mode=False):
    """
    build input image from file.
    crop, adjust the image alignment for the scale factor, resize, convert color space.
    """

    if width != 0 and height != 0:
        if image.shape[0] != height or image.shape[1] != width:
            x = (image.shape[1] - width) // 2
            y = (image.shape[0] - height) // 2
            image = image[y: y + height, x: x + width, :]

    if image.shape[2] >= 4:
        image = image[:, :, 0:3]

    if alignment > 1:
        image = set_image_alignment(image, alignment)

    if scale != 1:
        image = resize_image_by_pil(image, 1.0 / scale)

    if channels == 1 and image.shape[2] == 3:
        if convert_ycbcr:
            image = convert_rgb_to_y(image, jpeg_mode=jpeg_mode)
    else:
        if convert_ycbcr:
            image = convert_rgb_to_ycbcr(image, jpeg_mode=jpeg_mode)

    return image


class DataSet:
    def __init__(self, batch_image_size, channels=1, scale=1, max_value=255.0, alignment=0, jpeg_mode=False):

        self.batch_image_size = batch_image_size
        self.max_value = max_value
        self.channels = channels
        self.scale = scale
        self.max_value = max_value
        self.alignment = alignment
        self.jpeg_mode = jpeg_mode

        self.count = 0
        self.images = None
        self.quad_images = None

    def load_test_image(self, filename):

        image = load_input_image(filename, channels=self.channels, scale=1, alignment=self.alignment,
                                 jpeg_mode=self.jpeg_mode, print_console=False)
        if self.max_value != 255.0:
            image = np.multiply(image, self.max_value / 255.0)

        return image

    def load_input_image(self, filename, rescale=False, resampling_method="bicubic"):

        image = load_input_image(filename, channels=self.channels, scale=self.scale, alignment=self.alignment,
                                 jpeg_mode=self.jpeg_mode, print_console=True)
        if self.max_value != 255.0:
            image = np.multiply(image, self.max_value / 255.0)

        if rescale:
            rescaled_image = resize_image_by_pil(image, self.scale, resampling_method=resampling_method)
            return image, rescaled_image
        else:
            return image

    def load_batch_images(self, batch_dir, input_batch, count):

        print("Loading %d batch images from %s for [%s]" % (count, batch_dir, "input" if input_batch else "true"))

        self.count = count
        if input_batch:
            self.images = np.zeros(shape=[count, self.batch_image_size, self.batch_image_size, 1])  # type: np.ndarray
        else:
            self.images = None
        self.quad_images = np.zeros(
            shape=[count, self.batch_image_size, self.batch_image_size, self.scale * self.scale])  # type: np.ndarray

        for i in range(count):
            if input_batch:
                self.images[i] = load_image(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % i,
                                                 print_console=False)
                quad_image = load_image(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % i,
                                             print_console=False)
            else:
                quad_image = load_image(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.bmp" % i, print_console=False)

            convert_to_multi_channel_image(self.quad_images[i], quad_image, self.scale)

            if i % 1000 == 0:
                print('.', end='', flush=True)

        print("Finished")


class DataSets:
    def __init__(self, scale, batch_image_size, stride_size, channels=1,
                 jpeg_mode=False, max_value=255.0, resampling_method="nearest"):

        self.scale = scale
        self.batch_image_size = batch_image_size
        self.stride = stride_size
        self.channels = channels
        self.jpeg_mode = jpeg_mode
        self.max_value = max_value
        self.resampling_method = resampling_method

        self.input = DataSet(batch_image_size, channels=channels, scale=scale, alignment=scale, jpeg_mode=jpeg_mode,
                             max_value=max_value)
        self.true = DataSet(batch_image_size, channels=channels, scale=scale, alignment=scale, jpeg_mode=jpeg_mode,
                            max_value=max_value)

    def build_batch(self, data_dir, batch_dir):
        """ load from input files. Then save batch images on file to reduce memory consumption. """

        print("Building batch images for %s..." % batch_dir)
        filenames = get_files_in_directory(data_dir)
        images_count = 0

        make_dir(batch_dir)
        clean_dir(batch_dir)
        make_dir(batch_dir + "/" + INPUT_IMAGE_DIR)
        make_dir(batch_dir + "/" + INTERPOLATED_IMAGE_DIR)
        make_dir(batch_dir + "/" + TRUE_IMAGE_DIR)

        for filename in filenames:
            output_window_size = self.batch_image_size * self.scale
            output_window_stride = self.stride * self.scale
            input_image, input_bicubic_image = self.input.load_input_image(filename, rescale=True,
                                                                           resampling_method=self.resampling_method)
            test_image = self.true.load_test_image(filename)

            # split into batch images
            input_batch_images = get_split_images(input_image, self.batch_image_size, stride=self.stride)
            input_bicubic_batch_images = get_split_images(input_bicubic_image, output_window_size,
                                                               stride=output_window_stride)
            if input_batch_images is None or input_bicubic_batch_images is None:
                continue

            input_count = input_batch_images.shape[0]

            test_batch_images = get_split_images(test_image, output_window_size, stride=output_window_stride)

            for i in range(input_count):
                # save_image_data(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.npy" % images_count,
                #                      input_batch_images[i])
                # save_image_data(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.npy" % images_count,
                #                 input_bicubic_batch_images[i])
                # save_image_data(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.npy" % images_count,
                #                      test_batch_images[i])
                save_image(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % images_count, input_batch_images[i])
                save_image(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % images_count,
                                input_bicubic_batch_images[i])
                save_image(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.bmp" % images_count, test_batch_images[i])

                images_count += 1

        print("%d mini-batch images are built(saved)." % images_count)

        config = configparser.ConfigParser()
        config.add_section("batch")
        config.set("batch", "count", str(images_count))
        config.set("batch", "scale", str(self.scale))
        config.set("batch", "batch_image_size", str(self.batch_image_size))
        config.set("batch", "stride", str(self.stride))
        config.set("batch", "channels", str(self.channels))
        config.set("batch", "jpeg_mode", str(self.jpeg_mode))
        config.set("batch", "max_value", str(self.max_value))

        with open(batch_dir + "/batch_images.ini", "w") as configfile:
            config.write(configfile)

    def load_batch_train(self, batch_dir):
        """ load already built batch images. """

        config = configparser.ConfigParser()
        config.read(batch_dir + "/batch_images.ini")
        count = config.getint("batch", "count")

        self.input.count = count
        self.true.count = count

    def load_batch_test(self, batch_dir):
        """ load already built batch images. """

        config = configparser.ConfigParser()
        config.read(batch_dir + "/batch_images.ini")
        count = config.getint("batch", "count")

        self.input.load_batch_images(batch_dir, True, count)
        self.true.load_batch_images(batch_dir, False, count)

    def is_batch_exist(self, batch_dir):
        if not os.path.isdir(batch_dir):
            return False

        config = configparser.ConfigParser()
        try:
            with open(batch_dir + "/batch_images.ini") as f:
                config.read_file(f)

            if config.getint("batch", "count") <= 0:
                return False

            if config.getint("batch", "scale") != self.scale:
                return False
            if config.getint("batch", "batch_image_size") != self.batch_image_size:
                return False
            if config.getint("batch", "stride") != self.stride:
                return False
            if config.getint("batch", "channels") != self.channels:
                return False
            if config.getboolean("batch", "jpeg_mode") != self.jpeg_mode:
                return False
            if config.getfloat("batch", "max_value") != self.max_value:
                return False

            return True

        except IOError:
            return False

