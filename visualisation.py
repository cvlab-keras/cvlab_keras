from random import *

import matplotlib.pyplot as plt
import time
import webbrowser
import subprocess
import datetime
import pause
import tensorflow as tf

from cvlab.diagram.elements.base import *

plt.style.use('ggplot')


class Visualisation(NormalElement):
    name = "Live plotting"
    comment = "Generates plot from loss function in real time"

    def __init__(self):
        super(Visualisation, self).__init__()
        self.fig_num = 3
        self.counter = 0
        self.loss = []
        self.accuracy = []
        self.batch_num = []
        self.img = []
        self.figures = []
        self.lines = []
        self.ax = []
        self.combobox = ""

        self.set_all_figures()

        self.configure_plot(self.ax[0], 'Loss function for model training', 'Batch', 'Loss')
        self.configure_plot(self.ax[1], 'Accuracy function for model training', 'Batch', 'Accuracy')
        self.configure_plot(self.ax[2][0], 'Loss function for model training', 'Batch', 'Loss')
        self.configure_plot(self.ax[2][1], 'Accuracy function for model training', 'Batch', 'Accuracy')

    def set_all_figures(self):
        last_index = self.fig_num - 1
        for i in range(self.fig_num):
            if i == last_index:
                fig, ax = plt.subplots(nrows=last_index, ncols=1)
                fig.tight_layout(pad=4.0)
                self.figures.append(fig)
                self.ax.append(ax)
                self.lines.append(list())
            else:
                fig, ax = plt.subplots()
                self.figures.append(fig)
                self.ax.append(ax)
                line, = self.ax[i].plot([], [], lw=2)
                self.lines.append(line)

        for i in range(last_index):
                line, = self.ax[last_index][i].plot([], [], lw=2)
                self.lines[last_index].append(line)

    def configure_plot(self, ax, title, xlabel, ylabel):
        ax.set_autoscaley_on(True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def get_attributes(self):
        return [Input("input")], [Output("plot")], [ComboboxParameter("function",
                                                                      {"Loss": "loss", "Accuracy": "acc", "All": "all"})]

    def process_inputs(self, inputs, outputs, parameters):
        model_data = inputs["input"].value
        # self.loss.append(model_data["loss"])
        # self.accuracy.append(model_data["accuracy"])
        self.loss.append(model_data[0])
        self.accuracy.append(model_data[1])
        self.batch_num.append(self.counter)

        func = parameters["function"]
        if self.counter % 5 == 0 or func != self.combobox:
            self.combobox = func
            self.choose_param(func)

        self.counter += 1
        outputs["plot"] = Data(self.img)

    def choose_param(self, func):
        if func == "loss":
            self.lines[0].set_color('r')
            self.img = self.live_plotter(self.figures[0], self.lines[0], self.ax[0],
                                         self.batch_num, self.loss)

        elif func == "acc":
            self.lines[1].set_color('g')
            self.img = self.live_plotter(self.figures[1], self.lines[1], self.ax[1],
                                         self.batch_num, self.accuracy)

        elif func == "all":
            self.lines[2][0].set_color('r')
            self.lines[2][1].set_color('g')
            self.img = self.live_plotter(self.figures[2], self.lines[2], self.ax[2],
                                         self.batch_num, [self.loss, self.accuracy])

    def live_plotter(self, figure, lines, ax, x_vec, y_vec):
        if isinstance(lines, list):
            for i in range(len(lines)):
                self.set_ax(lines[i], x_vec, y_vec[i], ax[i])
        else:
            self.set_ax(lines, x_vec, y_vec, ax)
        figure.canvas.draw()
        figure.canvas.flush_events()
        pause.seconds(0.1)
        data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return data

    def set_ax(self, line, x_vec, y_vec, ax):
        line.set_xdata(x_vec)
        line.set_ydata(y_vec)
        ax.relim()
        ax.autoscale_view()


class RealTimeGenerator(InputElement):
    name = "Float generator"
    comment = "Generates endlessly float numbers with 1 second pause"

    def get_attributes(self):
        return [], [Output("output")], []

    def process_inputs(self, inputs, outputs, parameters):
        while True:
            self.set_state(Element.STATE_BUSY)
            self.may_interrupt()
            # model_data = {"loss": random(), "accuracy": random()}
            model_data = [random(), random()]
            self.outputs["output"].put(Data(model_data))
            self.set_state(Element.STATE_READY)
            time.sleep(1)


class TensorboardStarter(NormalElement):
    name = "Tensorboard"
    comment = "Opens web browser with Tensorboard visualisation"

    def get_attributes(self):
        return [Input("model")], [], [ButtonParameter("web", self.open_web, "Open website")]

    def process_inputs(self, inputs, outputs, parameters):
        self.set_state(Element.STATE_BUSY)
        self.may_interrupt()

        model = inputs["model"].value
        subprocess.Popen(['tensorboard', '--logdir', 'logs/fit'])
        time.sleep(3)
        self.set_state(Element.STATE_READY)

    def open_web(self):
        webbrowser.open('http://localhost:6006')


class Training(InputElement):
    name = "Training"
    comment = "Basic training for neural network"

    def get_attributes(self):
        return [], [Output("model")], []

    def process_inputs(self, inputs, outputs, parameters):
        self.set_state(Element.STATE_BUSY)
        self.may_interrupt()

        os.system("rm -rf ./logs/")

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = self.create_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        hist = model.fit(x=x_train,
                         y=y_train,
                         epochs=5,
                         validation_data=(x_test, y_test),
                         callbacks=[tensorboard_callback])

        loss = hist.history['loss']
        accuracy = hist.history['accuracy']
        model_data = {'loss': loss, 'accuracy': accuracy}

        self.outputs["model"].put(Data(model))
        self.set_state(Element.STATE_READY)

    def create_model(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])


register_elements_auto(__name__, locals(), "AI Visualisation", 10)
