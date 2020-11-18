from random import *

import matplotlib.pyplot as plt
import time
import webbrowser
import subprocess

from cvlab.diagram.elements.base import *

plt.style.use('ggplot')


class Diagram:
    def __init__(self, name, dim):
        self.name = name
        self.figure, self.ax = plt.subplots(nrows=dim[0], ncols=dim[1])

    def live_plotter(self, x_vec, y_vec):
        pass

    def set_ax(self, line, x_vec, y_vec, ax):
        line.set_xdata(x_vec)
        line.set_ydata(y_vec)
        ax.relim()
        ax.autoscale_view()

    def draw_figure(self):
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        data = np.fromstring(self.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(self.figure.canvas.get_width_height()[::-1] + (3,))
        return data


class MultiDiagram(Diagram):
    def __init__(self, name, dim):
        super().__init__(name, dim)
        self.figure.tight_layout(pad=4.0)
        self.line = list()
        for a in self.ax:
            tmp, = a.plot([], [], lw=2)
            self.line.append(tmp)

    def live_plotter(self, x_vec, y_vec):
        for i in range(len(self.line)):
            self.set_ax(self.line[i], x_vec, y_vec[i], self.ax[i])
        return self.draw_figure()


class SingleDiagram(Diagram):
    def __init__(self, name, dim):
        super().__init__(name, dim)
        self.line, = self.ax.plot([], [], lw=2)

    def live_plotter(self, x_vec, y_vec):
        self.set_ax(self.line, x_vec, y_vec, self.ax)
        return self.draw_figure()


class Visualisation(NormalElement):
    name = "Live plotting"
    comment = "Generates plot from loss function in real time"

    def __init__(self):
        super(Visualisation, self).__init__()
        self.counter = 0
        self.loss = []
        self.accuracy = []
        self.batch_num = []
        self.img = []
        self.diagrams = []
        self.combo_param = ""
        self.input_param = {}

        self.set_all_diagrams()

    def set_all_diagrams(self):
        # Settings for loss function
        self.diagrams.append(SingleDiagram("Loss", [1, 1]))
        self.configure_plot(self.diagrams[0].ax, 'Loss function', 'Batch', 'Loss')
        self.diagrams[0].line.set_color('r')

        # Settings for accuracy function
        self.diagrams.append(SingleDiagram("Accuracy", [1, 1]))
        self.configure_plot(self.diagrams[1].ax, 'Accuracy function', 'Batch', 'Accuracy')
        self.diagrams[1].line.set_color('g')

        # Settings for diagram with all functions
        self.diagrams.append(MultiDiagram("All", [2, 1]))
        self.configure_plot(self.diagrams[2].ax[0], 'Loss function', 'Batch', 'Loss')
        self.configure_plot(self.diagrams[2].ax[1], 'Accuracy function', 'Batch', 'Accuracy')
        self.diagrams[2].line[0].set_color('r')
        self.diagrams[2].line[1].set_color('g')

    def configure_plot(self, ax, title, xlabel, ylabel):
        ax.set_autoscaley_on(True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def get_attributes(self):
        return [Input("input")], [Output("plot")], [IntParameter("batch_step", value=5, min_=1),
                                                    ComboboxParameter("function", {"Loss": "loss",
                                                                                   "Accuracy": "acc", "All": "all"})]

    def process_inputs(self, inputs, outputs, parameters):
        model_data = inputs["input"].value
        if model_data != self.input_param:
            self.input_param = model_data
            self.loss.append(model_data["loss"])
            self.accuracy.append(model_data["accuracy"])
            self.batch_num.append(self.counter)
            self.counter += 1

        batch_step = parameters["batch_step"]
        func = parameters["function"]
        if self.counter % batch_step == 0 or func != self.combo_param:
            self.combo_param = func
            self.choose_param(func)

        outputs["plot"] = Data(self.img)

    def choose_param(self, func):
        if func == "loss":
            self.img = self.diagrams[0].live_plotter(self.batch_num, self.loss)

        elif func == "acc":
            self.img = self.diagrams[1].live_plotter(self.batch_num, self.accuracy)

        elif func == "all":
            self.img = self.diagrams[2].live_plotter(self.batch_num, [self.loss, self.accuracy])


class RealTimeGenerator(InputElement):
    name = "Float generator"
    comment = "Generates endlessly float numbers with 1 second pause"

    def get_attributes(self):
        return [], [Output("output")], []

    def process_inputs(self, inputs, outputs, parameters):
        while True:
            self.set_state(Element.STATE_BUSY)
            self.may_interrupt()
            model_data = {"loss": random(), "accuracy": random()}
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


register_elements("AI Visualisation", [Visualisation, RealTimeGenerator, TensorboardStarter], 10)
