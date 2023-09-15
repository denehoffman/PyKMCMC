"""Usage: klive <chains.h5> [-n <ndims>]

Options:
    -n <ndims>  Number of dimensions to plot
"""

import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class ChainHandler(FileSystemEventHandler):
    def __init__(self, ndim, path):
        self.ndim = ndim
        self.path = path
        self.samples = None  # Initialize samples
        with h5py.File(path, "r") as hf:
            self.samples = np.copy(hf["samples"])
        self.fig, self.axes = plt.subplots(self.ndim, 1, figsize=(16, 4 * self.ndim))
        plt.ion()  # Enable interactive mode
        super().__init__()

    def on_modified(self, event):
        read = False
        while not read:
            try:
                print(f"{event.src_path} has changed!")
                with h5py.File(event.src_path, "r") as hf:
                    self.samples = np.copy(hf["samples"])
                    read = True
            except Exception as e:
                print(e)

    def update_plot(self):
        if self.samples is not None:
            for n in range(self.ndim):
                self.axes[n].clear()  # Clear the existing subplot
                self.axes[n].plot(self.samples[:, :, n], alpha=0.2, color="k", lw=0.5)
            plt.tight_layout()
            plt.pause(1)


def watch():
    args = docopt(__doc__)
    ch = ChainHandler(int(args["-n"]), args["<chains.h5>"])
    obs = Observer()
    obs.schedule(ch, path=args["<chains.h5>"], recursive=False)
    obs.start()
    try:
        while True:
            ch.update_plot()  # Update the plot periodically
            plt.show(block=False)
            time.sleep(10)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()


if __name__ == "__main__":
    watch()
