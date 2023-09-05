"""
Usage:
    kmonitor [options] <chain.h5>

Options:
    -f --follow  Plot live data
    --burn <n>  Number of burned-in steps
"""

from docopt import docopt
import numpy as np
from pathlib import Path
import emcee
from rich.console import Console
console = Console()


def monitor():
    args = docopt(__doc__)
    filename = args['<chain.h5>']
    filepath = Path(filename)
    if not filepath.exists():
        print("File not found!")
        return
    reader = emcee.backends.HDFBackend(filename)
    console.print(f"Steps: {reader.iteration}")
    console.print(f"Acceptance Fraction: {np.mean(reader.accepted / float(reader.iteration)):%}")
    console.print(f"Autocorrelation Time: {np.mean(reader.get_autocorr_time(tol=0))}")
