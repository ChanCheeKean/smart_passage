import numpy as np
from utils.sort import Sort
deep_tracker = Sort()

inp = np.array([[22.0,22.0,50.0,50.0,0.5]])
deep_tracker.update(np.empty((0, 5)))
deep_tracker.update(inp)

