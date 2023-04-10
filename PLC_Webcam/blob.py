from typing import Tuple, List
from tracking import ElementTracked


class Blob(ElementTracked):
    def __init__(self):
        ElementTracked.__init__(self)
        self.in_gate: bool = False
        self.in_roi: bool = False
        self.bottom: int = 0
        self.top: int = 0
        self.left: int = 0
        self.right: int = 0
        self.zones: List = []
