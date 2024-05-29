from enum import Enum
from dt.task import *


class Stage(Enum):
    MODEL = 1
    SLICELINE = 2,
    DT = 3

    def next(self):
        if self == Stage.MODEL:
            return Stage.SLICELINE
        elif self == Stage.SLICELINE:
            return Stage.DT
        elif self == Stage.DT:
            return Stage.MODEL
        else:
            return ValueError


# TaskData stores information required to maintain the dashboard.
class DashData:
    def __init__(self, task: AbstractTask):
        self.iter = 0
        self.stage = Stage.MODEL
        self.train_agg_losses = []
        self.test_agg_losses = []
        self.train_losses = []
        self.test_losses = []
        self.slice_train_losses = []
        self.slice_test_losses = []
        self.slices = None
        self.sliceline_stats = None
        self.counts = None
        self.sources_stats = None
        self.dt_result = None
        self.train = task.initial_train
        self.test = task.test
        self.task = task
