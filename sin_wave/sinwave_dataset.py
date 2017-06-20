""" sin wave dataset """
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class SinDataset:
  def __init__(self, num_cycles=4, num_periods=1000,
               batch_size=10, dtype=np.float32):
    self._batch_size = batch_size
    self._num_cycles = num_cycles
    self._num_periods = num_periods
    self._dtype = dtype
    self.all_data = np.empty(0)
    self.train_data = {}
    self.test_data = {}
    self._init_data()

  def _init_data(self):
    """ """
    # build all_data
    self.all_data = pd.DataFrame(np.arange(
        self._num_cycles * self._num_periods + 1), columns=["t"])
    self.all_data["sin_t"] = self.all_data.t.apply(
        lambda x: math.sin(x * (2 * math.pi / self._num_periods)))
    self.all_data["sin_t+1"] = self.all_data["sin_t"].shift(-1)
    # build train_data
    datasize = len(self.all_data.index) - 1
    train_size = int(datasize / 2)
    self.train_data["input"] = np.array(
        [self.all_data["sin_t"][v] for v in range(train_size)],
        dtype=self._dtype)
    self.train_data["output"] = np.array(
        [self.all_data["sin_t+1"][v] for v in range(train_size)],
        dtype=self._dtype)
    # build test_data
    self.test_data["input"] = np.array(
        [self.all_data["sin_t"][v] for v in range(train_size, datasize)],
        dtype=self._dtype).reshape(-1, 1)
    self.test_data["output"] = np.array(
        [self.all_data["sin_t+1"][v] for v in range(train_size, datasize)],
        dtype=self._dtype).reshape(-1, 1)

  def fetch_train(self):
    train = {}
    trainsize = len(self.train_data["input"])
    batch_mask = np.random.choice(trainsize, self._batch_size)
    train["input"] = np.array(
        [self.train_data["input"][v] for v in batch_mask]).reshape(-1, 1)
    train["output"] = np.array(
        [self.train_data["output"][v] for v in batch_mask]).reshape(-1, 1)
    return train["input"], train["output"]

  def fetch_test(self):
    return self.test_data["input"], self.test_data["output"]
