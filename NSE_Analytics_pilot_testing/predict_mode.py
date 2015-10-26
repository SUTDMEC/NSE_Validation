"""Module to predict the user's state on the server side.

"""


from abc import ABCMeta, abstractmethod
import numpy as np


# mode representation
MODE_STOP_OUT = 0
MODE_STOP_IN = 1
MODE_WALK_OUT = 2
MODE_WALK_IN = 3
MODE_TRAIN = 4
MODE_BUS = 5
MODE_CAR = 6
MODE_TBD = 10
MODE_TBD_VC = 11


class AbstractPredictor(object):
    """Abstract base predictor class for trip predictors."""
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def fit(self, data, targets):
        pass

    @abstractmethod
    def predict(self, data, mode):
        pass



class DoNothingPredictor(AbstractPredictor):
    """Dummy predictor that simply returns the hardware mode."""

    def fit(self, data, target):
        pass

    def predict(self, data, mode):
        return data['MODE'].values



def getStartEndIdx(raw_idx):
    # this function is used to calculate the starting and end indices of a set of consecutive indices
    #   input: raw_idx: a vector of indices to separate
    #   output: start_idx: vector of all the starting indices
    #           end_idx: vector of all the end indices

    diff_idx = np.array([0])
    diff_idx = np.append(diff_idx, np.diff(np.array(raw_idx)))
    idx_idx_jump = np.where(diff_idx>1)[0] # indices of raw_idx where a index jump occurs
    num_seg = len(idx_idx_jump)+1;
    start_idx = [raw_idx[0]]
    end_idx = []
    
    for i_seg in xrange(0,num_seg-1):
        end_idx.append(raw_idx[idx_idx_jump[i_seg]-1])
        start_idx.append(raw_idx[idx_idx_jump[i_seg]])

    end_idx.append(raw_idx[len(raw_idx)-1])

    return start_idx, end_idx, num_seg
