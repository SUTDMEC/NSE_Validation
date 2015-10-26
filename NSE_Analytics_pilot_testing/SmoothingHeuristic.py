

class HeuristicPredictor(AbstractPredictor):
    """Heuristic-based predictor that 'smoothes' the hardware mode based
    on Yuren's heuristics."""

    def __init__(self):
        """Heuristic parameters"""
        self.NUM_AFT_WALKING = 3 # num of points after walking segment to be set as invalid hw mode
        self.TIME_NOT_HIDE = 60*5 # sec, time longer than which the above two are not applicable
        self.TIME_SET_STOPPED = 60*2 # sec, time shorter than which the not walking seg is set as stopped
        self.NOT_STOP_V = 10 # km/h, mean velocity above which the not walking seg is considered as nbot stopped
        self.IS_STOP_V = 5 # km/h, mean velocity below which the not walking seg is considered as stopped
        self.V_MAX = 150 # km/h, point velocity above which it's considered as invalid, NaN
    
    def fit(self, data, target):
        pass


    def consecutive_modes(self, modes):
        """Generator function to identify sequences of consecutive
        modes. Input is the list of modes. Yields a tuples of the
        start and end index of a consecutive mode.

        """
        start_index = 0
        last_mode = modes[0]
        for index, mode in enumerate(modes):
            if mode != last_mode:
                yield(start_index, index)
                start_index = index
                last_mode = mode
        yield (start_index, len(modes))


    def predict(self, data):
        """predict the heuristically smoothed travel mode. data is a pandas
        data frame withe the hardware measurements for hardware mode,
        timestamps and lat/lon locations. Return a list of refined modes.

        """
        ## initialization
        # total number of points in this trip
        num_pt_total = len(data) 
        # initialize the refined mode as a copy of the hardware mode
        hw_mode_refined = data[['MODE']]
        
        return hw_mode_refined[['MODE']].values[:,0]
    
        

