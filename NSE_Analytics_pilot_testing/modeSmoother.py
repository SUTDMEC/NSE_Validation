# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:01:12 2015

@author: I309943
"""
import numpy as np
from util import chunks, great_circle_dist
from predict_mode import AbstractPredictor, getStartEndIdx
from predict_mode import MODE_STOP_OUT, MODE_STOP_IN, MODE_WALK_OUT, MODE_WALK_IN, MODE_TRAIN, MODE_BUS, MODE_CAR, MODE_TBD, MODE_TBD_VC

def aveVelCalc(v_pt, delta_t):
    #  function used to calculate the average of velocity over a segment
    #  able to deal with NaN points and consider the delta time
    
    dist_sum = np.nansum(v_pt*delta_t)
    if dist_sum==0:
        return np.nan
    else:
        return dist_sum/np.nansum(delta_t)

def notWalkSegRefine(timestamps,hw_mode,v_pt_seg,delta_t_seg,travel_dist,jump_dist,NOT_STOP_V,IS_STOP_V,indoor_seg,TIME_SET_STOPPED,VC_MIN_DIST=100):
    #    function to refine the hw modes of non walking segment based on the percentage of each mode's time
    
    #    definition of mode code
#    MODE_WALK_IN = 3
#    MODE_WALK_OUT = 2
#    MODE_STOP_IN = 1
#    MODE_STOP_OUT = 0
#    MODE_CAR = 6
#    MODE_TRAIN = 4
#    MODE_TBD_VC = 11
    
    num_pt = len(hw_mode);
    
    # check the total time span, if too small, set as stopped
    if np.sum(delta_t_seg)<TIME_SET_STOPPED or travel_dist<3*VC_MIN_DIST or jump_dist<2*VC_MIN_DIST:
        time_stop_in = np.sum(delta_t_seg[np.where(hw_mode==MODE_STOP_IN)[0]])
        time_stop_out = np.sum(delta_t_seg[np.where(hw_mode==MODE_STOP_OUT)[0]])
        if time_stop_in!=0 or time_stop_out!=0:
            max_id = np.argmax(np.array([time_stop_in,time_stop_out,0,0]))+1
        elif indoor_seg: # if no "stopped" mode at all, use the previous IO
            max_id = 1
        else:
            max_id = 2
    else:
        
        # get out each non-walking mode and calculate the time percentage
        mode_type = np.unique(hw_mode)
        if np.any((mode_type==MODE_WALK_IN)| (mode_type==MODE_WALK_OUT)):
            # check if there's walking modes mixed in
            raise Exception("There are walking points mixed in this none walking segment!")
        
        
        time_stop_in = np.sum(delta_t_seg[np.where(hw_mode==MODE_STOP_IN)[0]])
        time_stop_out = np.sum(delta_t_seg[np.where(hw_mode==MODE_STOP_OUT)[0]])
        time_car = np.sum(delta_t_seg[np.where(hw_mode==MODE_CAR)[0]])
        time_train = np.sum(delta_t_seg[np.where(hw_mode==MODE_TRAIN)[0]]) 
        time_TBD_VC = np.sum(delta_t_seg[np.where(hw_mode==MODE_TBD_VC)[0]]) 
        time_stop_total = time_stop_in+time_stop_out
        time_vehicle_total = time_car+time_train+time_TBD_VC
        
        v_mean_seg = aveVelCalc(v_pt_seg, delta_t_seg)
        # check the mean velocity of this segment first
        max_id = 0
        
        if v_mean_seg<IS_STOP_V:
            # if mean v is smaller than a threshold, then it's stopped
            if time_stop_in!=0 or time_stop_out!=0:
                max_id = np.argmax(np.array([time_stop_in,time_stop_out,0,0]))+1
            elif indoor_seg: # if no "stopped" mode at all, use the previous IO
                max_id = 1
            else:
                max_id = 2
        elif v_mean_seg>NOT_STOP_V or time_vehicle_total>time_stop_total:
            # if mean v is larger than a threshold or total vehicle time is larger than 
            # the total stop time, then it is not stopped, must be vehicle
            if time_car !=0 or time_train!=0:
                max_id = np.argmax(np.array([0,0,time_car,time_train]))+1
            else:
                max_id = 3 # set car as default        
        else:
            max_id = np.argmax(np.array([time_stop_in,time_stop_out,time_car,time_train]))+1


    hw_mode_refined = np.array([])
    # take the mode with maximum time portion
    if max_id == 1:
        hw_mode_refined = np.array([1]*num_pt)*MODE_STOP_IN;
    elif max_id == 2:
            hw_mode_refined = np.array([1]*num_pt)*MODE_STOP_OUT;
    elif max_id == 3:
            hw_mode_refined = np.array([1]*num_pt)*MODE_CAR;
    elif max_id == 4:
            hw_mode_refined = np.array([1]*num_pt)*MODE_TRAIN
    else:
        hw_mode_refined = hw_mode
        raise Exception("Invalid max_id was given, no refinement was achieved!")
    #print hw_mode,hw_mode_refined
    return hw_mode_refined
    
def notWalkSegProcess(hw_mode_refined, ave_vel, delta_t, timestamp, lat, lon, dist, num_pt_total, NOT_STOP_V, IS_STOP_V, TIME_SET_STOPPED, VC_MIN_DIST):
    # function used to pick up all the none walking segments in the given mode vector
    # and call notWalkSegRefine() function to refine those modes
    
    # input: whole modes of the device
    # output: refined modes of the device
    
    # get the start and end idx of non walking segment
    idx_nonwalking = np.where((hw_mode_refined!=MODE_WALK_IN) & (hw_mode_refined!=MODE_WALK_OUT))[0]
    if len(idx_nonwalking)==0:
        num_nonwalking_seg = 0
    else:
        start_idx_nonwalking,end_idx_nonwalking,num_nonwalking_seg = getStartEndIdx(idx_nonwalking)
    
    # go through each not walking segment
    #print "nonwalking_seg:",num_nonwalking_seg
    for i_seg in range(0,num_nonwalking_seg):
        #print i_seg,start_idx_nonwalking[i_seg],end_idx_nonwalking[i_seg]
        timestamp_seg = timestamp[start_idx_nonwalking[i_seg]:end_idx_nonwalking[i_seg]+1]
        
        # check whether it's indoor or outdoor
        if start_idx_nonwalking[i_seg]>0: # set indoor/outdoor as previous state
            if hw_mode_refined[start_idx_nonwalking[i_seg]-1]==MODE_WALK_IN:
                indoor_seg = 1
            elif hw_mode_refined[start_idx_nonwalking[i_seg]-1]==MODE_WALK_OUT:
                indoor_seg = 0
            else:
                indoor_seg = 1 # assign default value
                raise Exception("The segment before this none walking segment is not walking segment!")

        elif end_idx_nonwalking[i_seg]<num_pt_total-1: # set indoor/outdoor as next state
            if hw_mode_refined[end_idx_nonwalking[i_seg]+1]==MODE_WALK_IN:
                indoor_seg = 1
            elif hw_mode_refined[end_idx_nonwalking[i_seg]+1]==MODE_WALK_OUT:
                indoor_seg = 0
            else:
                indoor_seg = 1 # assign default value
                raise Exception("The segment before this none walking segment is not walking segment!")
        else:
            indoor_seg = 1 # assign default value
        
        # refine the modes of none-walking period
        hw_mode_refined_seg = hw_mode_refined[start_idx_nonwalking[i_seg]:end_idx_nonwalking[i_seg]+1]
        v_pt_filtered_seg = ave_vel[start_idx_nonwalking[i_seg]: end_idx_nonwalking[i_seg]+1]
        delta_t_seg = delta_t[start_idx_nonwalking[i_seg]: end_idx_nonwalking[i_seg]+1]
        travel_dist = np.nansum(dist[start_idx_nonwalking[i_seg]: end_idx_nonwalking[i_seg]+1])
        jump_dist = great_circle_dist([lat[start_idx_nonwalking[i_seg]],lon[start_idx_nonwalking[i_seg]]],[lat[end_idx_nonwalking[i_seg]],lon[end_idx_nonwalking[i_seg]]],'meters')

        # call the function to refind the non-walking modes
        hw_mode_refined_seg = notWalkSegRefine(timestamp_seg,hw_mode_refined_seg,v_pt_filtered_seg,delta_t_seg,travel_dist,jump_dist,NOT_STOP_V,IS_STOP_V,indoor_seg,TIME_SET_STOPPED,VC_MIN_DIST)
        # update the modes into the entire trip
        hw_mode_refined[start_idx_nonwalking[i_seg]:end_idx_nonwalking[i_seg]+1] = hw_mode_refined_seg 

    return hw_mode_refined

def modeSmooth(hw_mode,timestamp,delta_t,lat,lon,vel,ave_vel,delta_steps,dist):
#     this function smooths the hw_mode code
#     Output: s_hw_mode : smoothed hw_code
#     Input: 
#        - hw_mode: a vector of hw_code
#        - timestamp: a vector of timestamp
#        - lat,lon: vectors of lat and lon representing location
#        - vel: vector of geographical velocity, m/s
#        - ave_vel: 5-window moving average of geographical velocity, m/s
    
    NUM_AFT_WALKING = 3 # num of points after walking segment to be set as invalid hw mode
    TIME_NOT_HIDE = 60*5 # sec, time longer than which the several points after each walking segment won't be set as TBD
    TIME_SET_STOPPED = 60*1 # sec, time shorter than which the not walking seg is set as stopped
    NOT_STOP_V = 5.0 # m/s, mean velocity above which the not walking seg is considered as not stopped
    IS_STOP_V = 1.0 # m/s, mean velocity below which the not walking seg is considered as stopped
    WALK_MAX_V_AVE = 7.0  # m/s, moving average velocity above which it's considered as TBD
    WALK_MAX_V_PT = 7.0  # m/s, single point velocity above which it's considered as TBD
    SINGLE_WALK_MAX_V = 1.5 # m/s, single point velocity above which it's considered as TBD for single walking point
    SLEEPING_TIME = 60  # s, time larger than which the mode is check and reset
    SLEEP_MAX_V = 2.0 # m/s, vel_ave or vel above which the sleeping point will be assigned as TBD_VC
    SLEEP_TO_WALK_STEPS = 1 # steps to time ratio below which the sleeping point will be assigned as stopped
    TBD_VC_TIME = 200 # s, time above which the point will be considered as vehicle mode
    TBD_VC_DIST = 300 # m, distance above which the point will be considered as vehicle mode
    SHORT_WALK = 200 # s, time below which the walking segment between two vehicle seg will be considered as invalid
    FEW_STEPS = 50 # steps below which the walking segment between two vehicle seg will be considered as invalid
    SHORT_WALK_MAX_V = 1.5    # m/s, single point velocity above which it's considered as TBD for single walking point
    VC_MIN_DIST = 100 #m, distance smaller than which the vehicle mode segment is needed to reprocess    
    WALK_IN_MAX_DIST = 150  #m, jump distance larger than which the walking mode segment is considered as outdoor
    
    #mode representation

#    MODE_WALK_IN = 3
#    MODE_WALK_OUT = 2
#    MODE_STOP_OUT = 0
#    MODE_STOP_IN = 1
#    MODE_TBD = 10
#    MODE_TBD_VC = 11

    # initialization
    num_pt_total = len(hw_mode) # total number of points in this trip
    hw_mode_refined = hw_mode.copy() # initialize the refined mode vector
    
    
    # check the long delta timestamp points
    # assign points with long delta timestamp but low velocity as "stopped indoor"
    idx_sleep = np.where(delta_t>SLEEPING_TIME)[0].tolist()
    prev_i_sp = 0
    for i_sp in idx_sleep:
        if ave_vel[i_sp] > SLEEP_MAX_V or vel[i_sp] > SLEEP_MAX_V:
            hw_mode_refined[i_sp] = MODE_TBD_VC
            if i_sp<num_pt_total-1:
                hw_mode_refined[i_sp+1] = hw_mode_refined[i_sp]
        elif delta_t[i_sp]>TBD_VC_TIME and delta_t[i_sp]*vel[i_sp]>TBD_VC_DIST:
            hw_mode_refined[i_sp] = MODE_TBD_VC
            if i_sp<num_pt_total-1:
                hw_mode_refined[i_sp+1] = hw_mode_refined[i_sp]
        elif (delta_steps[i_sp]/delta_t[i_sp]) < SLEEP_TO_WALK_STEPS:
            if hw_mode_refined[i_sp]==MODE_WALK_OUT:
                hw_mode_refined[i_sp] = MODE_STOP_OUT
                if delta_t[i_sp]>500:
                    if i_sp>0 and i_sp-1!=prev_i_sp:
                        hw_mode_refined[i_sp-1] = MODE_WALK_OUT
                        vel[i_sp-1] = 0
                        ave_vel[i_sp-1] = 0
                    if i_sp<num_pt_total-1:
                        if delta_t[i_sp+1]<SLEEPING_TIME:
                            hw_mode_refined[i_sp+1] = MODE_WALK_OUT
                            vel[i_sp+1] = 0
                            ave_vel[i_sp+1] = 0
            else:
                hw_mode_refined[i_sp] = MODE_STOP_IN
                if delta_t[i_sp]>500:
                    if i_sp>0 and i_sp-1!=prev_i_sp:
                        hw_mode_refined[i_sp-1] = MODE_WALK_IN
                        vel[i_sp-1] = 0
                        ave_vel[i_sp-1] = 0
                    if i_sp<num_pt_total-1:
                        if delta_t[i_sp+1]<SLEEPING_TIME:
                            hw_mode_refined[i_sp+1] = MODE_WALK_IN
                            vel[i_sp+1] = 0
                            ave_vel[i_sp+1] = 0
                        
        prev_i_sp = i_sp
    
    # refine the walking mode points by checking the moving average of velocity
    idx_walking = np.where((hw_mode_refined == MODE_WALK_IN) | (hw_mode_refined == MODE_WALK_OUT))[0].tolist()
    for i_walk in idx_walking:
        if ave_vel[i_walk] > WALK_MAX_V_AVE or vel[i_walk] > WALK_MAX_V_PT:
            hw_mode_refined[i_walk] = MODE_TBD_VC
            
    idx_walking = np.where((hw_mode_refined == MODE_WALK_IN) | (hw_mode_refined == MODE_WALK_OUT))[0].tolist()
    
#     get the start and end idx of each walking segment
    if(len(idx_walking)==0):
        num_walking_seg = 0
    else:
        start_idx_walking,end_idx_walking,num_walking_seg = getStartEndIdx(idx_walking)
        # check the single walking point, if vel>3m/s, set as TBD_VC
        idx_single_walking = list(set(start_idx_walking).intersection(end_idx_walking))
        for i_sw in idx_single_walking:
            if ave_vel[i_sw] > SINGLE_WALK_MAX_V or vel[i_sw] > SINGLE_WALK_MAX_V:
                hw_mode_refined[i_sw] = MODE_TBD_VC 
                start_idx_walking.remove(i_sw)
                end_idx_walking.remove(i_sw)
                num_walking_seg = num_walking_seg-1
                
        # go through each walking segment and change indoor to outdoor if dist larger than a threshold
        for i_walk_seg in xrange(0,num_walking_seg):
            jump_dist = great_circle_dist([lat[start_idx_walking[i_walk_seg]],lon[start_idx_walking[i_walk_seg]]],[lat[end_idx_walking[i_walk_seg]],lon[end_idx_walking[i_walk_seg]]],'meters')
            if jump_dist>WALK_IN_MAX_DIST:
                walk_seg_length = end_idx_walking[i_walk_seg]+1-start_idx_walking[i_walk_seg]
                hw_mode_refined[start_idx_walking[i_walk_seg]:end_idx_walking[i_walk_seg]+1] = np.ones(walk_seg_length)*MODE_WALK_OUT
        
#    # go through each walking segment
#    # modify modes of the several pts before and after the walking segment
#    # updated in hw_mode_trip_refined and idx_walking_trip
#    #print "walking_seg:",num_walking_seg
#    for i_seg in xrange(0,num_walking_seg):
#        if(i_seg<num_walking_seg-1):
#            start_next_seg = start_idx_walking[i_seg+1]
#        else:
#            start_next_seg = num_pt_total
#        
#        #print start_idx_walking[i_seg],end_idx_walking[i_seg]
#
#        if (end_idx_walking[i_seg]+NUM_AFT_WALKING < start_next_seg) and (timestamp[end_idx_walking[i_seg]+NUM_AFT_WALKING]-timestamp[end_idx_walking[i_seg]] < TIME_NOT_HIDE):
#            # make several pts after walking seg as MODE_TBD            
#             hw_mode_refined[end_idx_walking[i_seg]+1:end_idx_walking[i_seg]+NUM_AFT_WALKING+1] = MODE_TBD 
#        elif(timestamp[start_next_seg-1]-timestamp[end_idx_walking[i_seg]] < TIME_NOT_HIDE):
#            hw_mode_refined[end_idx_walking[i_seg]+1:start_next_seg] = MODE_TBD 

    hw_mode_refined = notWalkSegProcess(hw_mode_refined, ave_vel, delta_t, timestamp, lat, lon, dist, num_pt_total, NOT_STOP_V, IS_STOP_V, TIME_SET_STOPPED, VC_MIN_DIST)    
    
    # try to combine mode segments like: vehicle + stop/walk + vehicle
    temp_modes = np.array(hw_mode_refined.copy())
    temp_modes[(temp_modes==MODE_STOP_IN) | (temp_modes==MODE_STOP_OUT) | (temp_modes==MODE_WALK_OUT)]=MODE_WALK_IN
    mode_segs = list(chunks(temp_modes,True)) # take the mode chunk
    num_mode_segs = len(mode_segs)
    
    # go through each mode chunk
    for i_seg in xrange(1,num_mode_segs-1):
        mode_seg = mode_segs[i_seg]
        mode_seg_prev = mode_segs[i_seg-1]
        mode_seg_aft = mode_segs[i_seg+1]
        # check the steps and average velocity of walking seg between two vehicle seg
        if mode_seg[2]==MODE_WALK_IN:
            if mode_seg_prev[2]!=MODE_WALK_IN and mode_seg_aft[2]!=MODE_WALK_IN:
                time_span = np.sum(delta_t[mode_seg[0]:mode_seg[1]])
                tot_steps = np.nansum(delta_steps[mode_seg[0]:mode_seg[1]])
                v_mean_mode_seg = aveVelCalc(ave_vel[mode_seg[0]:mode_seg[1]], delta_t[mode_seg[0]:mode_seg[1]])
                if time_span<SHORT_WALK and (tot_steps<FEW_STEPS or v_mean_mode_seg>SHORT_WALK_MAX_V):
                    hw_mode_refined[mode_seg[0]:mode_seg[1]] = MODE_TBD_VC
            
                    
    hw_mode_refined = notWalkSegProcess(hw_mode_refined, ave_vel, delta_t, timestamp, lat, lon, dist, num_pt_total, NOT_STOP_V, IS_STOP_V, TIME_SET_STOPPED, VC_MIN_DIST)
    
    return hw_mode_refined


class SmoothingPredictor(AbstractPredictor):
    """Heuristic predictor that smoothes the travel mode."""
    
    def __init__(self):
        pass
        
    def fit(self, data, target):
        pass

    def predict(self, data, modes):
        """predict travel mode. Input is the pandas data frame of measurements
        and an array of current mode predictions.  Returns an array of
        predicted modes of the same size as the input data frame has
        rows.

        """
        timestamp = data['TIMESTAMP'].values
        lat = data['WLATITUDE'].values
        lon = data['WLONGITUDE'].values
        vel = data['VELOCITY'].values
        dist = data['DISTANCE_DELTA'].values
        ave_vel = data['AVE_VELOCITY'].values
        delta_t = data['TIME_DELTA'].values
        delta_steps = data['STEPS_DELTA'].values

        refined_modes = modeSmooth(modes, timestamp, delta_t, lat, lon, vel, ave_vel, delta_steps, dist)
        return refined_modes
