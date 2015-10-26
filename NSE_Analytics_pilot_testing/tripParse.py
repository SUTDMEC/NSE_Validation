"""Module to identify home and school locations and trips between
them.

"""


import datetime
import numpy as np
import pandas as pd
import math
from util import great_circle_dist
from util import chunks
from functools import partial
import logging
from collections import Counter, defaultdict
from itertools import izip
from util import chunks

def get_hour_SGT(timestamp):
    """Get the hour of the day (0-23) in Singapore UTC+8 timezone from a
    unix timestamp. timestamp is a UNIX timestamp in UTC, return value is
    integer between 0 and 23 that represents the hour of the day in SGT
    time for that time stamp."""
    # add 8h (28800 sec) for SGT time
    temp_t = int(timestamp)+28800
    # mod 24h (86400 sec) and integer divide by hours (3600 sec)
    hour = (temp_t%86400)/3600
    return hour


def process(predicted_modes, data_frame, stopped_thresh=0.1,
            poi_dwell_time=600, school_start=9, school_end=13,
            home_start=22, home_end=5,
            max_school_thresh=100,
            home_school_round_decimals=4, mode_thresh=100,
            poi_cover_range=30):
    """Identify trips from the provided modes and device
    data. predicted_modes is a list of integers corresponding to the
    travel mode. data_frame is a pandas data frame of the same length,
    containing the sensor measurements. Return a list of trips, each
    trip consists of a start time, end time, start lat/lon, end
    lat/lon, distance, travel mode.

    predicted_modes is the vector of modes after smoothing and bus/car ID
    data_frame is the data frame of all data
    stopped_thresh is the speed in m/s below which we consider the
    user to be non-moving. Default 0.1 m/s (= 0.4km/h)
    poi_dwell_time is the time in seconds above which a stopped
    location is considered a point of interest. Default 900 sec (=
    15min)
    school_start is the hour of the day when school starts. Default 9am.
    school_end is the hour of the day when school end. Default 1pm.
    home_start is the first hour of the day when students are assumed to be
    home at night. Default 10pm.
    home_end is the last hour of the day when students are assumed to be
    home at night. Default 5am.
    max_school_thresh is the threshold for school/home distances smaller than
    which the home/school poi is rejected - to negate the possibility
    of creating home/school links for sensors left at school by mistake
    round_decimals is the number of decimals used in the max_freq
    heuristic for rounding lat / lon values before taking the most
    frequent value
    time_offset is the offset in hours to subtract from each timestamp. Default 8.


    """
    def dwell_time(stop):
        """Return time spend in a stopped location in seconds"""
        start_time, end_time = stop[3], stop[4]
        return (end_time - start_time)

    def getAmPm(t):
        h = get_hour_SGT(t)
        if h < 12:
            return 'AM'
        else:
            return 'PM'

    def getFirstSecondOfDay(t):
        f = t - t % 86400 - 28800
        if t - f > 86400:
            f = f + 86400
        return f

    def getMostFreqMode(modes,ret_num):
        #returns the most common modes
        c = Counter(modes).most_common(ret_num)
        return c #c[0][0]

    def getCO2(modes, dist_trav):
        #returns the CO2 for the trip. values are in g/km, so return gram of CO2
        # assumes distance in km
        #mode_to_emissions= {'car': 187, 'bus': 19, 'train': 13, 'walk': 0}
        mode_to_emissions= {6: 187.0, 5: 19.0, 4: 13.0, 3: 0.0}
        #car = 6, bus = 5, train = 4, all others 0 CO2
        co2 = sum( (mode_to_emissions[mode] * dist for mode, dist in izip(modes, dist_trav) if dist > 0 ) )
        return co2

    def checkDist(dist_in,dist_lim):
        if dist_in > dist_lim:
            dist_in=dist_lim
        if dist_in< 0:
            dist_in=0
        return dist_in

    def checkTime(time_in,time_lim):
        if time_in> time_lim:
            time_in=time_lim
        if time_in<0:
            time_in=0
        return time_in

    def checkCO2(co2_in,co2_lim):
        if co2_in> co2_lim:
            co2_in=co2_lim
        if co2_in<0:
            co2_in=0
        return co2_in

    def pointDist(row, lat,lon):
        vlat = row['WLATITUDE']
        vlon = row['WLONGITUDE']
        return great_circle_dist([vlat,vlon],[lat,lon])

    def distanceFcn(df,lat,lon,tripStart=True,homeIdx=True):
        #finds the index and value of the min distance from a dataframe to a certain point
    
        search_range_school = 50; # m, the radios of searching for the start/end point of a trip (school)
        search_range_home = 0; # m, the radios of searching for the start/end point of a trip (home)

        latlon = df[['WLATITUDE','WLONGITUDE']].values.tolist()
        dist_list = map(lambda x: great_circle_dist(x, [lat,lon], unit="meters"), latlon)
        dist_array = np.array(dist_list)
        if homeIdx:
            # if the location is home, search range is smaller
            idx_in_range = np.where(dist_array<=search_range_home)[0]
            if len(idx_in_range)==0:
                dist = np.nanmin(dist_array)
                idx = dist_list.index(dist)
                if tripStart:
                    dist_col = np.array(dist_list)
                    idx = max(np.where(dist_col== dist)[0])
            else:
                if tripStart:
                    idx = idx_in_range[len(idx_in_range)-1]
                    dist = dist_array[idx]
                else:
                    idx = idx_in_range[0]
                    dist = dist_array[idx]    
        else:
            # if the location is school, search range is larger
            idx_in_range = np.where(dist_array<=search_range_school)[0]
            if len(idx_in_range)==0:
                dist = np.nanmin(dist_array)
                idx = dist_list.index(dist)
                if tripStart:
                    dist_col = np.array(dist_list)
                    idx = max(np.where(dist_col== dist)[0])
            else:
                if tripStart:
                    idx = idx_in_range[len(idx_in_range)-1]
                    dist = dist_array[idx]
                else:
                    idx = idx_in_range[0]
                    dist = dist_array[idx]


        return {'dist':dist,'idx': idx}

    def segFind(df, trip_return, mode_thresh=120, isAM = True, dist_lim = 45.3, max_mode=6, max_walk=4.0):
        #    definition of mode code
        MODE_WALK_IN = 3;
        MODE_WALK_OUT = 2;
        MODE_STOP_IN = 1;
        MODE_STOP_OUT = 0;
        MODE_CAR = 6;
        MODE_TRAIN = 4;
        
        # thresholds for calculating distance of mode segs
        ALL_WALK_TIME=15*60   # time shorter than which the distance of walk mode segment is calculated using all points
        real_to_jump_dist = 2;  # for short walking seg, limit the distance by 2 times the jump distance

        pred_modes = df[['PRED_MODE']].values[:,0] # take out the predicted modes
        # change all STOP_IN, STOP_OUT, WALK_OUT to WALK_IN
        pred_modes[(pred_modes==MODE_STOP_IN) | (pred_modes==MODE_STOP_OUT) | (pred_modes==MODE_WALK_OUT)]=MODE_WALK_IN
        mode_segs = list(chunks(pred_modes,True)) # take the mode chunks
        num_mode_segs = len(mode_segs)
        num_valid_mode_seg = 0
        logging.debug("Mode Segs: " + str(mode_segs))
        time_span = []
        valid_mode_segs = []
        prev_mode = 0

        # go through each mode chunk
        for mode_seg in mode_segs:
            time_span = np.sum(df['TIME_DELTA'].values[mode_seg[0]:mode_seg[1]])

            # abandon if the total segment time is less than threshold, and shorten the list down to 5 mode segments at most
            if time_span < mode_thresh or num_valid_mode_seg > max_mode-1:
                continue
            else:
                latlon_start = [df['WLATITUDE'].values[mode_seg[0]],df['WLONGITUDE'].values[mode_seg[0]]]
                latlon_end = [df['WLATITUDE'].values[mode_seg[1]-1],df['WLONGITUDE'].values[mode_seg[1]-1]]
                jump_dist = great_circle_dist(latlon_start,latlon_end,'meters')
                num_valid_mode_seg += 1
                if isAM:
                    mode_key = 'am_mode'
                    dist_key = 'am_distance'
                else:
                    mode_key = 'pm_mode'
                    dist_key = 'pm_distance'
                
                # calculate the distance of this mode segment
                if int(mode_seg[2]) == MODE_WALK_IN:
                    modes_cur_seg = df['PRED_MODE'].values[mode_seg[0]:mode_seg[1]]
                    dist_cur_seg = df['DISTANCE_DELTA'].values[mode_seg[0]:mode_seg[1]]
                    if time_span<ALL_WALK_TIME:
                        # if the time span is too small, consider all 0-3 modes as walking
                        dist_seg = np.nansum(dist_cur_seg)
                        dist_seg=checkDist(dist_seg,jump_dist*real_to_jump_dist)
                    else:
                        # else if the time span is not too small, only consider 2 and 3 modes
                        dist_seg = np.nansum(dist_cur_seg[np.where((modes_cur_seg==MODE_WALK_IN) | (modes_cur_seg==MODE_WALK_OUT))[0]])
                        dist_seg=checkDist(dist_seg,max_walk*1000)
                else:
                    dist_seg = np.nansum(df['DISTANCE_DELTA'].values[mode_seg[0]:mode_seg[1]])
                if dist_seg==0 or np.isnan(dist_seg):
                    # filter out the zero or nan values of dist_seg
                    continue
                if mode_seg[2]==prev_mode:
                    # if the current mode is same to the previous one, combine the two distance
                    prev_dist = trip_return[dist_key][len(trip_return[dist_key])-1]
                    cur_dist = checkDist((prev_dist*1000+dist_seg) / 1000,dist_lim)
                    trip_return[dist_key][len(trip_return[dist_key])-1]=cur_dist
                    continue
                trip_return[mode_key].append(int(mode_seg[2])) # append the mode
                trip_return[dist_key].append(checkDist(dist_seg / 1000,dist_lim))
                prev_mode = mode_seg[2]

        return num_valid_mode_seg
                
##        go through each mode chunk
#        for mode_seg in mode_segs:
#            logging.debug("Mode Segs start index: " + str(mode_seg[0]))
#            logging.debug("Mode Segs end index: " + str(mode_seg[1]))
##            start_time = pred_times[mode_seg[0]]
##            end_time = pred_times[mode_seg[1]-1]
#            start_time = df['TIMESTAMP'][mode_seg[0]]
#            end_time = df['TIMESTAMP'][mode_seg[1]-1]
#            time_span_cur = end_time-start_time
#            if time_span_cur>=mode_thresh:
#                time_span.append(time_span_cur)
#                valid_mode_segs.append(mode_seg)
#                num_valid_mode_seg += 1
#        
#        if num_valid_mode_seg <= max_mode:
#            for mode_seg in valid_mode_segs:
#                if isAM:
#                    trip_return['am_mode'].append(mode_seg[2]) # append the mode
#                    # calculate the distance of this mode segment
#                    dist_seg = np.nansum(df['DISTANCE_DELTA'].values[mode_seg[0]:mode_seg[1]])
#                    trip_return['am_distance'].append(checkDist(dist_seg / 1000,dist_lim))
#                else:
#                    trip_return['pm_mode'].append(mode_seg[2]) # append the mode
#                    # calculate the distance of this mode segment
#                    dist_seg = np.nansum(df['DISTANCE_DELTA'].values[mode_seg[0]:mode_seg[1]])
#                    trip_return['pm_distance'].append(checkDist(dist_seg / 1000,dist_lim))
#
#        else:
#            # sort the time list and get the indices
#            sorted_time_indices=[i[0] for i in sorted(enumerate(time_span), key=lambda x:x[1])]
#            top_5_time_indices = sorted_time_indices[-5:]
#            top_5_time_indices.sort()
#            for i_seg in top_5_time_indices:
#                mode_seg = valid_mode_segs[i_seg]
#                if isAM:
#                    trip_return['am_mode'].append(mode_seg[2]) # append the mode
#                    # calculate the distance of this mode segment
#                    dist_seg = np.nansum(df['DISTANCE_DELTA'].values[mode_seg[0]:mode_seg[1]])
#                    trip_return['am_distance'].append(checkDist(dist_seg / 1000,dist_lim))
#                else:
#                    trip_return['pm_mode'].append(mode_seg[2]) # append the mode
#                    # calculate the distance of this mode segment
#                    dist_seg = np.nansum(df['DISTANCE_DELTA'].values[mode_seg[0]:mode_seg[1]])
#                    trip_return['pm_distance'].append(checkDist(dist_seg / 1000,dist_lim))

    def tripFind(homeLoc,schoolLoc,day_frame,predicted_modes,mode_thresh=120):
        """
        :param homeLoc: (Lat,Lon) tuple for home location
        :param schoolLoc: (Lat,Lon) tuple for school location
        :param data_frame: data frame containing all measured data
        :param predicted_modes: list containing predicted modes
        :return: returns the modes and distances for the trips to/from school in the morning/evening, as well as distances
        mode = string;
        distance = km;
        CO2= g / day of travel
        """

        #limits on return values for dist/CO2
        dist_lim=45.3
        co2_lim=17010.2
        time_lim=3.5
        max_mode=6
        max_walk=4.0

        day_frame['PRED_MODE']= pd.Series(predicted_modes, index=day_frame.index)

        #intialize returned trip
        trip_return = {'am_mode': [],'pm_mode': [],'am_distance': [],'pm_distance': [],'travel_co2':
                       0,'outdoor_time': 0}

        #find the first home - school trip, within the time range for school/home travel
        #process string date index, no conversion for timezone
        home_sch_set=day_frame.between_time('6:00','10:00')

#        home_sch_set=day_frame.between_time('6:00','9:30')

        if not home_sch_set.empty and homeLoc[0] is not None and schoolLoc[0] is not None:
            home_idx=distanceFcn(home_sch_set,homeLoc[0],homeLoc[1],True,True) #index in the dataframe when the student was closest to home
            school_idx=distanceFcn(home_sch_set,schoolLoc[0],schoolLoc[1],False,False) #index in the dataframe when the student was closest to school

            if home_idx['idx']<school_idx['idx']:

                # define a new dataframe for the trip by selecting according to the time index between  home/school
                df_am=home_sch_set[home_idx['idx']:school_idx['idx']+1]
                isAM = True
                num_mode_seg = segFind(df_am, trip_return, mode_thresh, isAM, dist_lim, max_mode, max_walk)

            else:
                logging.info("No Home-School trip time data available")

        else:
            logging.info("No Home-School trip time data available")

        #find the first school - home trip between the times expected to make that travel
        sch_home_set=day_frame.between_time('13:00','21:00')

        # for NRF test trip
        #sch_home_set=day_frame.between_time('9:31','11:00')

        if not sch_home_set.empty and homeLoc[0] is not None and schoolLoc[0] is not None:

            school_idx=distanceFcn(sch_home_set,schoolLoc[0],schoolLoc[1],True,False) #index in the dataframe when the student was closest to school
            home_idx=distanceFcn(sch_home_set,homeLoc[0],homeLoc[1],False,True) #index in the dataframe when the student was closest to home

            if school_idx['idx']<home_idx['idx']:
                # define a new dataframe for the trip by selecting according to the time index between home/school
                df_pm=sch_home_set[school_idx['idx']:home_idx['idx']+1]
                isAM = False
                num_mode_seg = segFind(df_pm, trip_return, mode_thresh, isAM, dist_lim, max_mode, max_walk)

            else:
                logging.info("No School-Home trip time data available")
        else:
            logging.info("No School-Home trip time data available")

        #calculate outside time in hours
        outdoor=day_frame.loc[day_frame['PRED_MODE'].isin([0,2,4,5,6])]
        if not outdoor.empty:
            trip_return['outdoor_time']= checkTime(outdoor['TIME_DELTA'].sum() / 3600.0,time_lim)

        #sum CO2 for AM and PM travel
#        print trip_return
        trip_return['travel_co2']+= getCO2(trip_return['am_mode'], trip_return['am_distance'])
        trip_return['travel_co2']+= getCO2(trip_return['pm_mode'], trip_return['pm_distance'])
        trip_return['travel_co2']=checkCO2(trip_return['travel_co2'],   co2_lim)

        return trip_return

    # start of process()
    pois, idx_of_pois = trip_segment(data_frame, stopped_thresh, poi_dwell_time)
    logging.debug("Number of POIs FOUND: " + str(len(pois)))
    logging.debug("POIs found:")
    logging.debug(pois)
    logging.info("Identify home and school locations")
    home_loc, school_loc = identify_home_school(pois, idx_of_pois, data_frame,
                                                school_start=school_start,
                                                school_end=school_end,
                                                home_start=home_start,
                                                home_end=home_end,
                                                max_school_thresh=max_school_thresh,
                                                round_decimals=home_school_round_decimals,
                                                poi_cover_range=poi_cover_range)
    logging.debug("home location: " + str(home_loc))
    logging.debug("school location: " + str(school_loc))

    #Create new dataframe which is indexed by local time
    ts_date = data_frame[['TIMESTAMP']].values[:,0] # take in UNIX time
    local_ts = ts_date+28800 # add offset to change to SGT
    local_ts_date = pd.to_datetime(local_ts,unit='s')  # convert local time in second to local datetime
    # set_index creates a new object and returns it
    dfnew = data_frame.set_index(local_ts_date)
    daily_trips = tripFind(home_loc, school_loc,dfnew,predicted_modes,mode_thresh)

    #set the trips dict to be empty if there is no AM or PM mode found
#    if len(daily_trips['am_mode'])==0:
#        daily_trips.pop('am_mode',None)
#    if len(daily_trips['pm_mode'])==0:
#        daily_trips.pop('pm_mode',None)
    #if not 'am_mode' in daily_trips and not 'pm_mode' in daily_trips:
    #    daily_trips = {}
    return daily_trips, home_loc, school_loc



def identify_home_school(pois, idx_of_pois, data_frame, school_start=9,
                         school_end=13, home_start=22, home_end=5,
                         max_school_thresh=100, round_decimals=4,
                         poi_cover_range = 30):
    """Identify home and school locations. Input is a list of point of
    interests tuples. Each tuple has the number of measurements, start
    and end index in the original data frame (starting from zero),
    start and end timestamp, start and end latitude and longitude.
    Find the tuples that fall between start and end time and average
    the locations. Return two tuples representing home (lat, lon) and
    school (lat, lon) locations. If there are no POIs identified for
    home or school, the tuple will be (None, None).

    pois is a list of all the unique pois generated by trip_segment
    idx_of_pois is a list of all the indices of the correspoing pois
    data_frame is the pandas data frame with the original data.
    school_start is the hour of the day when school starts. Default 9am.
    school_end is the hour of the day when school end. Default 1pm.
    home_start is the first hour of the day when students are assumed to be
    home at night. Default 10pm.
    home_end is the last hour of the day when students are assumed to be
    home at night. Default 5am.
    max_school_thresh is the threshold for school/home distances smaller than
    which the home/school poi is rejected - to negate the possibility
    of creating home/school links for sensors left at school by mistake
    round_decimals is the number of decimals used in the max_freq
    heuristic for rounding lat / lon values before taking the most
    frequent value
    time_offset is the offset in hours to subtract from each timestamp. Default 8.

    """

    ########## identify home/school completely same as MATLAB version ############
    # check the length of pois
    if len(pois)<1:
        logging.info("No poi is passed in")
        return (None, None), (None, None)

    # get out time data
    all_timestamp = data_frame[['TIMESTAMP']].values
    all_delta_time = data_frame[['TIME_DELTA']].values

    # initialization
    time_at_school = []
    time_at_home = []
    lat_pois = []
    lon_pois = []

    # go though each poi
    for i_poi in xrange(0,len(pois)):
        poi= pois[i_poi]
        idx_of_poi = idx_of_pois[i_poi]

        # lat/lon values to return
        lat_pois.append(poi[0])
        lon_pois.append(poi[1])

        # calculate the distance between each point and the poi
        latlon = data_frame[['WLATITUDE', 'WLONGITUDE']].values
        dist_to_poi = map(lambda x: great_circle_dist(x, poi, unit="meters"), latlon)

        # find the indices of points that are near to this poi
        idx_near_poi = [x for x in xrange(0,len(dist_to_poi)) if dist_to_poi[x]<=poi_cover_range]
#        idx_near_poi = np.where(np.array(dist_to_poi)<=poi_cover_range)[0].tolist() # another way to find the indices, has warning

        # combine the indices found with the earlier poi indices
        idx_of_pois_new = np.unique(idx_of_poi+idx_near_poi).tolist()

        poi_timestamp = all_timestamp[idx_of_pois_new]
        poi_delta_time = all_delta_time[idx_of_pois_new]
        poi_hourtime = np.array(map(lambda x: get_hour_SGT(x), poi_timestamp))

        # find the idx of points among current idx_poi which fit school time range
        idx_at_school = np.where((poi_hourtime >= school_start) & (poi_hourtime < school_end))[0]
        time_at_school.append(np.nansum(poi_delta_time[idx_at_school]))

        # find the idx of points among current idx_poi which fit home time range
        idx_at_home = np.where((poi_hourtime < home_end) | (poi_hourtime >= home_start))[0]
        time_at_home.append(np.nansum(poi_delta_time[idx_at_home]))

    if len(time_at_school)==0 and len(time_at_home)==0:
        raise Exception("No poi is passed in!")
        return (None, None), (None, None)
    else:
        # get school
        max_sch_cnt = max(time_at_school)
        idx_max_sch_cnt = np.argmax(time_at_school)
        # get home
        max_home_cnt = max(time_at_home)
        idx_max_home_cnt = np.argmax(time_at_home)
        
        # if max_sch_cnt and max_home_cnt are all zero, get all loc as None
        if max_sch_cnt==0 and max_home_cnt==0:
            return (None, None), (None, None)
        # if the same pois is detected as home or school
        # decide by the school/home time
        elif idx_max_sch_cnt==idx_max_home_cnt:
            if max_sch_cnt > max_home_cnt:
                school_lat = lat_pois[idx_max_sch_cnt]
                school_lon = lon_pois[idx_max_sch_cnt]
                home_lat = np.nan
                home_lon = np.nan
            elif max_sch_cnt < max_home_cnt:
                home_lat = lat_pois[idx_max_home_cnt]
                home_lon = lon_pois[idx_max_home_cnt]
                school_lat = np.nan
                school_lon = np.nan
            else:
                return (None, None), (None, None)
                
        else:
            # only if there are hits for school time, assign school
            if max_sch_cnt>0:
                school_lat = lat_pois[idx_max_sch_cnt]
                school_lon = lon_pois[idx_max_sch_cnt]
            else:
                school_lat = np.nan
                school_lon = np.nan
            # only if there are hits for home time, assign home
            if max_home_cnt>0:
                home_lat = lat_pois[idx_max_home_cnt]
                home_lon = lon_pois[idx_max_home_cnt]
            else:
                home_lat = np.nan
                home_lon = np.nan
        
        # sort out home/school pairings < YY km away - these are anomolies
        school_home_dist = great_circle_dist([school_lat,school_lon],[home_lat,home_lon],unit="meters")
        if school_home_dist < max_school_thresh:
            logging.info("home school distance:" + str(school_home_dist))            
            return (None, None), (None, None)

        if ~np.isnan(home_lat) and ~np.isnan(school_lat):
            return (home_lat, home_lon), (school_lat, school_lon)
        elif ~np.isnan(home_lat) and np.isnan(school_lat):
            return (home_lat, home_lon), (None, None)
        elif np.isnan(home_lat) and ~np.isnan(school_lat):
            return (None, None), (school_lat, school_lon)
        else:
            return (None, None), (None, None)


def trip_segment(data_frame, stopped_thresh=0.5, stopped_dwell=480):
    """Find POI's in the data frame, return a list of POIs and a list of
       indices of the poi points inside data_frame. Implements logic
       from Yuren's Matlab code.

    """
    
    def store_poi(data_frame, idx_buffer, stop_time):
        """Identify the lat/lon location and the indices in the data frame for
        a POI. idx_buffer has the indices in the data frame for this POI.

        """
        # if the total stop time is larger than stopped_dwell,
        # then a poi is detected, otherwise return
        if stop_time <= stopped_dwell:
            return
        # select lat/lon locations for this POI, assignment makes a copy of the rows
        df_stop_all = data_frame.loc[idx_buffer][['WLATITUDE', 'WLONGITUDE']]
        # check if all location are nan
        lat_stop = df_stop_all['WLATITUDE'].values
        if np.all(np.isnan(lat_stop)):
            return
        df_stop_all = df_stop_all.apply(round_values)
        # get the most frequent poi of this stop segment
        df_poi_cnt = df_stop_all.groupby(['WLATITUDE', 'WLONGITUDE']).size()
        poi_lat, poi_lon = df_poi_cnt.idxmax()
        # record poi lat/lon and the indices that correspond to it
        pois['poi_latlon'].append([poi_lat,poi_lon])
        pois['poi_idx'].append(idx_buffer)
        pois['last_time'].append(stop_time)
#        pois[(poi_lat,poi_lon)].extend(idx_buffer)

    # start of trip_segment()
    pois = defaultdict(list)
    round_decimal = 4;
    dist_2comb = 30; # m, distance close to which the two pois are combined
    round_values = partial(pd.Series.round, decimals=round_decimal)
    stop_time = 0
    idx_buffer = []
    for index, row in data_frame.iterrows():
        if row['AVE_VELOCITY'] < stopped_thresh:
            # if stop time is zero we are at the beginning of a new stop
            # reset the idx_buffer
            if stop_time == 0:
                idx_buffer = []
            # remember that we are stopped this index
            idx_buffer.append(index)
            # add the time delta of this row to the time we are stopped here
            stop_time += row['TIME_DELTA']
        else:
            # we are moving, check if we have just concluded a POI
            # and store the POI
            if stop_time > 0:
                store_poi(data_frame, idx_buffer, stop_time)
            stop_time = 0

    # process the last stop after exiting loop
    if stop_time > 0:
        store_poi(data_frame, idx_buffer, stop_time)
    
        # combine the close pois
    pois_latlon = pois['poi_latlon']
    num_pois = len(pois_latlon)
    if num_pois>1:
        cur_poi_latlon = pois_latlon[0]
        unique_pois = defaultdict(list)
        unique_pois['poi_latlon'].append(pois['poi_latlon'][0])
        unique_pois['poi_idx'].append(pois['poi_idx'][0])
        unique_pois['last_time'].append(pois['last_time'][0])
        i_unique_poi = 0
        last_time_prev = 0
        for i_poi in xrange(1,num_pois):
            dist = great_circle_dist(cur_poi_latlon,pois_latlon[i_poi],'meters')
            if dist<dist_2comb:
                # if the two pois are close, combine as one based on the time
                unique_pois['poi_idx'][i_unique_poi].extend(pois['poi_idx'][i_poi])
                unique_pois['last_time'][i_unique_poi]=unique_pois['last_time'][i_unique_poi]+pois['last_time'][i_poi]
                if pois['last_time'][i_poi]>last_time_prev:
                    unique_pois['poi_latlon'][i_unique_poi]=pois['poi_latlon'][i_poi]
                    cur_poi_latlon = pois_latlon[i_poi]
                    last_time_prev=pois['last_time'][i_poi]
            else:
                # if the two pois are far, insert a new unique poi
                i_unique_poi = i_unique_poi+1
                cur_poi_latlon = pois_latlon[i_poi]
                unique_pois['poi_latlon'].append(pois['poi_latlon'][i_poi])
                unique_pois['poi_idx'].append(pois['poi_idx'][i_poi])
                unique_pois['last_time'].append(pois['last_time'][i_poi])
                last_time_prev=pois['last_time'][i_poi]
        
        return unique_pois['poi_latlon'], unique_pois['poi_idx']
    # keys() and values() in a dictionary respect the same order
#    return pois.keys(), pois.values()
    return pois['poi_latlon'], pois['poi_idx']
