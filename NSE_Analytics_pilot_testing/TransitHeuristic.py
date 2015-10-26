"""Heuristi for predicting if a motorized jorney is likely made by
public transit (bus, train) or by individual transport (car), based on
a heuristic of proximity to bus or train stops and bus or train route.

"""

import csv
from rtree import index
from collections import defaultdict
import logging
import pandas as pd
import numpy as np
import os
from itertools import ifilter

from predict_mode import AbstractPredictor, getStartEndIdx
from predict_mode import MODE_STOP_OUT, MODE_STOP_IN, MODE_WALK_OUT, MODE_WALK_IN, MODE_TRAIN, MODE_BUS, MODE_CAR, MODE_TBD
from util import great_circle_dist, chunks


def build_busstop_map(bus_stop_location_file, bus_route_file):
    """Pre-process bus stop locations and bus routes data.  The input is
    the filenames of two input csv data files.  bus_stop_location_file
    contains one bus stop per line: busstopID , busstop lon, busstop
    lan.  bus_route_file contains on each line bus service id,
    direction, sequence of stop on the route, busstop Returns a map
    from bus stop ids to lat/lon pair and a mapping from bus stop
    string ids to a list of all bus services stopping at that bus
    stop.

    """
    # read bus stop locations
    with open(bus_stop_location_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # each row contains (bus_stop_id, lon, lat)
        # rows starting with # are comments
        # map bus stop id to lat/lon pair
        busstop_to_location = {row[0]: (float(row[2]), float(row[1]))
                               for row in csv_reader if len(row) == 3 and not row[0].startswith("#")}

    with open(bus_route_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # each row contains (# bus service, direction, sequence of
        # stop on this route, bus stop id)
        # rows starting with # are # comments
        # map bus stop id to list of bu services serving that stop
        busstop_to_busroute = defaultdict(list)
        for row in csv_reader:
            if len(row) != 4 or row[0].startswith("#"):
                continue
            try:
                bus_stop, bus_service = row[3], row[0]
                busstop_to_busroute[bus_stop].append(bus_service)
            except KeyError:
                logging.warning("bus stop id %s has no location. skipped." % bus_stop)
    return busstop_to_location, busstop_to_busroute


def build_train_map(train_file):
    """Pre-process train station locations and routes data.  The input is
    the filename of an input csv data file that contains one train
    station per line: stationID , station name, lineID, sequence on
    the line, location(lon, lat). Returns a map from train station ids to
    lat/lon pair and a mapping from train station ids to a list of
    all train services stopping at that station.

    """
    trainstation_to_location = {}
    trainstation_to_trainroute = defaultdict(list)
    with open(train_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            # rows starting with # are comments
            if row[0].startswith("#"):
                continue
            stationId , station_name, lineId, _ , location = row
            # NOTE: stations with multiple lines appear multiple times
            # with slighlty different locations. We do not care and
            # just overwrite the locations which is okay for our use case
            lon, lat = map(float, location.lstrip("POINT (").rstrip(")").split()[:2])
            trainstation_to_location[stationId] = (lat, lon)
            trainstation_to_trainroute[stationId].append(lineId)
    return trainstation_to_location, trainstation_to_trainroute


def build_station_rtree(station_to_location):
    """Create Rtree index for stations. station_to_location is a
    dictionary from station IDs (strings) to location (lat/lon)
    pairs. The string station ID is attached as an object to each
    point in the index. Returns the Rtree index

    """
    # create Rtree index
    rtree = index.Index()
    for station_idx, (stationID, station_location) in enumerate(station_to_location.iteritems()):
        # station_idx: numeric bus stop index starting from zero, only used internally
        # stationID: station string ID
        # station_location: (lat, lon) location of station
        # include station string id as object in the index
        rtree.insert(station_idx, (station_location[0],
                                   station_location[1],
                                   station_location[0],
                                   station_location[1]), obj=stationID)
    return rtree


def find_nearest_station(lat, lon, rtree, threshold=100):
    """Find neares station(s) given a lat/lon position.  Input is a
    lat/lon location and the Rtree spatial index of stations. Return a
    list of tuples. Each tuple contains (station id, distance to the
    station) if the distance is smaller threshold in meters. If there
    is more than one nearest equidistant stations the list contains
    more than one tuple otherwise just one tuple.

    """
    # rtree_entry.object is the string station id
    # NOTE: bounds coordinates are 'interleaved' by default
    # insert as [x, y, x, y] and returned as [xmin, xmax, ymin, ymax]
    # See http://toblerity.org/rtree/tutorial.html
    return [ (rtree_entry.object,
              great_circle_dist((lat,lon),(rtree_entry.bounds[0],
                                           rtree_entry.bounds[2]), unit="meters")) for rtree_entry
             in rtree.nearest((lat,lon,lat,lon), num_results=1,
                              objects=True) if
             great_circle_dist((lat,lon),(rtree_entry.bounds[0],
                                          rtree_entry.bounds[2]), unit="meters") < threshold ]


def pass_any_route(route_set_list, thre):
    #The method returns true if the bus stops follow the same route
    all_route_set = set([])
    for route_set in route_set_list:
        for route in route_set:
            all_route_set.add(route)
    for route in all_route_set:
        counter = 0
        for route_set in route_set_list:
            if (route in route_set):
                counter += 1
        if counter >= thre:
            return True
    return False


def predict_mode_by_location(lat, lon, station_location_tree,
                             station_location_dict, transit_route_dict,
                             dist_thre = 50,
                             dist_pass_thres = 7, num_stops_thre = 3,
                             dist_pass_thres_perc = 0.2):
    #The method predicts whether a list of position follows a bus or train
    #route by detecting the nearest bus stops or train stations
    num_valid_pt = len(lat)

    if(num_valid_pt <= 5):
        dist_pass_thres = max([num_valid_pt/5.0,1])
    else:
        dist_pass_thres = max([num_valid_pt*dist_pass_thres_perc,dist_pass_thres])
    route_set_list = [];
    transit_stop_set = set([])
    dist_pass = 0
    for idx in range(0,num_valid_pt):
        route_set = set([])
        if np.isnan(lat[idx]) or np.isnan(lon[idx]):
            continue
        nearest_station_list = find_nearest_station(lat[idx], lon[idx], station_location_tree, dist_thre)
        for station in nearest_station_list:
            dist_pass += 1
            transit_stop_set.add(station[0])
            transit_route_list = transit_route_dict[station[0]]
            for transit_route in transit_route_list:
                route_set.add(transit_route[0])
            if len(route_set)>0:
                route_set_list.append(route_set)
    if (len(transit_stop_set)>num_stops_thre) or (dist_pass > dist_pass_thres):
        if(pass_any_route(route_set_list, len(transit_stop_set))):
            return True
    return False


class BusMapPredictor(AbstractPredictor):
    """Bus predictor that determine if travel mode is bus, based on bus
    stop and route mapping by Wang Jin."""

    def __init__(self, dist_thres_entry_exit=50):
        nse_directory = os.path.dirname(os.path.realpath(__file__))
        self.busstop_location_dict, self.busstop_route_dict = build_busstop_map(os.path.join(nse_directory, "bus_stop_location.csv"), os.path.join(nse_directory, "bus_stop_list.csv"))
        self.bus_location_tree = build_station_rtree(self.busstop_location_dict)
        self.dist_thres_entry_exit = dist_thres_entry_exit

    def fit(self, data, target):
        pass

    def predict(self, data, modes):
        """predict whether a list of position follows a bus route by detecting
        the nearest bus stops. Input is the pandas data frame of
        measurements and an array of current mode predictions.  Returns
        an array of predicted modes of the same size as the input data
        frame has rows.

        """
        # extract lat/lon from data frame
        lat = data['WLATITUDE'].values
        lon = data['WLONGITUDE'].values
        # array of indices of motorized mode, if no motorized mode, return
        idx_motor = np.where(modes == MODE_CAR)[0]
        if len(idx_motor) == 0:
            return modes
        start_idx_motor, end_idx_motor, num_motor_seg = getStartEndIdx(idx_motor)

        for i_seg in xrange(0,num_motor_seg):
            start_idx = start_idx_motor[i_seg]
            end_idx = end_idx_motor[i_seg]
            # test for distance first
            lat_seg = lat[start_idx:end_idx+1]
            lon_seg = lon[start_idx:end_idx+1]
            valid_lat_seg = lat_seg[np.where(np.invert(np.isnan(lat_seg)))[0]]
            valid_lon_seg = lon_seg[np.where(np.invert(np.isnan(lon_seg)))[0]]


            is_bus = predict_mode_by_location(valid_lat_seg,
                                             valid_lon_seg,
                                             self.bus_location_tree,
                                             self.busstop_location_dict,
                                             self.busstop_route_dict)

            #check entry point distance
            entry_pt_near = -1
            exit_pt_near = -1

            if start_idx-1>=0:
                if not np.isnan(lat[start_idx-1]):
                    nearest_busstops = find_nearest_station(lat[start_idx-1],lon[start_idx-1],self.bus_location_tree,self.dist_thres_entry_exit)
                    if len(nearest_busstops)!=0:
                        #print nearest_busstops[0][2]
                        entry_pt_near = 1
                    else:
                        entry_pt_near = 0

            if end_idx+1 < len(modes):
                if not np.isnan(lat[end_idx+1]):
                    nearest_busstops = find_nearest_station(lat[end_idx+1],lon[end_idx+1],self.bus_location_tree,self.dist_thres_entry_exit)
                    if len(nearest_busstops)!=0:
                        #print nearest_busstops[0][2]
                        exit_pt_near = 1
                    else:
                        exit_pt_near = 0
            # print "# of trip points:",end_idx - start_idx+1
            # print "# of points with valid positions:", len(valid_lat_seg)
            if is_bus or entry_pt_near + exit_pt_near == 2:
                # print "Bus"
                # print "---"
                modes[start_idx:end_idx+1] = 5
            #else:
            #    print "Car"
            #    print "---"
        return modes



class TrainPredictor(AbstractPredictor):
    """Train predictor that determine if travel mode is by train, based on train stations
    and routes.

    """

    def __init__(self, dist_thres_entry_exit=50):
        nse_directory = os.path.dirname(os.path.realpath(__file__))
        self.train_location_dict, self.train_route_dict = build_train_map(os.path.join(nse_directory, "sg_mrt.csv"))
        self.train_location_tree = build_station_rtree(self.train_location_dict)
        self.dist_thres_entry_exit = dist_thres_entry_exit

    def fit(self, data, target):
        pass

    def predict(self, data, modes):
        """predict whether a list of position follows atrain route by detecting
        the nearest train stops. Input is the pandas data frame of
        measurements and an array of current mode predictions.  Returns
        an array of predicted modes of the same size as the input data
        frame has rows.

        """
        # extract lat/lon from data frame
        lat = data['WLATITUDE'].values
        lon = data['WLONGITUDE'].values

        # chunk is a tuple (start_idx, end_idx, mode)
        for start_idx, end_idx, _ in ifilter(lambda chunk: chunk[2] in [MODE_CAR, MODE_BUS, MODE_TRAIN],
                                             chunks(modes, include_values=True)):
            # test for distance first
            lat_seg = lat[start_idx:end_idx]
            lon_seg = lon[start_idx:end_idx]
            valid_lat_seg = lat_seg[np.where(np.invert(np.isnan(lat_seg)))[0]]
            valid_lon_seg = lon_seg[np.where(np.invert(np.isnan(lon_seg)))[0]]

            if len(valid_lon_seg) == 0:
                continue
            # TODO: parameters have to be tuned carefully
            is_train = predict_mode_by_location(valid_lat_seg,
                                                valid_lon_seg,
                                                self.train_location_tree,
                                                self.train_location_dict,
                                                self.train_route_dict,
                                                dist_thre = 400,
                                                dist_pass_thres = 7, 
                                                num_stops_thre = 3,
                                                dist_pass_thres_perc = 0.7)

            #check entry point distance
            entry_pt_near = -1
            exit_pt_near = -1

            if start_idx-1>=0:
                if not np.isnan(lat[start_idx-1]):
                    nearest_station = find_nearest_station(lat[start_idx-1], lon[start_idx-1], self.train_location_tree, self.dist_thres_entry_exit)
                    if len(nearest_station)!=0:
                        entry_pt_near = 1
                    else:
                        entry_pt_near = 0

            if end_idx < len(modes):
                if not np.isnan(lat[end_idx]):
                    nearest_station = find_nearest_station(lat[end_idx],lon[end_idx],
                                                           self.train_location_tree,
                                                           self.dist_thres_entry_exit)
                    if len(nearest_station)!=0:
                        exit_pt_near = 1
                    else:
                        exit_pt_near = 0
            if is_train or entry_pt_near + exit_pt_near == 2:
                modes[start_idx:end_idx] = MODE_TRAIN
            else:
                modes[start_idx:end_idx] = MODE_CAR
        return modes
