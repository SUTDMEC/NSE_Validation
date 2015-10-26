"""Main test script for the analytics functions with given parameters.

Usage: analytics_test.py

"""
import requests
import logging
import sys
import pandas as pd
import numpy as np
import os
import sys
import base64
import datetime
import calendar

import process
from util import great_circle_dist


def main():
    # create logger to save
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    ## parameters
    #API to call
    url="https://data.nse.sg"
    nse_directory = os.path.dirname(os.path.realpath(__file__))
#    device_file="%s/exp1_devices_notripswithdata_Sept29(2)_short.csv" % nse_directory
    device_file="%s/nse_v4_deployment1.csv" % nse_directory
    current_date="2015-10-02"
    # geojson ouput file
    geo_json_file="%s/result.geojson" % nse_directory

    # load list of device IDs from file
    logging.info("Load device IDs")
    try:
        with open(device_file, 'r') as csvfile:
            devices = [ int(line.strip()) for line in csvfile if line.strip() ]
    except IOError as e:
        logging.error("Failed to load device IDs: %s" %  e.strerror)
        sys.exit(10)

    #return results from process
    results, geojson = process.main(url, device_file,
                                    current_date=current_date, testing=True)    
    with open(geo_json_file, 'w') as fout:
        fout.write('{ "type": "FeatureCollection",\n    "features": [\n' + \
                   ', '.join(geojson) + \
                   '\n    ]\n }')


if __name__ == "__main__":

    main()
