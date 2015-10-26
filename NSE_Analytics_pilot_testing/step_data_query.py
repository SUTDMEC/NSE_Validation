"""Queries device analysis status for a given device list and date range.

Usage: step_data_query.py [--deviceIDs=DEVICEFILE] URL

Arguments:
 URL            Base URL for the backend API, for example 'http://sensg.ddns.net/api/'

Options:
 --deviceIDs=DEVICEFILE    mandatory option with the filename with device IDs. Format is one ID per line.

"""


import datetime
from datetime import timedelta
import logging
import sys
import process
import requests
import time
import base64
import os
import json
import calendar
import pandas as pd 

def getDataFull(url, nid, start_time=0, end_time=int(time.time()), table=None):
    """Retrieve raw hardware data for device nid for the specified time
    frame and specified table if specified.  Return a pandas data frame of
    measurements or None if no data was returned.

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'nid' : nid, 'start' : start_time, 'end' : end_time, 'ts':int(time.time())}
    if table:
        payload['table'] = table

    header = {"Content-Type": "application/json", 'Authorization' : 'Basic %s' % base64.b64encode("sutd-nse_api:UZZbhMTNTIjTKqtOXJ3jReBnbbkTIWnxXiqIhXuKyZrHRRWS6cLpvZ3YcYXpITC")}
    req = requests.get("%s/getdatafull" % url, params=payload, headers=header)
    logging.debug("getdata url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getData returned http status %d" % req.status_code)
    resp = req.json()
    # resp["success"] has type bool
    if resp["success"]:
        return pd.DataFrame(resp["data"])
    else:
        logging.warning("getData for " +str(nid) + " for end: " + str(end_time) + " returned %s" % resp["error"])
        return None




def getAnalysis(url, nid, date):
    """Get the dates where no successful analysis has been performed for device nid yet.
    Return list of dates ('YYYY-MM-dd')

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'nid' : nid, 'date' : date, 'ts':int(time.time())}

    header = {"Content-Type": "application/json", 'Authorization' : 'Basic %s' % base64.b64encode("sutd-nse_api:UZZbhMTNTIjTKqtOXJ3jReBnbbkTIWnxXiqIhXuKyZrHRRWS6cLpvZ3YcYXpITC")}
    req = requests.get("%s/getdailysummary" % url, params=payload, headers=header)
    logging.debug("getAnalysis url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getAnalysis returned http status %d" % req.status_code)
    resp = req.json()
    return resp

def getAnalysisAvg(url, date):
    """Get the dates where no successful analysis has been performed for device nid yet.
    Return list of dates ('YYYY-MM-dd')

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = { 'date' : date, 'ts':int(time.time())}

    header = {"Content-Type": "application/json", 'Authorization' : 'Basic %s' % base64.b64encode("sutd-nse_api:UZZbhMTNTIjTKqtOXJ3jReBnbbkTIWnxXiqIhXuKyZrHRRWS6cLpvZ3YcYXpITC")}
    req = requests.get("%s/getanalysednationalavg" % url, params=payload, headers=header)
    logging.debug("getAnalysis url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getAnalysisAvg returned http status %d" % req.status_code)
    resp = req.json()
    return resp


def getAvgSteps(url):
    """Get the dates where no successful analysis has been performed for device nid yet.
    Return list of dates ('YYYY-MM-dd')

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = { 'ts':int(time.time())}

    header = {"Content-Type": "application/json", 'Authorization' : 'Basic %s' % base64.b64encode("sutd-nse_api:UZZbhMTNTIjTKqtOXJ3jReBnbbkTIWnxXiqIhXuKyZrHRRWS6cLpvZ3YcYXpITC")}
    req = requests.get("%s/stats/avg" % url, params=payload, headers=header)
    logging.debug("getAnalysis url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getAnalysisAvg returned http status %d" % req.status_code)
    resp = req.json()

    return resp


def getSteps(url, nid):
    """Get the total steps for a given device

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'ts':int(time.time())}

    header = {"Content-Type": "application/json", 'Authorization' : 'Basic %s' % base64.b64encode("sutd-nse_api:UZZbhMTNTIjTKqtOXJ3jReBnbbkTIWnxXiqIhXuKyZrHRRWS6cLpvZ3YcYXpITC")}
    req = requests.get(url+"/stats/total/dev/"+str(nid), params=payload, headers=header) #
    logging.debug("getAnalysis url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getAnalysisAvg returned http status %d" % req.status_code)
    resp = req.json()

    return resp

def getDailySteps(url, nid,date):
    """Get the total steps for a given device

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'ts':int(time.time())}

    header = {"Content-Type": "application/json", 'Authorization' : 'Basic %s' % base64.b64encode("sutd-nse_api:UZZbhMTNTIjTKqtOXJ3jReBnbbkTIWnxXiqIhXuKyZrHRRWS6cLpvZ3YcYXpITC")}
    req = requests.get(url+"/stats/total/dev/"+str(nid)+'/'+str(date), params=payload, headers=header) #
    logging.debug("getAnalysis url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getAnalysisAvg returned http status %d" % req.status_code)
    resp = req.json()

    return resp




def main(url, device_file):
    """Main processing loop. It expects a base url and the file name of
    the csv file of device ids. If testing is True the function
    returns detailed information for debugging. If the current_date
    string (%Y-%m-%d) is provided this is taken as the current date
    when processing is done, otherwise use the localtime to determine
    the date.

    """

    #create dump file
    nse_directory = os.path.dirname(os.path.realpath(__file__))
    target_stats = open("%s/step_data_stats.txt" % nse_directory , 'w') #_" + str(start_date) + "_" + str(end_date) + "

    # create logger
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    # load list of device IDs from file
    logging.info("Load device IDs")
    try:
        with open(device_file, 'r') as csvfile:
            device_ids = [ int(line.strip()) for line in csvfile if line.strip() ]
    except IOError as e:
        logging.error("Failed to load device IDs: %s" %  e.strerror)
        sys.exit(10)


    start_get = 1443343785 # Sun, 27 Sep 2015 08:49:45 GMT              int(process.getFirstSecondOfDay(analysis_unix))+12*3600 #12 pm of the analysis day
    end_get = 1443775785# Thu, 01 Oct 2015 08:49:45 GMT    int(start_get+24*3600-1) #12 pm of the day after the analysis day

    target_stats.write("NID, data_len,tot_steps,first_ts,last_ts")
    target_stats.write("\n")

    for nid in device_ids:

        GTMODE_data_frame = getDataFull(url, nid,start_get,end_get)
        if GTMODE_data_frame is None:
            data_len=0
            first_ts=0
            last_ts=0
        else:
            data_len=len(GTMODE_data_frame[['TIMESTAMP']].values[:,0])
            first_ts=GTMODE_data_frame[['TIMESTAMP']].values[0,0]
            last_ts=GTMODE_data_frame[['TIMESTAMP']].values[-1,0]

        nidStep=getSteps(url, nid)
        logging.info("NID: %s" +str(nid)+" has steps: "+ str(nidStep))
        steps=nidStep['steps']

        target_stats.write(str(nid)+','+ str(data_len) + ',' + str(steps) + ',' + str(first_ts) + ',' + str(last_ts))
        target_stats.write("\n")

    target_stats.close()

if __name__ == "__main__":
    from docopt import docopt
    # parse arguments
    arguments = docopt(__doc__)

    # deviceIDs is a mandatory option
    device_file = arguments['--deviceIDs']
    main(arguments['URL'], device_file)