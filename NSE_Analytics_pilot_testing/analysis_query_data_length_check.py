"""Queries device analysis status for a given device list and date range.

Usage: analysis_query.py [--deviceIDs=DEVICEFILE --start_date=DATE --end_date=DATE] URL

Arguments:
 URL            Base URL for the backend API, for example 'http://sensg.ddns.net/api/'

Options:
 --deviceIDs=DEVICEFILE    mandatory option with the filename with device IDs. Format is one ID per line.

 --start_date=DATE       option to specify today's date for test purposes (%Y-%m-%d), otherwise the local server time is used to determine the date. The data to be processed is 2 days before today's date

 --end_date=DATE       option to specify today's date for test purposes (%Y-%m-%d), otherwise the local server time is used to determine the date. The data to be processed is 2 days before today's date

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
import numpy as np

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


def getSteps(url, nid,date):
    """Get the dates where no successful analysis has been performed for device nid yet.
    Return list of dates ('YYYY-MM-dd')

    """
    # note: int 'ts' is added in order to work around API caching issue
    payload = {'ts':int(time.time())}

    header = {"Content-Type": "application/json", 'Authorization' : 'Basic %s' % base64.b64encode("sutd-nse_api:UZZbhMTNTIjTKqtOXJ3jReBnbbkTIWnxXiqIhXuKyZrHRRWS6cLpvZ3YcYXpITC")}
    req = requests.get(url+"/stats/total/dev/"+str(nid)+'/'+date, params=payload, headers=header) #
    logging.debug("getAnalysis url: %s" % str(req.url))
    if req.status_code != requests.codes.ok:
        raise Exception("getAnalysisAvg returned http status %d" % req.status_code)
    resp = req.json()

    return resp




def main(url, device_file, start_date, end_date):
    """Main processing loop. It expects a base url and the file name of
    the csv file of device ids. If testing is True the function
    returns detailed information for debugging. If the current_date
    string (%Y-%m-%d) is provided this is taken as the current date
    when processing is done, otherwise use the localtime to determine
    the date.

    """

    record_modes=True

    #create dump file
    nse_directory = os.path.dirname(os.path.realpath(__file__))
#    target = open("%s/analysis_test.txt" % nse_directory , 'w') #_" + str(start_date) + "_" + str(end_date) + "
    target = open("%s/valid_nid.txt" % nse_directory , 'w') #_" + str(start_date) + "_" + str(end_date) + "

    nse_directory = os.path.dirname(os.path.realpath(__file__))
    target_stats = open("%s/analysis_stats.txt" % nse_directory , 'w') #_" + str(start_date) + "_" + str(end_date) + "

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

    d1=datetime.datetime.strptime(start_date,"%Y-%m-%d").date()
    d2=datetime.datetime.strptime(end_date,"%Y-%m-%d").date()

    datelist = [d1 + timedelta(days=x) for x in range((d2-d1).days + 1)]

    json_out=[]

    step_resp= getAvgSteps(url)
    logging.info("National Average steps are: %s" % str(step_resp))
    json_out.append({'NationalSteps': step_resp})
    #target.write("National average steps " + str(lat_home)+','+str(lon_home)+ "\n")
    for analysis_date in datelist:
        logging.info("Getting analysis data for: %s" % analysis_date)

        analysis_date_tuple = datetime.datetime.strptime(analysis_date.strftime("%Y-%m-%d"), "%Y-%m-%d") - datetime.timedelta(days=1)
        analysis_unix = calendar.timegm(analysis_date_tuple.timetuple())
        start_get = int(process.getFirstSecondOfDay(analysis_unix))+12*3600 #12 pm of the analysis day
        end_get = int(start_get+24*3600-1) #12 pm of the day after the analysis day


        target_stats.write("For the analysis date:, " + analysis_date.strftime("%Y-%m-%d")+ ',')
        target_stats.write("\n")

        target_stats.write("NID, #AM,#PM,distAM,distPM,outdoor,co2,steps")
        target_stats.write("\n")

        resp_nat=getAnalysisAvg(url, analysis_date)
        if resp_nat['success']:
            logging.info("National Average: %s" % str(resp_nat))
        node_json=[]
        long_nid=[]
        num_valid_nid = 0
        short_nid = []
        num_short_nid = 0
        
        for nid in device_ids:
            resp=getAnalysis(url, nid, analysis_date)
            GTMODE_data_frame = process.getData(url, nid,start_get,end_get)
            if GTMODE_data_frame is None:
                modes=[]
                timestamp=[]
                lat = []
                lon = []
                logging.warning("No data is read out!")
            else:
                logging.warning("Data of %d is read out!" % nid)
                modes=np.int32(GTMODE_data_frame[['CMODE']].values[:,0])
                timestamp=np.int32(GTMODE_data_frame[['TIMESTAMP']].values[:,0])
                lat=GTMODE_data_frame[['WLATITUDE']].values[:,0]
                lon=GTMODE_data_frame[['WLONGITUDE']].values[:,0]
                num_pt = len(GTMODE_data_frame)
                time_span = (timestamp[num_pt-1]-timestamp[0])/3600
                if num_pt>500 and time_span>=18:
                    num_valid_nid += 1
                    long_nid.append(np.int32(nid))
                    logging.warning("long nid: "+str(nid)+", number of points: "+str(num_pt)+", time span: "+str(time_span))
                else:
                    num_short_nid += 1
                    short_nid.append(np.int32(nid))
                    logging.warning("short nid: "+str(nid)+", number of points: "+str(num_pt)+", time span: "+str(time_span))
                    
                node_json.append({"NID"+str(nid):{'time span':str(time_span),'number of pts':str(num_pt)}})

#            nidStep=getSteps(url, nid,analysis_date.strftime("%Y-%m-%d"))
#            logging.info("NID: %s" +str(nid)+" has steps: "+ str(nidStep))
#            steps=nidStep['steps']
#            if not 'success' in resp:   #paradoxically success is only returned if = false
#                logging.info("SUCCESS in getting analysis data for device: %s" % str(nid)+" with response: "+str(resp))
#                amnum=len(resp['am_mode'])
#                pmnum=len(resp['pm_mode'])
#                amdist=sum(resp['am_distance'])
#                pmdist=sum(resp['pm_distance'])
#                co2=resp['travel_co2']
#                timeout=resp['outdoor_time']
#                target_stats.write(str(nid)+','+ str(amnum) + ',' + str(pmnum) + ',' + str(amdist) + ',' + str(pmdist) + ',' + str(timeout) + ',' + str(co2)+ ',' + str(steps))
#                target_stats.write("\n")
#            else:
#                logging.info("FAILED in getting analysis data for device: %s" % str(nid))
#                target_stats.write(str(nid)+',NaN,NaN,NaN,NaN,NaN,NaN,'+ str(steps))
#                target_stats.write("\n")
#
#            if record_modes:
#                node_json.append({"NID"+str(nid):{'steps':nidStep,'analysis':resp,'modes':list(modes),'timestamp':list(timestamp),'latitude':list(lat),'longitude':list(lon)}})
#            else:
#                node_json.append({"NID"+str(nid):{'steps':nidStep,'analysis':resp}})

#            node_json.append({"NID"+str(nid):{'steps':nidStep,'modes':list(modes),'timestamp':list(timestamp),'latitude':list(lat),'longitude':list(lon)}})

#        json_out.append({"DATE"+analysis_date.strftime("%Y%m%d"):{'NationalStats': resp_nat,'Nodes':node_json}})
        json_out.append({"DATE"+analysis_date.strftime("%Y%m%d"):{'Valid nodes': list(long_nid),'Number of valid nodes':num_valid_nid,'Short nodes':list(short_nid),'Number of short nodes':num_short_nid,'Nodes':node_json}}) #
        

    target.write(json.dumps(json_out))
    target.close()
if __name__ == "__main__":
    from docopt import docopt
    # parse arguments
    arguments = docopt(__doc__)

    # deviceIDs is a mandatory option
    device_file = arguments['--deviceIDs']
    start_date = arguments['--start_date'] if arguments['--start_date'] else datetime.datetime.strptime(datetime.date.today().strftime("%Y-%m-%d"), "%Y-%m-%d") - datetime.timedelta(days=5)
    end_date = arguments['--end_date'] if arguments['--end_date'] else datetime.datetime.strptime(datetime.date.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
    main(arguments['URL'], device_file, start_date,end_date)