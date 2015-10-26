"""Clears device analysis status for a given device list and date.

Usage: analysis_clear.py --deviceIDs=DEVICEFILE [--current_date=DATE] URL

Arguments:
 URL            Base URL for the backend API, for example 'http://sensg.ddns.net/api/'

Options:
 --deviceIDs=DEVICEFILE    mandatory option with the filename with device IDs. Format is one ID per line.

 --current_date=DATE       option to specify today's date for test purposes (%Y-%m-%d), otherwise the local server time is used to determine the date. The data to be processed is 2 days before today's date

"""


import datetime
import logging
import sys
import process

def main(url, device_file, current_date =
         datetime.date.today().strftime("%Y-%m-%d")):
    """Main processing loop. It expects a base url and the file name of
    the csv file of device ids. If testing is True the function
    returns detailed information for debugging. If the current_date
    string (%Y-%m-%d) is provided this is taken as the current date
    when processing is done, otherwise use the localtime to determine
    the date.

    """
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
        # determine which date to process data for which is two days before the current date
    analysis_date_tuple = datetime.datetime.strptime(current_date, "%Y-%m-%d") - datetime.timedelta(days=2)
    analysis_date =  analysis_date_tuple.strftime("%Y-%m-%d")
    #reset the analysis status of the devices to 0
    for nid in device_ids:
        logging.info("Clearing data for: %s" % analysis_date)
        if process.setStatus(url, nid, analysis_date, 0):
            logging.info("SUCCESS in resetting process status for device: %s" % nid)
        else:
            logging.info("FAIL in resetting process status for device: %s" % nid)

if __name__ == "__main__":
    from docopt import docopt
    # parse arguments
    arguments = docopt(__doc__)

    # deviceIDs is a mandatory option
    device_file = arguments['--deviceIDs']
    current_date = arguments['--current_date'] if arguments['--current_date'] else datetime.date.today().strftime("%Y-%m-%d")

    main(arguments['URL'], device_file, current_date=current_date)