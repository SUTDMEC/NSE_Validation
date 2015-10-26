# NSE_Analytics
Repository for developing the NSE analytics code


Dependencies
Python 3.0 > version >= 2.7

Python packages
pandas
numpy
scikit-learn
docopt


Example curl calls to test REST API, for synthetic data

curl -v -L -G -d "nid=200001&table=10_bus_6sept15" http://54.251.119.96:3000/getgroundtruth

curl -v -L -G -d "nid=200001&start=1439806470&end=1439848951&table=10_bus_6sept15" http://54.251.119.96:3000/getdata

curl -v -L -G -d "table=10_bus_6sept15" http://54.251.119.96:3000/getdevices

Example curl calls to test REST API, for pilot data (without ground truth): (NOTE: ONLY HOME/SCHOOL LOCATIONS ARE REAL, all other data is synthetic as well in this data set)

curl -v -L -G -d "nid=200188&table=pilot2&start=14361682&end=1436443875" http://54.251.119.96:3000/getdata

curl -v -L -G -d "table=pilot2" http://54.251.119.96:3000/getdevices

curl -v -L -G -d "nid=200188&table=pilot2" http://54.251.119.96:3000/getgroundtruth
