# -*-coding: utf-8 -*-
from NSE_Analytics_pilot_testing.process import *
import re

def dist(u,v):
	if abs(Filter(u[0])-Filter(v[0]))+abs(Filter(u[1])-Filter(v[1]))>0.001:
		return True
	else:
		return False

def geoProcess(geojson, home_loc, school_loc):
	geoList = []
	regPoint = re.compile(r'\[.*?\]')
	Num = 0
	for item in geojson:
		points = regPoint.findall(item)
		for term in points:
			term = map(float,term.strip("[]").split(","))
			if Num>0:
				if (abs(term[0]-geoList[Num-1][1])+abs(term[1]-geoList[Num-1][0]))>0.001:
					geoList.append(tuple([term[1],term[0]]))
					Num += 1
			else:
				geoList.append(tuple([term[1],term[0]]))
				Num += 1
	#geoRaw = list(set(geoList))
	geoRaw = geoList;
	geoList = []
	for item in geoRaw:
		Count = 0
		for checker in geoList:
			if dist(item,checker):
				Count += 1
		if Count == len(geoList):
			geoList.append(item)
	mid = -1
	for i in range(len(geoList)):
		if not dist(geoList[i],home_loc):
			mid = i

	return [geoList[mid+1:-1],geoList[0:mid]]#AM, geoListPM


def getMainModeOfAM(data):
	regAm_mode = re.compile(r"'am_mode':\s\[.*?\],")
	matchMode = regAm_mode.search(str(data))
	ModeList = []
	try:
		ModeList = map(int,matchMode.group().strip("'am_mode': [").rstrip("],").split(','))
	except Exception:
		print "No travel mode!"
	if 4 in ModeList:
		return "Train"
	if 5 in ModeList:
		return "Bus"
	if 6 in ModeList:
		return "Car"
	return "Walk"
	#return matchMode.group()

def getMainModeOfPM(data):
	regPm_mode = re.compile(r"'pm_mode':\s\[.*?\]")
	matchMode = regPm_mode.search(str(data))
	ModeList = []
	print matchMode.group().strip("'pm_mode': [").rstrip("]").split(',')
	try:
		ModeList = map(int,matchMode.group().strip("'pm_mode': [").rstrip("]").split(','))
	except Exception:
		print "No travel mode!"
	if 4 in ModeList:
		return "Train"
	if 5 in ModeList:
		return "Bus"
	if 6 in ModeList:
		return "Car"
	return "Walk"
	
	#return matchMode.group()


def Filter(number):
	if number == None:
		return 0
	else:
		return number

if __name__=="__main__":
	overview, geojson = main("https://data.nse.sg","nodeFile.csv","2015-09-29",testing = True)
	#print geojson
	print overview
	home_loc = overview['Home'][(510132, "2015-09-28")]
	school_loc = overview['School'][(510132, "2015-09-28")]
	geoList = geoProcess(geojson,home_loc,school_loc)
	print "PM:"
	for item in geoList[0]:
		print item
	print "AM:"
	for item in geoList[1]:
		print item
	#print overview
	AMMode = getMainModeOfAM(overview)
	PMMode = getMainModeOfPM(overview)

	print AMMode,PMMode