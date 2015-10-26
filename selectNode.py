# -*-coding: utf-8 -*-
from NSE_Analytics_pilot_testing.process import *
import re

def check(overview,line,day):
	i = 0
	if tuple([int(line), day]) in overview['Home'] and tuple([int(line), day]) in overview['School']:
		i += 1
	else:
		print "No Home or School"
	regAm_mode = re.compile(r"'am_mode':\s\[.*?\],")
	matchMode = regAm_mode.search(str(overview))
	if matchMode!=None:
		if len(matchMode.group())!=12:
			i += 1
		else:
			print matchMode.group(),len(matchMode.group())
	regPm_mode = re.compile(r"'pm_mode':\s\[.*?\]")
	matchMode = regPm_mode.search(str(overview))
	if matchMode!=None:
		if len(matchMode.group())!=12:
			i += 1
		else:
			print matchMode.group(),len(matchMode.group())
	if i == 3:
		return True
	else:
		return False

if __name__=="__main__":
	dateList = ["2015-09-28","2015-09-29","2015-09-30","2015-10-01","2015-10-02"]
	fr = open("all_Node","r")
	fnode = open("Selected_Node","a")
	#nodelist = ["510132"]
	TagFile = open("TagFile","r")
	Tag = TagFile.readline()
	print Tag
	Tag = int(Tag)
	TagFile.close()
	count = 0
	for line in fr.readlines():
		if count < Tag:
			count += 1
			continue
 		fw = open("allNodeFile.csv","w")
		fw.write(str(line))
		fw.close()
		print "Node: ",line
		i = 0 
		for day in range(len(dateList)-1):
			print dateList[i+1]
			overview, geojson = main("https://data.nse.sg","allNodeFile.csv",dateList[i+1],testing = True)
			print overview
			if check(overview,line,dateList[i]):
				i += 1
		if i==4 :
			fnode.write(str(line)+"\n")
		print ""
		count += 1
		TagFile = open("TagFile","w")
		TagFile.write(str(count))
		TagFile.close()


		