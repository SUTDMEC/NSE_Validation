from flask import *
from forms import LoginForm
from NSE_Analytics_pilot_testing.process import *
from geoProcess import *

app = Flask(__name__)
app.config.from_object('config')

@app.route('/', methods = ['GET','POST'])
@app.route('/query', methods = ['GET', 'POST'])
def root():
	form = LoginForm()
	if form.validate_on_submit():
		flash('Login requested for OpenID="' + form.openid.data )
		return redirect("/visualization/"+str(form.openid.data))
	return render_template("query.html",form = form)

@app.route('/visualization/<int:NodeID>')
def visualization(NodeID):
	nodeFile = open("nodeFile.csv","w")
	dateList = ["2015-09-28","2015-09-29","2015-09-30","2015-10-01","2015-10-02"]
	nodeFile.write(str(NodeID))
	nodeFile.close()
	locationFile = open("location","w")
	home_loc = []
	school_loc = []
	center_loc = []
	geoList = []
	AMModeList = []
	PMModeList = []
	AMColorList = []
	PMColorList = []
	mapName = []
	mappreName = []
	maptruthName = []
	mark = [True, True, True, True]
	for i in range(len(dateList)-1):
		overview, geojson = main("https://data.nse.sg","nodeFile.csv",dateList[i+1],testing = True)
		print overview
		#print geojson
		#process the home location and school location
		if (None,None) == (overview['Home'][(NodeID, dateList[i])]):
			mark[i] = False
		if (None,None) == (overview['School'][(NodeID, dateList[i])]):
			mark[i] = False
		home_loc.append(overview['Home'][(NodeID, dateList[i])])
		school_loc.append(overview['School'][(NodeID, dateList[i])])
		center_loc.append(((Filter(home_loc[-1][0])+Filter(school_loc[-1][0]))/2,(Filter(home_loc[-1][1])+Filter(school_loc[-1][1]))/2))
		#process the geojson
		geoItem = geoProcess(geojson,home_loc[-1],school_loc[-1])
		geoList.append(geoItem)

		#get the main mode
		AMMode = getMainModeOfAM(overview)
		if AMMode == "Loss":
			mark[i] = False
		PMMode = getMainModeOfPM(overview)
		if PMMode == "Loss":
			mark[i] = False
		AMModeList.append(AMMode)
		PMModeList.append(PMMode)
		AMColor = ""
		PMColor = ""
		if AMMode == "Train":
			AMColor = "Blue"
		if AMMode == "Bus":
			AMColor = "red"
		if AMMode == "Car":
			AMColor = "green"
		if PMMode == "Train":
			PMColor = "Blue"
		if PMMode == "Bus":
			PMColor = "red"
		if PMMode == "Car":
			PMColor = "green"
		AMColorList.append(AMColor)
		PMColorList.append(PMColor)
		mapName.append("map"+str(i+1))
		mappreName.append("map"+str(i+1)+"pre")
		maptruthName.append("map"+str(i+1)+"truth")

	home_loc_r = []
	school_loc_r = []
	center_loc_r = []
	geoList_r = []
	AMModeList_r = []
	PMModeList_r = []
	AMColorList_r = []
	PMColorList_r = []
	mapName_r = []
	mappreName_r = []
	maptruthName_r = []
	dateList_r = ["2015-09-28"]
	countMap = 0
	for i in range(len(dateList)-1):
		if mark[i]==True:
			countMap += 1
			home_loc_r.append(home_loc[i])
			school_loc_r.append(school_loc[i])
			center_loc_r.append(center_loc[i])
			geoList_r.append(geoList[i])
			AMModeList_r.append(AMModeList[i])
			PMModeList_r.append(PMModeList[i])
			AMColorList_r.append(AMColorList[i])
			PMColorList_r.append(PMColorList[i])
			mapName_r.append("map"+str(countMap))
			mappreName_r.append("map"+str(countMap)+"pre")
			maptruthName_r.append("map"+str(countMap)+"truth")
			dateList_r.append(dateList[i+1])
	return render_template('root.html',
		home_loc = home_loc_r, 
		school_loc = school_loc_r, 
		center_loc = center_loc_r, 
		geoList = geoList_r, 
		amMode = AMModeList_r, 
		pmMode = PMModeList_r, 
		amColor = AMColorList_r, 
		pmColor = PMColorList_r,
		date = dateList_r,
		length = len(dateList_r),
		mapName = mapName_r,
		mappreName = mappreName_r,
		maptruthName = maptruthName_r
	)

if __name__=="__main__":
	app.run(debug=True)
