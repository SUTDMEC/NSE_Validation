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
	for i in range(len(dateList)-1):
		overview, geojson = main("https://data.nse.sg","nodeFile.csv",dateList[i+1],testing = True)
		print overview
		#print geojson
		#process the home location and school location
		locationFile.write(str(overview['Home'][(NodeID, dateList[i])])+"\n")
		locationFile.write(str(overview['School'][(NodeID, dateList[i])])+"\n")
		home_loc.append(overview['Home'][(NodeID, dateList[i])])
		school_loc.append(overview['School'][(NodeID, dateList[i])])
		center_loc.append(((Filter(home_loc[-1][0])+Filter(school_loc[-1][0]))/2,(Filter(home_loc[-1][1])+Filter(school_loc[-1][1]))/2))
		#process the geojson
		geoItem = geoProcess(geojson,home_loc[-1],school_loc[-1])
		geoList.append(geoItem)

		#get the main mode
		AMMode = getMainModeOfAM(overview)
		PMMode = getMainModeOfPM(overview)
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
	return render_template('root.html',
		home_loc = home_loc, 
		school_loc = school_loc, 
		center_loc = center_loc, 
		geoList = geoList, 
		amMode = AMModeList, 
		pmMode = PMModeList, 
		amColor = AMColorList, 
		pmColor = PMColorList,
		date = dateList,
		length = len(dateList),
		mapName = mapName,
		mappreName = mappreName,
		maptruthName = maptruthName
	)

if __name__=="__main__":
	app.run(debug=True)
