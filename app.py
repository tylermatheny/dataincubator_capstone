from flask import Flask, render_template, url_for,request, flash, redirect
import pandas as pd
import statsmodels.api as sm
import matplotlib as plt
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import Range1d
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
from bokeh.models import LinearColorMapper
from bokeh.models import ColumnDataSource
from bokeh.models import Axis
from bokeh.models import NumeralTickFormatter
from colorcet import rainbow as palette
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from scipy import stats
from bokeh.embed import components
import numpy as np
import os


app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////Users/tyler_matheny/studentloans/flask-project/loan.db' 
# db = SQLAlchemy(app)

# class LoanTable(db.Model):
# 	__tablename__ = 'loan'
# 	name = db.Column('name', db.String, primary_key = True)
# 	test_score = db.Column('test_score', db.Integer)
# 	school = db.Column('school', db.String)
# 	major = db.Column('major', db.String)
# 	family_income = db.Column('family_income', db.Integer)
@app.route('/index')
def ShowMachineLearningSATPlot2():
	scorecard_data = pd.read_csv('MERGED2016_17_PP.csv',low_memory = False)
	machine_learning_data = scorecard_data[['SAT_AVG','MD_FAMINC','RPY_3YR_RT']]
	machine_learning_data = machine_learning_data.apply(pd.to_numeric, errors='coerce')
	machine_learning_data = machine_learning_data.dropna()
	X_loan = machine_learning_data[[ 'SAT_AVG','MD_FAMINC']]
	y_loan = machine_learning_data['RPY_3YR_RT']
	min_max_scaler = preprocessing.MinMaxScaler()
	poly_scaler = PolynomialFeatures(degree=3)
	X_loan_poly = poly_scaler.fit_transform(X_loan)
	X_train, X_test, y_train, y_test = train_test_split(X_loan_poly, y_loan, random_state = 0)
	linreg = Ridge().fit(X_train, y_train)
	model = Ridge()
	model.fit(X_train,y_train)
	predictions = model.predict(X_train)
	real = y_train
	lst = [] 
	for x in range(700,1550, 20):
	    for y in range(20000,80000, 1000):
	        example = [[x, y]]
	        example_scaled = poly_scaler.fit_transform(example)
	        probability = model.predict(example_scaled)
	        lst.append((x, y, probability[0]))
	data = pd.DataFrame(lst, columns = ['SAT', 'FAMILY_INCOME', 'PROB'])
	output_file("toolbar.html")
	palette.reverse()
	source = ColumnDataSource(data)
	exp_cmap = LinearColorMapper(palette,
	                             low = .3, 
	                             high = .9)


	# mapper = linear_cmap(field_name= data['RPY_3YR_RT'], low = .2, high = ., palette=Spectral6 )
	TOOLTIPS = [
	    ("repayment rate", '@PROB'),
	]


	p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
	           title="Mouse over the chart to see repayment scores at various scores and incomes")

	p.square('FAMILY_INCOME', 'SAT',  source=source, size = 13, line_color=None,fill_color={"field":"PROB", "transform":exp_cmap})


	color_bar = ColorBar(color_mapper=exp_cmap, label_standoff=12, border_line_color=None, location=(0,0))


	p.add_layout(color_bar, 'right')

	p.xaxis.axis_label = "Family Income (dollars)"

	p.xaxis.axis_label_text_font_style = 'bold'
	p.xaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_style = 'bold'
	p.xaxis.major_label_text_font_size = '15pt'
	p.yaxis.major_label_text_font_size = '15pt'
	p.yaxis.axis_label = 'SAT score'
	p.xaxis[0].formatter = NumeralTickFormatter(format="0.00")
	# show(p)
	script, div = components(p)
	scorecard_data = pd.read_csv('MERGED2016_17_PP.csv',low_memory = False)	
	machine_learning_data = scorecard_data[['ACTCMMID','MD_FAMINC','RPY_3YR_RT']]
	machine_learning_data = machine_learning_data[machine_learning_data.ACTCMMID > 16]
	machine_learning_data = machine_learning_data.apply(pd.to_numeric, errors='coerce')
	machine_learning_data = machine_learning_data.dropna()
	X_loan = machine_learning_data[[ 'ACTCMMID','MD_FAMINC']]
	y_loan = machine_learning_data['RPY_3YR_RT']
	min_max_scaler = preprocessing.MinMaxScaler()
	poly_scaler = PolynomialFeatures(degree=3)
	X_loan_poly = poly_scaler.fit_transform(X_loan)
	X_train, X_test, y_train, y_test = train_test_split(X_loan_poly, y_loan, random_state = 0)
	model = Ridge()
	model.fit(X_train,y_train)
	lst = [] 
	for x in np.arange(15.0,36.0, .5):
	    for y in np.arange(20000.0,80000.0, 1000.0):
	        example = [[x, y]]
	        example_scaled = poly_scaler.fit_transform(example)
	        probability = model.predict(example_scaled)
	        lst.append((x, y, probability[0]))
	dataACT = pd.DataFrame(lst, columns = ['ACT', 'FAMILY_INCOME', 'PROB'])
	output_file("toolbar.html")
	source = ColumnDataSource(dataACT)
	exp_cmap = LinearColorMapper(palette,
	                             low = .3, 
	                             high = .9)


	# mapper = linear_cmap(field_name= data['RPY_3YR_RT'], low = .2, high = ., palette=Spectral6 )
	TOOLTIPS = [
	    ("repayment rate", '@PROB'),
	]


	p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
	           title="Mouse over the chart to see repayment scores at various scores and incomes")

	p.square('FAMILY_INCOME', 'ACT',  source=source, size = 13, line_color=None,fill_color={"field":"PROB", "transform":exp_cmap})


	color_bar = ColorBar(color_mapper=exp_cmap, label_standoff=12, border_line_color=None, location=(0,0))


	p.add_layout(color_bar, 'right')

	p.xaxis.axis_label = "Family Income (dollars)"

	p.xaxis.axis_label_text_font_style = 'bold'
	p.xaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_style = 'bold'
	p.xaxis.major_label_text_font_size = '15pt'
	p.yaxis.major_label_text_font_size = '15pt'
	p.yaxis.axis_label = 'ACT score'
	p.xaxis[0].formatter = NumeralTickFormatter(format="0.00")
	# show(p)
	script2, div2 = components(p)

	data = scorecard_data[['INSTNM','MD_FAMINC', 'RPY_3YR_RT', 'SAT_AVG']]
	data = data.set_index('INSTNM')
	data = data[['MD_FAMINC', 'RPY_3YR_RT', 'SAT_AVG']].apply(pd.to_numeric, errors='coerce')
	data = data.dropna()



	output_file("toolbar.html")

	source = ColumnDataSource(data)

	exp_cmap = LinearColorMapper(palette, 
	                             low = .3, 
	                             high = .9)

	# mapper = linear_cmap(field_name= data['RPY_3YR_RT'], low = .2, high = ., palette=Spectral6 )

	TOOLTIPS = [
	    ("college", '@INSTNM'),
	]

	p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
	           title="Mouse over the dots to see different universities")

	p.circle('MD_FAMINC', 'SAT_AVG',  source=source, size = 10, line_color=None,fill_color={"field":"RPY_3YR_RT", "transform":exp_cmap})
	p.x_range=Range1d(0, 120000)

	color_bar = ColorBar(color_mapper=exp_cmap, label_standoff=12, border_line_color=None, location=(0,0))


	p.add_layout(color_bar, 'right')

	p.xaxis.axis_label = "Median Family Income (dollars)"
	p.yaxis.axis_label = 'SAT score'
	p.xaxis[0].formatter = NumeralTickFormatter(format="0.00")
	p.xaxis.axis_label_text_font_style = 'bold'
	p.xaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_style = 'bold'
	p.xaxis.major_label_text_font_size = '15pt'
	p.yaxis.major_label_text_font_size = '15pt'
	p.yaxis.axis_label = 'Median SAT score'

	script3, div3 = components(p)

	data = scorecard_data[['INSTNM','MD_FAMINC', 'RPY_3YR_RT', 'ACTCMMID']]
	data = data.set_index('INSTNM')
	data = data[['MD_FAMINC', 'RPY_3YR_RT', 'ACTCMMID']].apply(pd.to_numeric, errors='coerce')
	data = data[data.ACTCMMID >= 15]
	data = data.dropna()



	output_file("toolbar.html")

	source = ColumnDataSource(data)

	exp_cmap = LinearColorMapper(palette, 
	                             low = .3, 
	                             high = .9)

	# mapper = linear_cmap(field_name= data['RPY_3YR_RT'], low = .2, high = ., palette=Spectral6 )

	TOOLTIPS = [
	    ("college", '@INSTNM'),
	]

	p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
	           title="Mouse over the dots to see different universities")

	p.circle('MD_FAMINC', 'ACTCMMID',  source=source, size = 10, line_color=None,fill_color={"field":"RPY_3YR_RT", "transform":exp_cmap})
	p.x_range=Range1d(0, 120000)

	color_bar = ColorBar(color_mapper=exp_cmap, label_standoff=12, border_line_color=None, location=(0,0))


	p.add_layout(color_bar, 'right')

	p.xaxis.axis_label = "Median Family Income (dollars)"
	p.yaxis.axis_label = 'Median ACT score'
	p.xaxis[0].formatter = NumeralTickFormatter(format="0.00")
	p.xaxis.axis_label_text_font_style = 'bold'
	p.xaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_style = 'bold'
	p.xaxis.major_label_text_font_size = '15pt'
	p.yaxis.major_label_text_font_size = '15pt'


	script4, div4 = components(p)
	return render_template('index2.html', script = script, div =div, script2 = script2, div2 = div2, script3 = script3, div3 = div3, script4 = script4, div4 = div4)


@app.route('/', methods = ['GET', 'POST'])
def ShowMachineLearningSATPlot():
	scorecard_data = pd.read_csv('MERGED2016_17_PP.csv',low_memory = False)
	machine_learning_data = scorecard_data[['SAT_AVG','MD_FAMINC','RPY_3YR_RT']]
	machine_learning_data = machine_learning_data.apply(pd.to_numeric, errors='coerce')
	machine_learning_data = machine_learning_data.dropna()
	X_loan = machine_learning_data[[ 'SAT_AVG','MD_FAMINC']]
	y_loan = machine_learning_data['RPY_3YR_RT']
	min_max_scaler = preprocessing.MinMaxScaler()
	poly_scaler = PolynomialFeatures(degree=3)
	X_loan_poly = poly_scaler.fit_transform(X_loan)
	X_train, X_test, y_train, y_test = train_test_split(X_loan_poly, y_loan, random_state = 0)
	linreg = Ridge().fit(X_train, y_train)
	model = Ridge()
	model.fit(X_train,y_train)
	predictions = model.predict(X_train)
	real = y_train
	lst = [] 
	for x in range(700,1550, 20):
	    for y in range(20000,80000, 1000):
	        example = [[x, y]]
	        example_scaled = poly_scaler.fit_transform(example)
	        probability = model.predict(example_scaled)
	        lst.append((x, y, probability[0]))
	data = pd.DataFrame(lst, columns = ['SAT', 'FAMILY_INCOME', 'PROB'])
	output_file("toolbar.html")
	palette.reverse()
	source = ColumnDataSource(data)
	exp_cmap = LinearColorMapper(palette,
	                             low = .3, 
	                             high = .9)


	# mapper = linear_cmap(field_name= data['RPY_3YR_RT'], low = .2, high = ., palette=Spectral6 )
	TOOLTIPS = [
	    ("repayment rate", '@PROB'),
	]


	p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
	           title="Mouse over the chart to see repayment scores at various scores and incomes")

	p.square('FAMILY_INCOME', 'SAT',  source=source, size = 13, line_color=None,fill_color={"field":"PROB", "transform":exp_cmap})


	color_bar = ColorBar(color_mapper=exp_cmap, label_standoff=12, border_line_color=None, location=(0,0))


	p.add_layout(color_bar, 'right')

	p.xaxis.axis_label = "Family Income (dollars)"

	p.xaxis.axis_label_text_font_style = 'bold'
	p.xaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_style = 'bold'
	p.xaxis.major_label_text_font_size = '15pt'
	p.yaxis.major_label_text_font_size = '15pt'
	p.yaxis.axis_label = 'SAT score'
	p.xaxis[0].formatter = NumeralTickFormatter(format="0.00")
	# show(p)
	script, div = components(p)
	scorecard_data = pd.read_csv('MERGED2016_17_PP.csv',low_memory = False)	
	machine_learning_data = scorecard_data[['ACTCMMID','MD_FAMINC','RPY_3YR_RT']]
	machine_learning_data = machine_learning_data[machine_learning_data.ACTCMMID > 16]
	machine_learning_data = machine_learning_data.apply(pd.to_numeric, errors='coerce')
	machine_learning_data = machine_learning_data.dropna()
	X_loan = machine_learning_data[[ 'ACTCMMID','MD_FAMINC']]
	y_loan = machine_learning_data['RPY_3YR_RT']
	min_max_scaler = preprocessing.MinMaxScaler()
	poly_scaler = PolynomialFeatures(degree=3)
	X_loan_poly = poly_scaler.fit_transform(X_loan)
	X_train, X_test, y_train, y_test = train_test_split(X_loan_poly, y_loan, random_state = 0)
	model = Ridge()
	model.fit(X_train,y_train)
	lst = [] 
	for x in np.arange(15.0,36.0, .5):
	    for y in np.arange(20000.0,80000.0, 1000.0):
	        example = [[x, y]]
	        example_scaled = poly_scaler.fit_transform(example)
	        probability = model.predict(example_scaled)
	        lst.append((x, y, probability[0]))
	dataACT = pd.DataFrame(lst, columns = ['ACT', 'FAMILY_INCOME', 'PROB'])
	output_file("toolbar.html")
	source = ColumnDataSource(dataACT)
	exp_cmap = LinearColorMapper(palette,
	                             low = .3, 
	                             high = .9)


	# mapper = linear_cmap(field_name= data['RPY_3YR_RT'], low = .2, high = ., palette=Spectral6 )
	TOOLTIPS = [
	    ("repayment rate", '@PROB'),
	]


	p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
	           title="Mouse over the chart to see repayment scores at various scores and incomes")

	p.square('FAMILY_INCOME', 'ACT',  source=source, size = 13, line_color=None,fill_color={"field":"PROB", "transform":exp_cmap})


	color_bar = ColorBar(color_mapper=exp_cmap, label_standoff=12, border_line_color=None, location=(0,0))


	p.add_layout(color_bar, 'right')

	p.xaxis.axis_label = "Family Income (dollars)"

	p.xaxis.axis_label_text_font_style = 'bold'
	p.xaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_style = 'bold'
	p.xaxis.major_label_text_font_size = '15pt'
	p.yaxis.major_label_text_font_size = '15pt'
	p.yaxis.axis_label = 'ACT score'
	p.xaxis[0].formatter = NumeralTickFormatter(format="0.00")
	# show(p)
	script2, div2 = components(p)

	data = scorecard_data[['INSTNM','MD_FAMINC', 'RPY_3YR_RT', 'SAT_AVG']]
	data = data.set_index('INSTNM')
	data = data[['MD_FAMINC', 'RPY_3YR_RT', 'SAT_AVG']].apply(pd.to_numeric, errors='coerce')
	data = data.dropna()



	output_file("toolbar.html")

	source = ColumnDataSource(data)

	exp_cmap = LinearColorMapper(palette, 
	                             low = .3, 
	                             high = .9)

	# mapper = linear_cmap(field_name= data['RPY_3YR_RT'], low = .2, high = ., palette=Spectral6 )

	TOOLTIPS = [
	    ("college", '@INSTNM'),
	]

	p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
	           title="Mouse over the dots to see different universities")

	p.circle('MD_FAMINC', 'SAT_AVG',  source=source, size = 10, line_color=None,fill_color={"field":"RPY_3YR_RT", "transform":exp_cmap})
	p.x_range=Range1d(0, 120000)

	color_bar = ColorBar(color_mapper=exp_cmap, label_standoff=12, border_line_color=None, location=(0,0))


	p.add_layout(color_bar, 'right')

	p.xaxis.axis_label = "Median Family Income (dollars)"
	p.yaxis.axis_label = 'SAT score'
	p.xaxis[0].formatter = NumeralTickFormatter(format="0.00")
	p.xaxis.axis_label_text_font_style = 'bold'
	p.xaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_style = 'bold'
	p.xaxis.major_label_text_font_size = '15pt'
	p.yaxis.major_label_text_font_size = '15pt'
	p.yaxis.axis_label = 'Median SAT score'

	script3, div3 = components(p)

	data = scorecard_data[['INSTNM','MD_FAMINC', 'RPY_3YR_RT', 'ACTCMMID']]
	data = data.set_index('INSTNM')
	data = data[['MD_FAMINC', 'RPY_3YR_RT', 'ACTCMMID']].apply(pd.to_numeric, errors='coerce')
	data = data[data.ACTCMMID >= 15]
	data = data.dropna()



	output_file("toolbar.html")

	source = ColumnDataSource(data)

	exp_cmap = LinearColorMapper(palette, 
	                             low = .3, 
	                             high = .9)

	# mapper = linear_cmap(field_name= data['RPY_3YR_RT'], low = .2, high = ., palette=Spectral6 )

	TOOLTIPS = [
	    ("college", '@INSTNM'),
	]

	p = figure(plot_width=800, plot_height=600, tooltips=TOOLTIPS,
	           title="Mouse over the dots to see different universities")

	p.circle('MD_FAMINC', 'ACTCMMID',  source=source, size = 10, line_color=None,fill_color={"field":"RPY_3YR_RT", "transform":exp_cmap})
	p.x_range=Range1d(0, 120000)

	color_bar = ColorBar(color_mapper=exp_cmap, label_standoff=12, border_line_color=None, location=(0,0))


	p.add_layout(color_bar, 'right')

	p.xaxis.axis_label = "Median Family Income (dollars)"
	p.yaxis.axis_label = 'Median ACT score'
	p.xaxis[0].formatter = NumeralTickFormatter(format="0.00")
	p.xaxis.axis_label_text_font_style = 'bold'
	p.xaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_size = '20pt'
	p.yaxis.axis_label_text_font_style = 'bold'
	p.xaxis.major_label_text_font_size = '15pt'
	p.yaxis.major_label_text_font_size = '15pt'


	script4, div4 = components(p)
	return render_template('input.html', script = script, div =div, script2 = script2, div2 = div2, script3 = script3, div3 = div3, script4 = script4, div4 = div4)



# def add_data():
# 	name = request.form['name']
# 	test_score = request.form['test_score']
# 	school = request.form['school']
# 	major = request.form['major']
# 	family_income = request.form['family_income']
# 	post = Exmaple(name = name, test_score = test_score, school = school, major = major, family_income = family_income, date_posted = datetime.now())
# 	db.session.add(post)
# 	db.session.commit
	




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # app.run(host='0.0.0.0', port=port)
    app.run(debug = True)