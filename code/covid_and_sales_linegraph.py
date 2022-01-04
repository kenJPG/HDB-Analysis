# ====================================
#	Produces a line chart that allows users to see the number of resales over the years
# ====================================

# Import modules
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensuring that the current directory is correct
os.chdir(
	(os.path.abspath(os.path.dirname(__file__)))
)

# Reading files
resale_prices = np.genfromtxt('../datasets/resale_flat_prices_all.csv', dtype=[('month', 'U32'), ('town', 'U32'), ('floor_area_sqm', np.int32), ('price', np.int32)], skip_header=1, delimiter=',')

# =============
# 	Cleaning
# =============
# Change from monthly to quarterly data.
# This is done to allow for quarterly seasonal adjustment.

# Given a month, this function returns the quarter that month is in
def getQuarter(month):
	return 'Q' + str(((month - 1) // 3) + 1)


# Given an array of quarters and the corresponding data for that quarter,
# this function returns a seasonally adjusted version of that data
# NOTE for this function to work properly, Q1 to Q4 is REQUIRED for every year.
# There should be no missing quarter
def seasonalAdjustment(quarters_arr, data_arr, colname):

	dataTableShape = (int(len(data_arr)//4), 4)

	dataTable = (np.array(data_arr)).reshape(dataTableShape)

	# Calculate the sum of sales for each year
	yearMeans = np.mean(dataTable, axis=1)
	
	# Duplicate yearMeans across columns such that
	# every number in dataTable has a corresponding
	# year mean.

	# [a, b, c, d]

	# to

	# [
	# 	[a, a, a, a]
	# 	[b, b, b, b]
	# 	[c, c, c, c]
	# 	[d, d, d, d]
	# ]
	yearMeansFull = np.transpose(
		np.resize(yearMeans, (dataTableShape[1], dataTableShape[0]))
	)

	dataTableRatio = dataTable / yearMeansFull

	seasonal_indexes = np.mean(dataTableRatio, axis=0)

	# Calculate seasonally adjusted data using seasonal index
	seasAdjust_data_arr = []
	for i in range(len(data_arr)):

		# We get the number after 'Q' in quarter, and make it 0-indexed
		quarterIndex = int(quarters_arr[i].split('-Q')[-1]) - 1
		seasAdjust_data_arr.append(data_arr[i] / seasonal_indexes[quarterIndex])

	# Return the seasonally adjusted data
	return np.array(
		list(zip(quarters_arr, seasAdjust_data_arr)),
		dtype=[('quarters', 'U32'), (colname, np.int32)]
	)

# Change month column to quarter column
quarter_column = np.array(list(
	map(

		# Change from 2020-03 to 2020-Q1
		lambda x: x.split('-')[0] + '-' + getQuarter(int(x.split('-')[1])),
		[row['month'] for row in resale_prices]
	)
))


# Create a new data frame with only the relevant columns: quarter, town and price
clean_resale_prices = np.array(
	list(zip(quarter_column, resale_prices['town'], resale_prices['price'])),
	dtype=[('quarter', 'U32'), ('town', 'U32'), ('price', np.int32)]
)

# ----- Number of Resales by Quarter -----

# This dictionary contains the total number of sales as
# the value, for a given key of 'quarter'
resales_by_quarterDict = dict()

# Filling up the dictionary
for row in clean_resale_prices:
	if (resales_by_quarterDict.get(row['quarter'])):
		resales_by_quarterDict[row['quarter']] += 1
	else:
		resales_by_quarterDict[row['quarter']] = 1

resales_by_quarter = np.array(
	list(zip(np.unique(quarter_column), resales_by_quarterDict.values())),
	dtype=[('quarter', 'U32'), ('sales', np.int32)]
)


# ----- Number of Resales by Quarter By Town -----
# This 2d dictionary contains the total number of sales as
# the value, for a given key of 'quarter' and 'town'
resales_by_quarterTownDict = {
	singleQuarter: dict() for singleQuarter in clean_resale_prices['quarter']
}

# Filling up the dictionary 
for row in clean_resale_prices:
	if resales_by_quarterTownDict[row['quarter']].get(row['town']):
		resales_by_quarterTownDict[row['quarter']][row['town']] += 1
	else:
		resales_by_quarterTownDict[row['quarter']][row['town']] = 1

# This 2d list contains
# 	[<quarter>, <town>, <sales>]
# as its elements
resales_by_quarterTown = []

# Filling up resales_by_quarterTown
for quarter in resales_by_quarterTownDict.keys():
	for town in resales_by_quarterTownDict[quarter].keys():
		resales_by_quarterTown.append([quarter, town, resales_by_quarterTownDict[quarter][town]])

# ----- Seasonally Adjusted resales_by_quarter -----
adjusted_resales_by_quarter = seasonalAdjustment(resales_by_quarter['quarter'], resales_by_quarter['sales'], 'sales')


# ==========================
# 	Creating Figure
# ==========================
fig = plt.figure()
fig.suptitle(
	"How has Covid-19 affected resales?",
	verticalalignment='center',
	fontsize='xx-large'
)
fig.set_size_inches(10, 6)
fig.set_dpi(120)

colors = {
	'line_back': (142/255, 202/255, 230/255, 0.1),
	'line_color': (33/255, 158/255, 188/255, 1),
	'line_marker': (47/255, 114/255, 130/255, 1),
	'line_markeredge': (2/255, 48/255, 71/255, 1),
	'bar_color': '#006d77',
	'bar_edgecolor': '#14213d',
	'bar_bgcolor': '#83c5be',
	'green_indicator': (93/255, 176/255, 32/255),
	'red_indicator': (238/255, 76/255, 54/255)
}

# ==========================
# 	Creating subplots
# ==========================

linegraphSubplot = fig.subplots(2, 1, gridspec_kw={'height_ratios': [1.5, 1]})

# Note that height_ratio is slightly larger, this is to enlarge the padding between the two subplots
parameterAndBarSubplot = fig.subplots(2, 2, gridspec_kw={'height_ratios': [1.6, 1]})

# ==========================
# 	Creating linegraph
# ==========================

# This function is used to format the y-axis for the
# linegraph
def thousands(value, pos):
	return '{:1.0f}K'.format(value*1e-3)

# We use the first space of the 2x1 subplot
linegraphAxes = linegraphSubplot[0]

# Hide the other space we are not using
linegraphSubplot[1].set_visible(False)

# Plot the linegraph
linegraph, = linegraphAxes.plot(
	resales_by_quarter['sales'],
	color=colors['line_color']
)

# Linegraph marker format
linegraph.set_markerfacecolor(colors['line_marker'])
linegraph.set_markeredgecolor(colors['line_markeredge'])
linegraph.set_markersize(5)

# Formatting x-axis
linegraphAxes.set_xticks(ticks=[0, len(resales_by_quarter)-1], labels=[resales_by_quarter['quarter'][0], resales_by_quarter['quarter'][-1]])
linegraphAxes.set_ylim([0, 12000])
linegraphAxes.set_yticks(np.arange(0, 12001, 2000))

# Formatting y-axis
linegraphAxes.set_ylabel('Number of Resales')
linegraphAxes.yaxis.set_major_formatter(thousands)


# Removing borders
linegraphAxes.spines['top'].set_color((0,0,0,0))
linegraphAxes.spines['bottom'].set_color((0,0,0,0))
linegraphAxes.spines['left'].set_color((0,0,0,0))
linegraphAxes.spines['right'].set_color((0,0,0,0))

# Background color of linegraph
linegraphAxes.set_facecolor(colors['line_back'])

# This marker will appear when user hovers
# over a point
lineMarker, = linegraphAxes.plot(
	0,
	0,
	marker='X',
	markersize=5,
	markerfacecolor=colors['line_marker'],
	markeredgecolor=colors['line_markeredge']
)

# Hide the marker on load
lineMarker.set_visible(False)

# This marker will appear when user clicks
# over a point
clickedLineMarker, = linegraphAxes.plot(
	0,
	0,
	marker='o',
	markersize=5,
	markerfacecolor=colors['bar_bgcolor'],
	markeredgecolor='black'
)

# Hide the marker on load
clickedLineMarker.set_visible(False)

linegraphAnno = linegraphAxes.annotate(
	"",
	xy=(0, 0),
	xytext=(0, 15),
	textcoords='offset points',
	bbox={
		'boxstyle': 'round',
		'facecolor': 'w'
	},
	fontfamily='monospace'
)

linegraphPercentDiff = linegraphAxes.text(0, 0, "")

linegraphPercentDiff.set_visible(False)

# Hide the annotation on load
linegraphAnno.set_visible(False)

# ==========================
# 	Creating parameter
# ==========================
# Hide the first two empty charts
parameterAndBarSubplot[0][0].set_visible(False)
parameterAndBarSubplot[0][1].set_visible(False)

parameterAxes = parameterAndBarSubplot[1][0]
barAxes = parameterAndBarSubplot[1][1]

def lineContains(line, x, y):
	point_1, point_2 = line.get_xydata()
	if point_1[1]-0.1 <= y and y <= point_2[1] + 0.1:
		if point_1[0] <= x and x <= point_2[0]:
			return True
	return False

parameterLabels = ["Show Percent Diff on Hover", "Seasonally Adjust"]
parameter_x = [0.2, 0.2]
parameter_y = [0.3, 0.7]

# Determine the width for the parameter boxes
boxSize = 25

parameterBoxes = [
	parameterAxes.plot(
		[0.2, 0.8],
		np.full((2,1), parameter_y[i]),
		linewidth=boxSize,
		solid_capstyle='round',
		color = (0,0,0,0.05),
		label = parameterLabels[i]
	)
	for i in range(0, 2)
]

# The circles that indicate the currently selected
# aggregate
parameterMarkers = [
	parameterAxes.plot(
		parameter_x[i],
		parameter_y[i],
		label=parameterLabels[i],
		marker='o',
		color=colors['red_indicator'],
		linestyle='None'
	)
	for i in range(0,2)
]

# Display the aggregate labels
for text, x, y in zip(parameterLabels, parameter_x, parameter_y):
	parameterAxes.text(
		x + 0.08,
		y,
		text,
		verticalalignment='center',
		label=text
	)

# Hide borders
parameterAxes.spines['top'].set_color((0,0,0,0))
parameterAxes.spines['bottom'].set_color((0,0,0,0))
parameterAxes.spines['left'].set_color((0,0,0,0))
parameterAxes.spines['right'].set_color((0,0,0,0))

# Clear ticks
parameterAxes.set_xticks([])
parameterAxes.set_yticks([])

# This ensures that the limits are correct and that
# pyplot does NOT auto-adjust the limits.
parameterAxes.set_xlim(0, 1)
parameterAxes.set_ylim(0, 1)

# Setting title for parameters
parameterAxes.set_title(
	"Linegraph Parameters",
	fontdict={
		'fontfamily': 'monospace'
	}
)

# ==========================
# 	Creating horizontal bar chart
# ==========================

barchart = barAxes.barh(
	np.linspace(0.2, 0.8, 3),
	width=1,
	height=0.2,
	align='center',
	facecolor=colors['bar_color']
)

# Hovering annotation for barchart
barchartAnno = barAxes.annotate(
	"",
	xy=(0, 0),
	xytext=(0, 4),
	textcoords='offset points',
	bbox={
		'boxstyle': 'round',
		'facecolor': 'w'
	},
	fontfamily='monospace'
)

barAxes.set_facecolor(colors['bar_bgcolor'])

barAxes.set_title(
	"Click on a datapoint from the linegraph",
	fontdict={
		'fontfamily': 'monospace'
	}
)

# Barchart Border
barAxes.spines['top'].set_color((0,0,0,0))
barAxes.spines['bottom'].set_color((0,0,0,0))
barAxes.spines['right'].set_color((0,0,0,0))
barAxes.spines['left'].set(
	color = (0,0,0,1),
	linewidth = 4
)

barAxes.set_xticks([min(barAxes.get_xticks()), max(barAxes.get_xticks())])
barAxes.set_ylim(0, 1)
barAxes.set_yticks([])

# ==========================
# 	Adding interactivity
# ==========================

# 'cur' stores the information about the nearest point on the linegraph where
# the distance between the point and mouse is less than
# what is stored in the variable 'distanceLimit'
cur = []
# cur[0] stores the x coord of the point
# cur[1] stores the y coord of the point
# cur[2] stores the distance between the point and the mouse
# cur[3] stores the data point's index of all data points of the line

# This variable stores the distance needed
# between the mouse and a point, for the
# point to be considered 'hovered'.
distanceLimit = 10 

# Contains information on whether a parameter has been selected
parametersData = {
	'Seasonally Adjust': False,
	'Show Percent Diff on Hover': False
}

# Contains information on all the bar charts that are
# being displayed
barChartData = {
	'names': [],
	'sales': [],
	'position': []
}

# Given two pairs of coordinates, return the euclidean distance
# between them
def euclideanDistance(x1, y1, x2, y2):
	return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# This function takes an axes and a data point, returning
# the pixel coordinates of where that data point is, relative
# to the axes.
def toPixel(ax, xpoint, ypoint):

	fig = ax.get_figure()

	# Gets the box of the figure
	bbox = (ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()))

	# Get the height and width of that axes in pixels
	ax_width, ax_height = bbox.width * fig.dpi, bbox.height * fig.dpi

	# Get the data point limits of the x and y axis
	ax_xdatamin, ax_xdatamax = ax.get_xlim()
	ax_ydatamin, ax_ydatamax = ax.get_ylim()

	# Calculate the pixel coordinates
	pixel_x = ax_width * (1 - (ax_xdatamax - xpoint) / (ax_xdatamax - ax_xdatamin))
	pixel_y = ax_height * (1 - (ax_ydatamax - ypoint) / (ax_ydatamax - ax_ydatamin))

	return [pixel_x, pixel_y]


def updateBarChart():
	global barchart, barChartData

	# This function only runs when a click occurs on the linegraph
	# AND when that click is on some data point. As a result, we can access
	# the 'clicked' data point by using 'cur', despite 'cur' tracking the data point
	# that is currently the closest to the mouse.
	clicked_quarter = resales_by_quarter['quarter'][cur[3]]

	barAxes.set_title(f'Top 3 Towns in sales for {clicked_quarter}')

	# As we cannot directly update the barchart like we can a linegraph,
	# we will have to remove the barchart and reconstruct it
	barchart.remove()

	# Get all the town and sales for that quarter
	sort_arr = []
	for quarter, town, sales in resales_by_quarterTown:
		if (quarter == clicked_quarter):
			sort_arr.append([sales, town])

	# Sort in descending order, the total sales for a town.
	sort_arr.sort(reverse=True)

	# Change to np array for 2-dimensional indexing
	sort_arr = np.array(sort_arr)

	# Takes the top three towns and makes sure
	# that the values inside the list are integers
	barWidths = list(map(int, list(sort_arr[:3, 0])[::-1]))
	barHeights = [0.2] * 3
	barLabels = list(sort_arr[:3, 1])[::-1]

	barChartData['names'] = barLabels
	barChartData['sales'] = barWidths

	barchart = barAxes.barh(
		np.linspace(0.2, 0.8, 3),

		# Get the sales for the top three towns
		width=barWidths,
		height=barHeights,

		# Get the names of the top three towns and
		# set it as labels
		align='center',
		facecolor=colors['bar_color']
	)

	# Determine the x-axis limits. 
	# We get the range of the top three bars and divide
	# by 5. This will be used as 'padding'
	padding = (max(barWidths) - min(barWidths)) / 5
	min_xlim = max(0, min(barWidths) - padding)
	max_xlim = max(barWidths) + padding
	barAxes.set_xlim(min_xlim, max_xlim)

	# Setting ticks on the maximum and minimum of the bar
	barAxes.set_xticks([min(barWidths), max(barWidths)])

def updateClickedMarker():
	if (len(cur) > 0):
		clickedLineMarker.set_visible(True)
		clickedLineMarker.set_data(cur[0], cur[1])
	else:
		clickedLineMarker.set_visible(False)

def percentDiff(show = True):

	# If false is specified as a parameter, we will forcibly turn the
	# percent diff off.
	# This is useful when user moves their mouse out of the linegraph

	# Check the parameter is enabled
	if show and parametersData['Show Percent Diff on Hover']:
		linegraph_xydata = linegraph.get_xydata()
		for i, points in enumerate(linegraph_xydata):
			if (len(cur) > 0 and cur[0] == points[0] and cur[1] == points[1]):

				# Update position of the percent diff text
				linegraphPercentDiff.set_position([points[0], points[1]-1300])

				# The number of resales for the previous quarter.
				# If there is no previous quarter, take on the ydata
				# of the current quarter instead. This prevents
				# out of bound errors.
				prevYData = linegraph_xydata[max(0, i-1)][1]

				# Get the number of resales for the current quarter
				curYData = linegraph_xydata[i][1]


				# Calculate the percentage difference between the two
				calcPercentDifference = round(((curYData / prevYData) - 1) * 100, 2)

				# Formatting the calculation to make it appear nicer
				format_calcPercentDifference = str(abs(calcPercentDifference))
				if (calcPercentDifference < 0):
					format_calcPercentDifference = '▼ -' + format_calcPercentDifference + '%'
				else:
					format_calcPercentDifference = '▲ +' + format_calcPercentDifference + '%'

				# Displaying the results of the formatted calculation
				linegraphPercentDiff.set_text(
					format_calcPercentDifference
				)
				if (calcPercentDifference < 0):
					linegraphPercentDiff.set_color('red')
				else:
					linegraphPercentDiff.set_color('green')

				linegraphPercentDiff.set_visible(True)
	else:
		linegraphPercentDiff.set_visible(False)


def onclick(event):

	# If the click occurred in the linegraph axes
	if (event.inaxes == linegraphAxes):

		# A click is only valid when there is a point stored in cur
		# If cur has something stored inside, it means there is a point of 
		# data that the user is currently hovering over
		if (len(cur) > 0):
			updateClickedMarker()
			updateBarChart()

	# If the click occurred in the parameter axes 
	elif (event.inaxes == parameterAxes):
		global parametersData
		
		for box in parameterBoxes:

			# NOTE: 'box' is a list that contains 1 Line2D object.
			# We change it to that Line2D object
			box = box[0]
			if (lineContains(box, event.xdata, event.ydata)):
				parametersData[box.get_label()] = not parametersData[box.get_label()]

		for marker in parameterMarkers:
			marker = marker[0]
			if (parametersData[marker.get_label()]):

				# Add seasonal adjustment
				if (marker.get_label() == 'Seasonally Adjust'):
					updateClickedMarker()
					linegraph.set_ydata(adjusted_resales_by_quarter['sales'])

				# Enable percentage difference on hover
				elif (marker.get_label() == 'Show Percent Diff on Hover'):
					percentDiff()

				marker.set_color(colors['green_indicator'])
			else:

				# Remove seasonal adjustment
				if (marker.get_label() == 'Seasonally Adjust'):
					updateClickedMarker()
					linegraph.set_ydata(resales_by_quarter['sales'])

				# Disable percentage difference on hover
				elif (marker.get_label() == 'Show Percent Diff on Hover'):
					# NOTE we do not need to specify TRUE/FALSE as an argument.
					# This is because there we have already updated parametersData.
					# If there is no argument in percentDiff, it uses parameterData
					# as a default
					percentDiff()

				marker.set_color(colors['red_indicator'])
	fig.canvas.draw_idle()

def onmove(event):

	# If the mouse is NOT on the linegraph,
	# we will remove the point-hover effects
	if event.inaxes != linegraphAxes:
		lineMarker.set_visible(False)
		linegraphAnno.set_visible(False)

		# Force remove the percent difference text
		percentDiff(False)

	# If mouse is outside of the barchart,
	# remove the barchart annotation
	if (event.inaxes != barAxes):
		barchartAnno.set_visible(False)

	# If mouse is on the parameter axes
	if (event.inaxes == parameterAxes):

		# Darken the box we hover over
		for box in parameterBoxes:

			# NOTE: box is a list that contains 1 Line2D object.
			# We change it to that Line2D object it contains
			box = box[0]

			if (lineContains(box, event.xdata, event.ydata)):
				box.set_color((0,0,0,0.1))
			else:
				box.set_color((0,0,0,0.05))

	# If mouse is on the linegraph
	elif (event.inaxes == linegraphAxes):

		global cur

		# We need to clear cur for every move,
		# else the hover point will stay stuck until
		# the distance between mouse and a point becomes even
		# smaller than the currently stored closest distance, cur[2]
		cur = []

		for i, [point_x, point_y] in enumerate(linegraph.get_xydata()):

			# Transform data coordinates to pixel coordinates
			pixel_point_x, pixel_point_y = toPixel(linegraphAxes, point_x, point_y)

			event_x, event_y = toPixel(linegraphAxes, event.xdata, event.ydata)

			distanceMouseAndPoint = euclideanDistance(pixel_point_x, pixel_point_y, event_x, event_y)
			if (distanceMouseAndPoint <= distanceLimit):

				# If there is a point that is closer to the mouse, update the current point
				# to that closer point
				if (len(cur) == 0 or cur[2] > distanceMouseAndPoint):
					cur = point_x, point_y, distanceMouseAndPoint, i

		# If there is a point that meets the distance limit, we will consider
		# that point to be hovered
		if (len(cur) != 0):

			# Make both the hover marker and annotation appear
			lineMarker.set_visible(True)
			linegraphAnno.set_visible(True)

			# Update the x, y coords and the text for annotation
			lineMarker.set_data(cur[0], cur[1])

			linegraphAnno.set_text(
				f"{resales_by_quarter['quarter'][cur[3]]}\n"+

				(f"{adjusted_resales_by_quarter['sales'][cur[3]]} sales (Seasonally adjusted)\n"
				if parametersData['Seasonally Adjust']
				else
					f"{resales_by_quarter['sales'][cur[3]]} sales\n")
				+
				"Click to filter the bar graph"
			)


			# As cur[0] and cur[1] stores the x and y coordinates of the current
			# closest data point, we can directly assign the location of the annotation
			# as below
			linegraphAnno.xy = cur[0:2]


		percentDiff()

	# If mouse is on barchart
	elif (event.inaxes == barAxes):

		# Iterate through all bars
		for i, rectangle in enumerate(barchart):

			# Get the bottom-left-point and top-right-point of a bar
			bottomLeftPoint = rectangle.get_xy()
			topRightPoint = bottomLeftPoint[0] + rectangle.get_width(), bottomLeftPoint[1] + rectangle.get_height()

			# We use the 2 points from above to determine which bar the user is hovering over
			if (bottomLeftPoint[0] <= event.xdata and event.xdata <= topRightPoint[0]):
				if (bottomLeftPoint[1] <= event.ydata and event.ydata <= topRightPoint[1]):

					# We only continue if the barchart contains data from clicking on the linegraph
					# If this is removed, there will be an error when the user hovers over the barchart
					# upon load 
					if (len(barChartData['names']) > 0):

						# Update and display the annotation
						barchartAnno.set_text(
							f"{barChartData['names'][i]}\n"+
							f"{barChartData['sales'][i]} sales"
						)
						barchartAnno.xy = barChartData['sales'][i], np.linspace(0.2, 0.8, 3)[i]
						barchartAnno.set_visible(True)

	fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmove)

plt.show()

