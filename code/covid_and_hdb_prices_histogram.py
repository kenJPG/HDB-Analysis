# ====================================
#	Produces a diverging histogram that allows users to compare the distribution of resale prices
# 	of two years.
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

# ==========================
# 	Creating Figure
# ==========================
fig = plt.figure()
fig.suptitle(
	"How has Covid-19 affected resale prices?",
	verticalalignment='center',
	fontsize='xx-large'
)
fig.set_size_inches(10, 6)
fig.set_dpi(120)


# ==========================
# 	Creating subplots
# ==========================

infoBoxSubplot = fig.subplots(3, 4, gridspec_kw={'width_ratios': [1, 4, 2, 1], 'height_ratios': [5, 7, 1]})
histogramSubplot = fig.subplots(3, 3, gridspec_kw={'width_ratios': [1, 6, 1], 'height_ratios': [5, 7, 1]})
yearsSubplot = fig.subplots(3, 3, gridspec_kw={'width_ratios': [1, 6, 1], 'height_ratios': [5, 7, 1]})

# Contains the colors for the histogram
colors = {
	'left': (214/255, 40/255, 40/255, 0.5),
	'right': (0/255, 109/255, 119/255, 0.5), 
}

# ==========================
# 	Creating the info box
# ==========================
for i, blankArea in enumerate(infoBoxSubplot.flatten()):
	if (not(i == 1 or i == 2)):
		blankArea.set_visible(False)
infoBoxAxes = infoBoxSubplot[0][1]
aggregateParam = infoBoxSubplot[0][2]

# Remove spines
infoBoxAxes.spines['top'].set_color((0,0,0,0))
infoBoxAxes.spines['bottom'].set_color((0,0,0,0))
infoBoxAxes.spines['left'].set_color((0,0,0,0))
infoBoxAxes.spines['right'].set_color((0,0,0,0))

# Removing all the xticks and yticks
infoBoxAxes.set_xticks([])
infoBoxAxes.set_yticks([])

bigNumber = infoBoxAxes.text(
	0.5,
	0.5,
	"Click on aggregate ->",
	horizontalalignment='center',
	fontsize='xx-large',
	fontweight='bold',
	fontfamily='monospace'
)

explanation = infoBoxAxes.text(
	0.5,
	0.3,
	"",
	horizontalalignment='center',
)

difference = infoBoxAxes.text(
	0.5,
	0.1,
	"",
	horizontalalignment='center',
	fontsize='small'
)

# ==========================
# 	Creating the aggregate parameter
# ==========================

def lineContains(line, x, y):
	point_1, point_2 = line.get_xydata()
	if point_1[1]-0.1 <= y and y <= point_2[1] + 0.1:
		if point_1[0] <= x and x <= point_2[0]:
			return True
	return False
aggregateLabels = ["Mean", "Median", "Max", "Std Deviation"]
aggregate_x = np.full((4,1), 0.2)
aggregate_y = np.linspace(0.9, 0.1, 4)

boxSize = 25

# On loadup, show "Mean" as default
aggregateSelected = "Mean"

aggregateBoxes = [
	aggregateParam.plot(
		[0.2, 0.8],
		np.full((2,1), aggregate_y[i]),
		linewidth=boxSize,
		solid_capstyle='round',
		color = (0,0,0,0.05),
		label = aggregateLabels[i]
	)
	for i in range(0, 4)
]

# The circles that indicate the currently selected
# aggregate
aggregateMarkers = [
	aggregateParam.plot(
		aggregate_x[i],
		aggregate_y[i],
		label=aggregateLabels[i],
		marker='o',
		color='white',
		linestyle='None'
	)
	for i in range(0,4)
]

# Display the aggregate labels
for text, x, y in zip(aggregateLabels, aggregate_x, aggregate_y):
	aggregateParam.text(
		x + 0.08,
		y,
		text,
		verticalalignment='center',
		label=text
	)

# Hide borders
aggregateParam.spines['top'].set_color((0,0,0,0))
aggregateParam.spines['bottom'].set_color((0,0,0,0))
aggregateParam.spines['left'].set_color((0,0,0,0))
aggregateParam.spines['right'].set_color((0,0,0,0))

# Clear ticks
aggregateParam.set_xticks([])
aggregateParam.set_yticks([])

# This ensures that the limits are correct and that
# pyplot does NOT auto-adjust the limits.
aggregateParam.set_xlim(0, 1)
aggregateParam.set_ylim(0, 1)

# ==========================
# 	Creating the histogram
# ==========================

# This function is used to format the y-axis for the
# histogram
def dollar(value, pos):
	return '${:1.1f}mil'.format(value*1e-6)

# Hiding areas of the subplot that are not used
for i, blankArea in enumerate(histogramSubplot.flatten()):
	if (not(i == 4)):
		blankArea.set_visible(False)

histogramAxes = histogramSubplot[1][1]

# Specify the bin size here
binsize = 80000
bins = np.arange(0, 1500000, binsize)

# Given a year, we filter the data by that year and return the price column 
def generatePriceColumn(year):
	return [row['price'] for row in resale_prices if row['month'].split('-')[0] == str(year)]

# By default we create a histogram that displays years 2019 and 2020
_, _, patches = histogramAxes.hist(
	[generatePriceColumn(2019), generatePriceColumn(2020)],
	color=[colors['left'], colors['right']],
	bins = bins,
	rwidth=0.85,
	orientation='horizontal',
	stacked=True
)

# Histograms are not similar to plots, instead of line data, we have
# to work with patches.

# The code below changes a count-based histogram to a proportion-based histogram.
# Doing so allows easier price comparison between years.
total = sum([patch.get_width() for patch in patches[0]])

for patch in patches[0]:

	# Center pivot starts at 1, the height is auto determined
	patch.xy = (1, patch.xy[1])

	# Negative width to point to the left
	patch.set_width(-(patch.get_width() / total))

total = sum([patch.get_width() for patch in patches[1]])
for i, patch in enumerate(patches[1]):
	prefix = patches[0][i].get_width()

	# We want to copy the same starting point as the left histogram
	patch.xy = (patches[0][i].xy)
	patch.set_width((patch.get_width() / total))

# Move the small values to the top and move
# the larger values to the bottom
histogramAxes.set_ylim(0, 1500000)
histogramAxes.invert_yaxis()
histogramAxes.set_xlim((0+0.5, 2-0.5))

# Removing the x-axis ticks
histogramAxes.set_xticks([])

# Formatting the y-axis
histogramAxes.yaxis.set_major_formatter(dollar)

# Remove the borders
histogramAxes.spines['top'].set_color((0,0,0,0))
histogramAxes.spines['bottom'].set_color((0,0,0,0))
histogramAxes.spines['left'].set_color((0,0,0,0))
histogramAxes.spines['right'].set_color((0,0,0,0))

histogramAxes.set_facecolor((0,0,0,0.05))

# Black middle line
histogramAxes.plot(
	[1, 1],
	[0, 1500000],
	color='black',
	linewidth=2
)

leftYearText = histogramAxes.text(
	0.5+0.05,
	1300000,
	"2019",
	fontsize="xx-large",
)

rightYearText = histogramAxes.text(
	1.5-0.05,
	1300000,
	"2020",
	horizontalalignment='right',
	fontsize="xx-large"
)

# Annotation when hovering over histogram
annot = histogramAxes.annotate(
	"",
	xy=(0, 0),
	xytext=(10,10),
	textcoords="offset points",
	bbox=dict(boxstyle="round", facecolor="w"),
	fontstyle="italic"
)

# ==========================
# 	Creating the years parameter
# ==========================

for i, blankArea in enumerate(yearsSubplot.flatten()):
	if (not(i == 7)):
		blankArea.set_visible(False)
yearsParam = yearsSubplot[2][1]

unique_years = np.unique([row['month'].split('-')[0] for row in resale_prices])

yearsParam.set_ylim(0, 1)

yearsParam.spines['top'].set_color((0,0,0,0))
yearsParam.spines['bottom'].set_color((0,0,0,0))
yearsParam.spines['left'].set_color((0,0,0,0))
yearsParam.spines['right'].set_color((0,0,0,0))

yearsParam.set_xticks([])
yearsParam.set_yticks([])

years_x = np.linspace(0, 1, len(unique_years))
years_y = np.full((len(unique_years)), 0.5);

# We need to make an array of individual plots. 
# If we don't, when one color changes, it will change
# all other colors and we don't want that
yearsButtons = [
	yearsParam.plot(
		years_x[i],
		years_y[i],
		label=unique_years[i],
		marker='s',
		color='#DDDDDD',
		linestyle='None'
	)
	for i in range(0,len(unique_years))
]

yearsText = [
	yearsParam.text(
		years_x[i],
		years_y[i] - 0.9,
		unique_years[i],
		horizontalalignment='center'
	)
	for i in range(0,len(unique_years))
]

# This information will be displayed to help the
# user understand how to interact with the squares
yearsIntroText = yearsParam.text(
	0.5,
	-1.5,
	"LEFT click on a square to change the year for the red histogram",
	horizontalalignment='center',
	color=(0,0,0,0.5),
	fontstyle='italic',
	fontweight='bold'
)

# These variables will track whether or not the user
# has tried using the year parameters
introLeftClick = False
introRightClick = False

# ==========================
# 	Adding interactivity
# ==========================


# Given two pairs of coordinates, return the euclidean distance
# between them
def euclideanDistance(x1, y1, x2, y2):
	return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# This list stores the buttons that are clicked 
# On load we set the clicked buttons to 2019 and 2020
yearButtonTrack = [button[0] for button in yearsButtons if button[0].get_label() == '2019' or button[0].get_label() == '2020']

# This function updates all the graphs.
# This should be called after a parameter is selected
# or a filter is applied
def updateHistogram():
	global patches

	# It is not possible to update a histogram like updating a plot
	# As a result, we have to remove the old histogram and
	# create a new histogram after we filter

	# Remove the histogram
	[patch.remove() for patch in patches]

	# Creating new histogram [Same as the original creation code]
	_, _, patches = histogramAxes.hist(
		[generatePriceColumn(yearButtonTrack[0].get_label()), generatePriceColumn(yearButtonTrack[1].get_label())],
		color=[colors['left'], colors['right']],
		bins = bins,
		rwidth=0.85,
		orientation='horizontal',
		stacked=True
	)

	leftYearText.set_text(yearButtonTrack[0].get_label())
	rightYearText.set_text(yearButtonTrack[1].get_label())

	total = sum([patch.get_width() for patch in patches[0]])

	for patch in patches[0]:

		# Center pivot starts at 1, the height is auto determined
		patch.xy = (1, patch.xy[1])

		patch.set_width(-(patch.get_width() / total))

	total = sum([patch.get_width() for patch in patches[1]])
	for i, patch in enumerate(patches[1]):

		# We want to copy the same starting point as the left histogram
		patch.xy = (patches[0][i].xy)
		patch.set_width((patch.get_width() / total))

# Re-generate and display information for the info box.
def updateInfoBox():

	leftyear_price = generatePriceColumn(yearButtonTrack[0].get_label())
	rightyear_price = generatePriceColumn(yearButtonTrack[1].get_label())

	# We change the aggregate information and message depending on what
	# the user has chosen
	if (aggregateSelected == "Mean"):
		left_aggregate = int(np.mean(leftyear_price))
		right_aggregate = int(np.mean(rightyear_price))
		explanation.set_text(f"is the mean for {yearButtonTrack[1].get_label()}")
	elif (aggregateSelected == "Median"):
		left_aggregate = int(np.median(leftyear_price))
		right_aggregate = int(np.median(rightyear_price))
		explanation.set_text(f"is the median for {yearButtonTrack[1].get_label()}")
	elif (aggregateSelected == "Max"):
		left_aggregate = int(np.max(leftyear_price))
		right_aggregate = int(np.max(rightyear_price))
		explanation.set_text(f"is the max for {yearButtonTrack[1].get_label()}")
	elif (aggregateSelected == "Std Deviation"):
		left_aggregate = int(np.std(leftyear_price))
		right_aggregate = int(np.std(rightyear_price))
		explanation.set_text(f"is the standard deviation for {yearButtonTrack[1].get_label()}")

	bigNumber.set_text(f"${right_aggregate:,}")
	percentDiff = round((right_aggregate - left_aggregate) * 100 / left_aggregate, 2)

	if (right_aggregate < left_aggregate):
		difference.set_text(f"▼ {percentDiff}% change from {yearButtonTrack[0].get_label()}")
		difference.set_color('red')
	else:
		difference.set_text(f"▲ +{percentDiff}% change from {yearButtonTrack[0].get_label()}")
		difference.set_color('green')


# This function resets all the year buttons to their
# non-selected color
def resetButtonColor():
	for button in yearsButtons:
		button = button[0]
		button.set_color('#DDDDDD')

# This function applies effects onto the square that
# has been clicked
def yearClicked(button, mouseButtonPressed):
	global yearTrack

	if (str(mouseButtonPressed) == "MouseButton.LEFT"):
		yearButtonTrack[0].set_color('#DDDDDD')
		yearButtonTrack[0] = button
		button.set_color(colors['left'])
	elif (str(mouseButtonPressed) == "MouseButton.RIGHT"):
		yearButtonTrack[1].set_color('#DDDDDD')
		yearButtonTrack[1] = button
		button.set_color(colors['right'])
	updateHistogram()
	updateInfoBox()


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

# This function runs when button_press_event triggers
def onclick(event):
	global aggregateSelected, introRightClick, introLeftClick

	# If user clicks anywhere in the aggregate parameter
	if event.inaxes == aggregateParam:

		# This variable contains the label of the 
		# aggregate box that is clicked on
		labelChange = None

		for box in aggregateBoxes:

			# NOTE: 'box' is a list that contains 1 Line2D object.
			# We change it to that Line2D object
			box = box[0]
			if (lineContains(box, event.xdata, event.ydata)):
				labelChange = box.get_label()
				aggregateSelected = box.get_label()

		for marker in aggregateMarkers:
			marker = marker[0]
			if (labelChange == marker.get_label()):
				marker.set_color('gray')
			else:
				marker.set_color('white')
		updateInfoBox()

	# If user clicks anywhere in the year parameter
	elif event.inaxes == yearsParam:

		for button in yearsButtons:

			# NOTE: 'button' is a list that contains 1 Line2D object.
			# We change it to that Line2D object
			button = button[0]
			button_x, button_y = button.get_xydata()[0]


			# We convert our x and y data into x and y pixels
			# If we calculate euclidean distance on the chart without converting the coordinates into pixels,
			# the mouse will have to be closer on the y-axis than the x-axis in order to register our click. This
			# is because our x-axis is much wider.
			# By converting the data points into pixels, we get a 1:1 ratio of x-axis-scale:y-axis-scale, which is want we want.
			button_x, button_y = toPixel(yearsParam, button_x, button_y)

			event_x, event_y = toPixel(yearsParam, event.xdata, event.ydata)

			if (euclideanDistance(button_x, button_y, event_x, event_y) <= 9):
				yearClicked(button, event.button)

				# Guide to help the user understand left/right clicks
				if not introLeftClick:
					yearsIntroText.set_text("RIGHT click for the blue histogram")
					introLeftClick = True
				else:
					if not introRightClick:
						yearsIntroText.set_text("")
						introRightClick = True
	fig.canvas.draw_idle()

# This function runs when any mouse movement is detected
def onmove(event):

	# If we move out of the aggregate boxes, 
	# reset the dark effect from hovering
	if (event.inaxes != aggregateParam):
		for box in aggregateBoxes:
			box = box[0]
			box.set_color((0,0,0,0.05))

	# If we move out of the histogram, we will
	# hide the annotation and reset rectangle edge color
	if (event.inaxes != histogramAxes):
		annot.set_visible(False)
		for barContainer in patches:
			for rectangle in barContainer:
				rectangle.set_edgecolor((0,0,0,0))

	# If the hovered axes is the aggregate axes,
	if (event.inaxes == aggregateParam):

		# Darken the box we hover over
		for box in aggregateBoxes:

			# NOTE: box is a list that contains 1 Line2D object.
			# We change it to that Line2D object it contains
			box = box[0]

			if (lineContains(box, event.xdata, event.ydata)):
				box.set_color((0,0,0,0.1))
			else:
				box.set_color((0,0,0,0.05))

	elif (event.inaxes == histogramAxes):

		# Handle the hovering of histograms
		for barContainer in patches:
			for rectangle in barContainer:
				if (rectangle.contains_point([event.x, event.y])):

					# Black border around the hovered rectangle
					rectangle.set_edgecolor('black')

					# Calculate the bin for that rectangle
					rectanglebin = int((rectangle.get_y() // binsize) * binsize)

					# Update the text for annotation and display it
					annot.set_text(
						f"\${rectanglebin:,}-\${rectanglebin + binsize:,}\n"+
						f"{round(abs(rectangle.get_width()) * 100, 2)}% of distribution"
					)
					annot.xy = (event.xdata, event.ydata)
					annot.set_visible(True)
				else:

					# No border for those unhovered rectangles
					rectangle.set_edgecolor((0,0,0,0))

	fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmove)

plt.show()

