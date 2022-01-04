
# ====================================
#	Produces a piechart that displays the proportion of the population that lives in HDBs,
#	compared to other types of residential properties
# ====================================

# Import modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import pi

# Ensuring that the current directory is correct
os.chdir(
	(os.path.abspath(os.path.dirname(__file__)))
)

# Reading files
hdb_proportion = np.genfromtxt('../datasets/hdb_percent_of_population.csv', dtype=[('year', np.int32), ('type', 'U32'), ('percent', np.int32)], skip_header=1, delimiter=',')
cumulative_hdb = np.genfromtxt('../datasets/cumulative_hdb.csv', dtype=[('year', np.int32), ('type', 'U32'), ('units', np.int32)], skip_header=1, delimiter=',')

# Certain datasets may have years that others do not. We want to display
# the latest year that exists in both datasets
hdb_proportion_years = set(row['year'] for row in hdb_proportion)
cumulative_hdb_years = set(row['year'] for row in cumulative_hdb)
latest_year = max(hdb_proportion_years.intersection(cumulative_hdb_years))

print("Latest year that exists in both datasets:",latest_year)

# Filter by that latest year and by type of HDB flats
filter_array = [row['year'] == latest_year and row['type'] == 'HDB Flats' for row in hdb_proportion] # Generates a boolean array
filtered_hdb_proportion = hdb_proportion[filter_array]

# Filter the cumulative_hdb dataset by the latest year
filter_array = [row['year'] == latest_year and row['type'] == 'HDB' for row in cumulative_hdb] # Generates a boolean array
filtered_cumulative_hdb = cumulative_hdb[filter_array]

# This variable represents the proportion of the resident population that live in HDBs
percent_of_hdb = filtered_hdb_proportion[0]['percent']

# This dictionary contains all colors that are used
colors = {
	'hdb_fill': '#92C5DE',
	'HDB': '#607E8C',
	'others_fill': '#F9ACB1',
	'Others': '#C47071',
	'null': '#F7F7F7',
	'dark_null': '#D1D1D1'
}

# ==========================
# 	Creating Figure
# ==========================
fig = plt.figure()
fig.suptitle(
	"What proportion of residents live in HDBs?",
	verticalalignment='center',
	fontsize='xx-large'
)
fig.set_size_inches(10, 6)
fig.set_dpi(120)

# ==========================
# 	Creating subplots
# ==========================
# Linegraph subplot needs to be made first.
# If we don't do this, any annotation on the piechart will be covered by the linegraph
linegraph = fig.add_subplot(4, 3, 9)

# infoBox displays numerical information depending on which wedge is clicked 
infoBox = fig.add_subplot(4, 3, 6)

# We only want to show the text. This hides the chart but not the text.
# infoBox.set_visible(False) also hides the text and thus is not used
infoBox.axis('off')

# Piechart subplot
piechartAxes = fig.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [2, 0]})
# [Explanation]
# Each character represents a chart. The code above transforms the subplot as shown below.

# From this 2 row x 2 column
# ab
# cd

# to this 1 row x 2 column
# aab
# aab

# We want our piechart to take up the space of 'a', which is accessed by index 0,0
piechart = piechartAxes[0][0]

# Hide all other charts that are not used.
for i in range(0, 2):
	for j in range(0, 2):
		if (not (i == 0 and j == 0)):
			print(i, j)
			piechartAxes[i][j].set_visible(False)

# ==========================
# 	Creating the piechart
# ==========================
patches, _ = piechart.pie(
	[percent_of_hdb, 100 - percent_of_hdb],
	labels=['HDB', 'Others'],
	colors=[colors['hdb_fill'], colors['others_fill']],
	startangle=90,
	wedgeprops = {
		'linewidth': 2.5
	}
)

# The white space to make a donut chart
donut_space, _ = piechart.pie(
	[percent_of_hdb, 100 - percent_of_hdb],
	colors=['white'],
	startangle=90,
	wedgeprops = {
		'linewidth': 3,
		'edgecolor': 'white'
	},
	radius = 0.67
)

# Year text
piechart.text(
	0,
	-0.05,
	f"In {latest_year}",
	fontsize='xx-large',
	fontweight='ultralight',
	horizontalalignment='center'
)

# Piechart annotation
annot = piechart.annotate(
	"Click to highlight on area chart",
	xy=(0, 0),
	xytext=(10,10),
	textcoords="offset points",
	bbox=dict(boxstyle="round", facecolor="w"),
	fontstyle="italic"
)

annot.set_visible(False)

# ==========================
# 	Creating the infoBox
# ==========================
textBoxData = {
	'HDB': {
		'percent': percent_of_hdb,
		'count': filtered_cumulative_hdb[0]['units']

	},
	'Others': {
		'percent': 100 - percent_of_hdb
	}
}

percentNumber = infoBox.text(
	0.5,
	0.9,
	"",
	horizontalalignment='center',
	fontweight='bold',
	fontfamily='monospace',
	fontsize='xx-large'
)

infoBoxText = infoBox.text(
	0.5,
	0.3,
	"Click on the sections\nof the piechart to show info",
	horizontalalignment='center'
)

# ==========================
# 	Creating the linegraph
# ==========================

def filterData(data, column, value = []):
	filter_array = [row[column] in value for row in data] # Generates a boolean array

	# Apply the filter
	return data[filter_array]

filtered_linegraph = filterData(hdb_proportion, 'type', ["HDB Flats"])
year_column = np.array([row['year'] for row in filtered_linegraph])
percent_column = np.array([row['percent'] for row in filtered_linegraph])
print(year_column, percent_column)

proportion_line, = linegraph.plot(year_column, percent_column, color=colors['dark_null'])
linegraph_fill = linegraph.stackplot(year_column, [percent_column, 100 - percent_column], labels=['HDB', 'Others'], colors=['#EFEFEF', '#EFEFEF'])

linegraph.spines['top'].set_color((0,0,0,0))
linegraph.spines['left'].set_color((0,0,0,0))
linegraph.spines['right'].set_color((0,0,0,0))

# Axis limits
linegraph.set_xlim(2008, 2018)
linegraph.set_ylim(0, 100)

# Axis ticks
linegraph.set_xticks([min(year_column), latest_year]);
linegraph.set_yticks([0, 100]);
linegraph.yaxis.set_major_formatter(ticker.PercentFormatter())

# ==========================
# 	Adding interactivity
# ==========================


# This function explodes a wedge by a factor
def wedgeExplode(wedge, factor):
	explode_angle = ((wedge.theta1 + wedge.theta2) / 2) % 360

	# Given an angle, we calculate the x and y displacement and convert it to radians
	explode_x = factor * np.cos((explode_angle / 360) * 2 * pi)
	explode_y = factor * np.sin((explode_angle / 360) * 2 * pi)

	wedge.set_center((explode_x, explode_y))

# This function updates the text in the info box depending on the wedge
# that is clicked
def updateInfoBox(wedge):
	if (wedge.get_label() == 'HDB'):
		percentNumber.set_text(f"{textBoxData[wedge.get_label()]['percent']}%")

		infoBoxText.set_text("of the resident population live in HDBs\n\n"+
		f"{textBoxData[wedge.get_label()]['count']:,} total HDB units")
	else:
		percentNumber.set_text(
			f"{textBoxData[wedge.get_label()]['percent']}%"
		)
		infoBoxText.set_text("of the resident population live in other types\n\n")


# This variable tracks whether any wedge is clicked
wedgeClicked = False

# This function runs when the button press event triggers
def onclick(event):
	global wedgeClicked

	# We only want the click event to occur in the piechart
	if event.inaxes == piechart:

		# If there is a wedge that is still clicked and exploded,
		# we need to reset that wedge to normal
		if (wedgeClicked):
			wedgeClicked = False
			proportion_line.set_color(colors['dark_null'])

			for inner_wedge in donut_space:
				wedgeExplode(inner_wedge, 0)
			for wedge in patches:
				wedgeExplode(wedge, 0)
				wedge.set_edgecolor((0,0,0,0))
			for area_fill in linegraph_fill:
				area_fill.set_facecolor(colors['null'])

		# If there is no wedge that is currently clicked, we can
		# add the effects onto that wedge
		else:

			wedgeClicked = True

			for i, wedge in enumerate(patches):

				# Only apply the effects on the wedge that the mouse is touching
				if (wedge.contains_point([event.x, event.y])):

					# Update the info box depending on the wedge the user clicked
					updateInfoBox(wedge)
					wedge_label = wedge.get_label()
					wedge.set_edgecolor(colors[wedge_label])
					wedgeExplode(donut_space[i], 0.1)
					wedgeExplode(wedge, 0.1)
					proportion_line.set_color(colors[wedge_label])

					# We apply the effects onto the linegraph
					for area_fill in linegraph_fill:
						if (area_fill.get_label() == wedge_label):
							if (wedge_label == 'HDB'):
								area_fill.set_facecolor(colors['hdb_fill'])
							else:
								area_fill.set_facecolor(colors['others_fill'])
						else:
							area_fill.set_facecolor(colors['null'])

				else:
					# If this wedge is not the wedge that is touching 
					# the mouse, we will make sure it is not exploded
					wedgeExplode(donut_space[i], 0)
					wedgeExplode(wedge, 0)
	else:
		# If the user clicks anywhere that is outside of the
		# piechart, we will reset the figure to a default state
		wedgeClicked = False

		# Reset wedges
		for i, wedge in enumerate(patches):
			wedgeExplode(donut_space[i], 0)
			wedgeExplode(wedge, 0)
			wedge.set_edgecolor((0,0,0,0))

		# Reset area chart
		for area_fill in linegraph_fill:
			area_fill.set_facecolor(colors['null'])
		proportion_line.set_color(colors['dark_null'])

		# Reset info box
		percentNumber.set_text("")
		infoBoxText.set_text("Click on the sections\nof the piechart to show info")
	fig.canvas.draw_idle()

# This function runs when any mouse movement is detected
def onmove(event):
	global wedgeClicked

	# If user is not hovering over the piechart axes
	if event.inaxes != piechart:
		if annot.get_visible():
			annot.set_visible(False)

	# If user is hovering over the linegraph axes
	if event.inaxes == linegraph:
		pass

	# If user is hovering over the piechart axes
	elif event.inaxes == piechart:

		# Apply effects onto the wedge
		# that the user is hovering over
		for wedge in patches:
			if (wedge.contains_point([event.x, event.y])):
				annot.set_visible(True)
				annot.xy = (event.xdata, event.ydata)
				if (not wedgeClicked):
					wedge.set_edgecolor(colors[wedge.get_label()])
			else:
				if (not wedgeClicked):
					wedge.set_edgecolor((0,0,0,0))
	fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmove)

plt.show()