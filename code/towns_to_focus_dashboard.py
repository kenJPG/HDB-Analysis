# ====================================
#	Produces a dashboard consisting of a scatterplot, barchart 
# 	and a map, to determine which planning areas seem to have
#	the most growth.

# 	NOTE geopandas is needed for this Python script. Please follow
# 	the instructions in geopanda_setup.docx if you do not have geopandas
#	set up.
# ====================================

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shapely
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon

# Ensuring that the current directory is correct
os.chdir(
	(os.path.abspath(os.path.dirname(__file__)))
)

# Reading files
completion_status = np.genfromtxt('../datasets/completion_status.csv', dtype=[('year', np.int32), ('town', 'U32'), ('type', 'U32'), ('status', 'U32'), ('units', np.int32)], skip_header=1, delimiter=',')
resale_prices = np.genfromtxt('../datasets/resale_flat_prices_all.csv', dtype=[('month', 'U32'), ('town', 'U32'), ('area', np.int64), ('price', np.int64)], skip_header=1, delimiter=',')

# =============
# 	Cleaning
# =============


# ----- Completion Status -----

# Changing town names to all uppercase
completion_status['town'] = np.array(list(map(lambda x: x.upper(), completion_status['town'])), dtype='U32')

# Filter out DBSS
completion_status = completion_status[completion_status['type'] == 'HDB']

# Given year and town as a key, this 3d dictionary contains the number of buildings
# under construction and the number of buildings completed as of the latest year recorded.
completion_status_townDict = {
	singleYear: {
		singleTown: dict() for singleTown in completion_status['town']
	}
	for singleYear in completion_status['year']
}

for row in completion_status:
	if (row['status'] == 'Completed'):
		if (completion_status_townDict[row['year']][row['town']].get("completed")):
			completion_status_townDict[row['year']][row['town']]["completed"] += row['units']
		else:
			completion_status_townDict[row['year']][row['town']]["completed"] = row['units']
	else:
		if (completion_status_townDict[row['year']][row['town']].get("under construction")):
			completion_status_townDict[row['year']][row['town']]["under construction"] += row['units']
		else:
			completion_status_townDict[row['year']][row['town']]["under construction"] = row['units']

# ----- Number of Resales by Quarter By Town -----
# Set of all town names
# NOTE: This will be used to check if a town is inside the HDB dataset. set
# is used due to it being more efficient than a list
allTowns = set(np.unique(resale_prices['town']))

# This 2d dictionary contains the total number of sales and 
# the total area for a given key of 'town'.
total_price_area_by_townDict = {
	singleTown: dict() for singleTown in resale_prices['town']
}

# Adding up total price and area for each town
for singleTown in allTowns:
	total_price_area_by_townDict[singleTown]['price'] = np.sum(resale_prices[resale_prices['town'] == singleTown]['price'], dtype=np.int64)
	total_price_area_by_townDict[singleTown]['area'] = np.sum(resale_prices[resale_prices['town'] == singleTown]['area'], dtype=np.int64)

# This 2d list contains
# 	[<town>, <total_price>, <total_area>]
# as its elements
total_price_area_by_town = []

# Filling up the above list
for town in total_price_area_by_townDict.keys():
	total_price_area_by_town.append((town, total_price_area_by_townDict[town]['price'], total_price_area_by_townDict[town]['area']))

# Convert from Python list to numpy array as it is easier to work with
total_price_area_by_town = np.array(total_price_area_by_town, dtype=[('town', 'U32'), ('price', np.int64), ('area', np.int64)])

# -- Shape file ---
# Read the shapes file. This is retrieved from
# https://data.gov.sg/dataset/planning-area-census2010?resource_id=076de3a7-eaa6-426e-8331-557469dd7ce8
Singapore = gpd.read_file('../datasets/planning_area_shape/Planning_Area_Census2010.shp')

# We will only extract the necessary columns, which are 'PLN_AREA_N' planning area, 'CA_IND' central area indicator, 'geometry' contains
# the polygons needed to make the map
Singapore = {
	'PLN_AREA_N': list(Singapore['PLN_AREA_N']),
	'CA_IND': list(Singapore['CA_IND']),
	'geometry': list(Singapore['geometry'])
}

# The geometry column contains both Polygon and MultiPolygon data types. In order to use numpy arrays, we need
# both the data types to be the same for the entire column. Thus, we will be changing all Polygons to MultiPolygons
Singapore['geometry'] = list(map(lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x]), Singapore['geometry']))

Singapore = np.array(list(zip(Singapore['PLN_AREA_N'], Singapore['CA_IND'], Singapore['geometry'])), dtype=[('area', 'U32'), ('central', 'U32'), ('geometry', MultiPolygon)])


# Certain towns are grouped together as part of the 'Central Area'.
# https://www.hdb.gov.sg/cs/infoweb/about-us/history/hdb-towns-your-home/central

# The towns that are considered part of the 'Central Area' are indicated by the column 'CA_IND'
# Within that column, 'Y' tells us that that specific town IS part of the Central Area, while 'N' tells us that it IS NOT.

# We will merge all individual towns(their polygons) that are part of the central area to form the larger 'Central Area' group
central_union = shapely.ops.unary_union(Singapore[Singapore['central'] == "Y"]['geometry'])


# Our resales_flat_prices_all.csv dataset combines Kallang and Whampoa.
# Whampoa is a housing estate part of both Novena and Kallang planning area in Singapore. As a result, 
# we need to merge Kallang and Novena.
kallang_area = Singapore[Singapore['area'] == 'KALLANG']['geometry'][0] # Kallng Polygon
whampoa_area = Singapore[Singapore['area'] == 'NOVENA']['geometry'][0] # Novena Polygon (We will call it Whampoa area)

# Merged Kallang and Whampoa polygons
kallang_whampoa_union = shapely.ops.unary_union([kallang_area, whampoa_area])
kallang_whampoa_union = MultiPolygon([kallang_whampoa_union])

# Now that we have merged polygons, we filter out their old versions
clean_Singapore = Singapore[Singapore['central'] != 'Y']
clean_Singapore = clean_Singapore[clean_Singapore['area'] != 'KALLANG']
clean_Singapore = clean_Singapore[clean_Singapore['area'] != 'NOVENA']

# Append 'CENTRAL AREA' and 'KALLANG/WHAMPOA' to the numpy array containing all the towns and polygons.
# As we cannot directly append into a numpy ndarray, we will first need to create rows and then concatenate.
central_area_row = np.array([('CENTRAL AREA', 'Y', central_union)], dtype=[('area', 'U32'), ('central', 'U32'), ('geometry', MultiPolygon)])
kallang_whampoa_row = np.array([('KALLANG/WHAMPOA', 'N', kallang_whampoa_union)], dtype=[('area', 'U32'), ('central', 'U32'), ('geometry', MultiPolygon)])
clean_Singapore = np.concatenate([clean_Singapore, central_area_row])
clean_Singapore = np.concatenate([clean_Singapore, kallang_whampoa_row])

# ==========================
# 	Creating figure
# ==========================

fig = plt.figure()
fig.suptitle(
	"Which towns should we focus on?",
	verticalalignment='center',
	fontsize='xx-large'
)

fig.set_size_inches(11, 6)
fig.set_dpi(150)

colors = {
	'not_tab': '#F5F5F5',
	'tab': '#DDD',
	'map_selected': '#6377C6',
	'map_selected_edge': '#436E8F',
	'map_unselected': '#FDFDFD',
	'map_unselected_edge': '#DDD',
	'scatterline': '#CCC',
	'guide_text_color': '#CBCBCB',
	'type_box_bg': '#FAFAFA',
	'result_box_bg_blue': '#94AEF2',
	'result_box_bg_grey': '#FEFEFE',
	'result_box_edge': '#EDEDED',
	'result_box_text_blue': '#6377C6',
	'result_box_text_grey': '#BBB',
}

# ==========================
# 	Creating subplots
# ==========================

scatterSubplot = fig.subplots(3, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [2, 20, 10]})
barSubplot = fig.subplots(4, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [2, 10, 15, 5]})
tabsSubplot = fig.subplots(3, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [2, 25, 5]})
parameterSubplot = fig.subplots(3, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [2, 20, 10]})
mapSubplot = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1]})
colorBarSubplot = fig.subplots(3, 3, gridspec_kw={'width_ratios': [0.0000001, 500, 1000], 'height_ratios': [2, 20, 10]})

# ==========================
# 	Creating scatterplot
# ==========================

blue_cmap = plt.cm.Blues
grey_cmap = plt.cm.Greys

price_per_area = total_price_area_by_town['price']/total_price_area_by_town['area']
maximum_price_per_area = max(price_per_area)

# Return the color using a colormap
def getColor(price, area, use_bluecmap = False):
	price_per_area = price / area
	if (use_bluecmap):
		return blue_cmap(price_per_area / maximum_price_per_area)
	else:
		return grey_cmap(price_per_area / maximum_price_per_area)

# This function is used to format the y-axis(price) for the
# scatterplot
def dollar(value, pos):
	return '${:1.1f}bil'.format(value*1e-9)

# This function is used to format the x-axis(area) for the
# scatterplot
def metersquared(value, pos):
	return '{:.1f}km²'.format(value*1e-6)

for i, blankArea in enumerate(scatterSubplot.flatten()):
	if (not(i == 2)):
		blankArea.set_visible(False)

scatterAxes = scatterSubplot[1][0]

scatterplot = scatterAxes.scatter(
	total_price_area_by_town['area'],
	total_price_area_by_town['price'],
	color = list(
		map(
			lambda row: getColor(row['price'], row['area']),
			total_price_area_by_town
		)
	)
)

# Generating linear function for trendline
coefficient = np.polyfit(total_price_area_by_town['area'], total_price_area_by_town['price'], 1)
linear_regression = np.poly1d(coefficient)

min_area_point = min(total_price_area_by_town['area'])
max_area_point = max(total_price_area_by_town['area'])
min_lin_point = linear_regression(min_area_point)
max_lin_point = linear_regression(max_area_point)

# Plotting trendline for scatterplot
scatterline, = scatterAxes.plot(
	[min_area_point, max_area_point],
	[min_lin_point, max_lin_point],
	dashes=[6,5],
	color=colors['scatterline'],
	zorder=-1
)

# As pyplot scatter does not have any options for labels,
# we will make a 2d dictionary that maps price and area 
# to a label. This is not a perfect solution, if two towns
# have the same total price and area, one town will be overwritten.
# However, this is quite unlikely.
town_label_dict = {
	total_price: dict() for total_price in total_price_area_by_town['price']
}

# We can access the label by town_label_dict[<total price>][<total area>]
for row in total_price_area_by_town:
	town_label_dict[row['price']][row['area']] = row['town']

scatterAnnot = scatterAxes.annotate(
	"",
	xy=(0, 0),
	xytext=(10,10),
	textcoords="offset points",
	bbox=dict(boxstyle="round", facecolor="w"),
)

scatterAxes.set_zorder(3)

# Hide all spines
scatterAxes.spines[:].set_color((0,0,0,0))

scatterAxes.set_xticks(np.arange(0, 1800000, 300000))

scatterAxes.xaxis.set_major_formatter(metersquared)
scatterAxes.yaxis.set_major_formatter(dollar)

# Show scatterplot on load
scatterAxes.set_visible(True)

# ==========================
# 	Creating colorbar for scatterplot
# ==========================

for i, blankArea in enumerate(colorBarSubplot.flatten()):
	if (not(i == 4)):
		blankArea.set_visible(False)

colorBarAxes = colorBarSubplot[1][1]

# We first make an image from the resales dataset. This image is used in producing the colorbar
colorBarImageData = price_per_area.reshape(len(price_per_area), 1)

# The problem right now is that the colorbar will start at 0 white. 
# This is useless as we do not have 0 as a datum.
# We want to re-interpolate the colormap such that the minimum color
# begins at the minimum data point we have
colorbar_cmap = matplotlib.cm.Greys(np.linspace(0,1,100))

# Determine the ratio of the bottom of the color map
min_ratio = min(price_per_area) / max(price_per_area)

# Use the new re-interpolated colormap
colorbar_cmap = matplotlib.colors.ListedColormap(colorbar_cmap[round(min_ratio * 100):,:-1])

# Generating the images for colorbar
colorBarImageData = np.concatenate([colorBarImageData, [[0]]])
colorBarImage = colorBarAxes.imshow(colorBarImageData, cmap=colorbar_cmap, interpolation="None", origin='lower', vmin=min(price_per_area), vmax=max(price_per_area))
colorBarImage.set_visible(False)

# Making the colorbar.
# We need this as a function as it is not possible to hide/show colorbars
# As a result, we need to remove and recreate the colorbar if we want
# to hide/show
def makeColorBar():
	global colorbar

	colorbar = fig.colorbar(
		colorBarImage,
		ax=colorBarAxes,
		location="left",
		fraction=0.015,
		anchor=(0.0,1),
		ticks=[min(price_per_area), maximum_price_per_area],
		format=matplotlib.ticker.FormatStrFormatter("$%d / m²")
	)
	colorbar.ax.set_zorder(3)
	colorbar.ax.yaxis.set_ticks_position('right')
makeColorBar()

colorBarAxes.set_facecolor((0,0,0,0))

colorBarAxes.spines['top'].set_color((0,0,0,0))
colorBarAxes.spines['bottom'].set_color((0,0,0,0))
colorBarAxes.spines['left'].set_color((0,0,0,0))
colorBarAxes.spines['right'].set_color((0,0,0,0))

colorBarAxes.set_xticks([])
colorBarAxes.set_yticks([])



# ==========================
# 	Creating barchart
# ==========================

max_year = max(completion_status['year'])

def filterData(data, column, value = []):
	filter_array = [row[column] in value for row in data] # Generates a boolean array

	# Apply the filter
	return data[filter_array]

for i, blankArea in enumerate(barSubplot.flatten()):
	if (not(i == 2 or i == 4)):
		blankArea.set_visible(False)

barTextAxes = barSubplot[1][0]
barAxes = barSubplot[2][0]

under_construction = filterData(completion_status, 'year', [max_year]) # Filter to latest year
under_construction = filterData(under_construction, 'status', ['Under Construction']) # Filter to only 'Under Construction'
under_construction = filterData(under_construction, 'type', ['HDB']) # Filter to only 'Under Construction'

# Sort by units
under_construction = under_construction[(under_construction['units']).argsort()][::-1]

barchart  = barAxes.bar(
	np.arange(0, 5),
	under_construction['units'][0:5],
	label=under_construction['town'][0:5],
	color=colors['map_selected']
)

barTownText = barTextAxes.text(
	0.5,
	0.8,
	"",
	fontfamily='monospace',
	horizontalalignment='center',
	verticalalignment='center',
	color=colors['map_selected'],
	fontweight='bold',
	fontsize='large'
)

barUnitsText = barTextAxes.text(
	0.5,
	0.4,
	"click on a bar for information",
	horizontalalignment='center',
	verticalalignment='center'
)

# --- Map hover annotation ---
barAnnot = barAxes.annotate(
	"",
	xy=(0, 0),
	xytext=(10, 10),
	textcoords="offset points",
	bbox=dict(boxstyle="round", facecolor='w'),
)

barAxes.spines[:].set_color((0,0,0,0))
barAxes.set_xticks([])

barTextAxes.spines[:].set_color((0,0,0,0))
barTextAxes.set_xticks([])
barTextAxes.set_yticks([])

barAxes.set_ylabel("Units under construction")

# Hide bar chart on load
barAxes.set_visible(False)
barTextAxes.set_visible(False)

# ==========================
# 	Creating tabs
# ==========================

for i, blankArea in enumerate(tabsSubplot.flatten()):
	if (not(i == 0)):
		blankArea.set_visible(False)
tabsAxes = tabsSubplot[0][0]

scatterTabPolygon = Polygon([(0, 0), (0, 1), (0.5, 1), (0.5, 0)])
barTabPolygon = Polygon([(0.5, 0), (0.5, 1), (1, 1), (1, 0)])

scatterTab, = tabsAxes.fill(*scatterTabPolygon.exterior.xy, color=colors['tab'])
barTab, = tabsAxes.fill(*barTabPolygon.exterior.xy, color=colors['not_tab'])

scatterTabText = tabsAxes.text(
	0.25,
	0.5,
	"Sales and area by town",
	horizontalalignment='center',
	verticalalignment='center',
	fontweight='bold'
)

barTabText = tabsAxes.text(
	0.75,
	0.5,
	"Upcoming constructions",
	horizontalalignment='center',
	verticalalignment='center'
)

tabsAxes.set_xlim(0, 1)
tabsAxes.set_ylim(0, 1)

tabsAxes.spines[:].set_color((0,0,0,0))

tabsAxes.set_xticks([])
tabsAxes.set_yticks([])

# ==========================
# 	Creating the parameters
# ==========================

for i, blankArea in enumerate(parameterSubplot.flatten()):
	if (not(i == 4)):
		blankArea.set_visible(False)
parameterAxes = parameterSubplot[2][0]

textBoxPolygon = Polygon([(0.15, 0.3), (0.15, 0.7), (0.85, 0.7), (0.85, 0.3)])

# Height of each result box
box_size = 0.4

# This contains all the boxes of the results
result_box = []

result_text = [
	parameterAxes.text(
		0.25,
		0.5 - (box_size * i),
		"",
		fontfamily='monospace',
		verticalalignment='center'
	)
	for i in range(1, 5)
]

def generateBox(y):
	displacement = (box_size) * y
	return Polygon([(0.15, 0.3 - displacement), (0.15, 0.7 - displacement), (0.85, 0.7 - displacement), (0.85, 0.3 - displacement)])

def updateResultText(y, townname, selected=False):
	result_text[y].set_text(
		townname
	)
	result_text[y].set_color(
		colors['result_box_text_blue'] if selected else
		colors['result_box_text_grey']
	)

textBox = parameterAxes.fill(
	*textBoxPolygon.exterior.xy,
	color=colors['type_box_bg']
)

# Text that tells user to click before typing
parameterClickGuideText = parameterAxes.text(
	0.2,
	0.5,
	"click here to search",
	verticalalignment='center',
	horizontalalignment='left',
	fontfamily='monospace',
	color=colors['guide_text_color']
)
parameterClickGuideText.set_visible(True)

# Text that tells user to start typing(placeholder)
parameterTypeGuideText = parameterAxes.text(
	0.2,
	0.5,
	"start typing...",
	verticalalignment='center',
	horizontalalignment='left',
	fontfamily='monospace',
	color=colors['guide_text_color']
)

parameterTypeGuideText.set_visible(False)

# Text that the user is typing
parameterResultClickGuideText = parameterAxes.text(
	0.2,
	-1.65,
	"^ you can click on the results",
	color=colors['guide_text_color'],
	verticalalignment='bottom',
	horizontalalignment='left',
	fontstyle='italic',
	fontweight='light',
	fontfamily='monospace',
)
parameterResultClickGuideText.set_visible(False)

# Text that the user is typing
parameterText = parameterAxes.text(
	0.2,
	0.5,
	"",
	verticalalignment='center',
	horizontalalignment='left',
	fontfamily='monospace',
)

parameterText.set_visible(False)

# This array contains all key presses that are allowed.
acceptableKeys = [chr(ord('a')+i) for i in range(0, 26)]
acceptableKeys += [chr(ord('A')+i) for i in range(0, 26)]
acceptableKeys += ['backspace', ' ', '/']

def addToText(textstring, key):
	if key in acceptableKeys:
		if (key == 'backspace'):
			return textstring[:-1]
		else:
			return textstring + key
	else:
		return textstring

parameterAxes.spines[:].set_color((0,0,0,0))

parameterAxes.set_xlim(0, 1)
parameterAxes.set_ylim(-1.7, 1)

parameterAxes.set_xticks([])
parameterAxes.set_yticks([])

# ==========================
# 	Creating map
# ==========================


# This list will contain all patches from the map
patches = []

for i, blankArea in enumerate(mapSubplot.flatten()):
	if (not(i == 1)):
		blankArea.set_visible(False)
mapAxes = mapSubplot[1]

for i in range(0, len(clean_Singapore['geometry'])):
	areaname = clean_Singapore['area'][i]
	item = clean_Singapore['geometry'][i]

	for geom in item.geoms:

		# We only fill areas that are part of allTowns
		if (areaname in allTowns):
			patch, = mapAxes.fill(*geom.exterior.xy, label=str(areaname), color=colors['map_unselected'], edgecolor=colors['map_unselected_edge'], linewidth=0.8)
			patch.set_zorder(2)
			patches.append(patch)
		# Any other area(such as inhabitable areas) are grayed out
		else:
			patch, = mapAxes.fill(*geom.exterior.xy, label='blank', color='#EEE', edgecolor=(0,0,0,0), linestyle="None", linewidth=0)
			patch.set_zorder(1)

# This function deletes and recreates 
# the map. This is useful when switching
# between the scatterplot and barchart as
# certain towns don't exist in the scatterplot
# but do exist in the barchart
def refreshMap():
	global patches

	# Remove all patches from the map
	[patch.remove() for patch in patches]

	patches = []

	for i in range(0, len(clean_Singapore['geometry'])):
		areaname = clean_Singapore['area'][i]
		item = clean_Singapore['geometry'][i]

		for geom in item.geoms:

			# We only fill areas that are part of allTowns
			if (areaname in allTowns):
				patch, = mapAxes.fill(*geom.exterior.xy, label=str(areaname), color=colors['map_unselected'], edgecolor=colors['map_unselected_edge'], linewidth=0.8)
				patch.set_zorder(2)
				patches.append(patch)
			# Any other area(such as inhabitable areas) are grayed out
			else:
				patch, = mapAxes.fill(*geom.exterior.xy, label='blank', color='#EEE', edgecolor=(0,0,0,0), linestyle="None", linewidth=0)
				patch.set_zorder(1)

# --- Map hover annotation ---
mapAnnot = mapAxes.annotate(
	"",
	xy=(0, 0),
	xytext=(10, 10),
	textcoords="offset points",
	bbox=dict(boxstyle="round", facecolor='w'),
)

mapAnnot.set_visible(False)

mapAxes.axis('equal')

mapAxes.spines[:].set_color((0,0,0,0))

mapAxes.set_xticks([])
mapAxes.set_yticks([])

# ==========================
# 	Adding interactivity 
# ==========================

# This array tracks indexes of data points from the scatterplot
# that the user has clicked. 'True' tells us the user has clicked on 
# that data point an odd number of times, while 'False' tells us the user
# has clicked an even number of times. 
# This is used in 'turning on/off' the colors of data points
clickedScatterIndex = [False] * len(total_price_area_by_town['town'])

# This variable tracks whether or not the user
# is allowed to type in the textbox parameter
allowTyping = False

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

# Given event and a list of indexes for scatterplot,
# return the index that points to the data point that is
# the closest to the mouse
def getClosestIndex(event, indexes):

	# Closest stores the index
	# closest_dist stores the distance to the mouse
	# of that index
	closest = -1
	closest_dist = np.inf

	for idx in indexes:

		# Get the pixel coordinates
		pixel_x, pixel_y = toPixel(scatterAxes, total_price_area_by_town['price'][idx], total_price_area_by_town['area'][idx])
		event_x, event_y = toPixel(scatterAxes, event.xdata, event.ydata)

		# Get the index that is the closest
		distanceToMouse = euclideanDistance(event_x, event_y, pixel_x, pixel_y)
		if (distanceToMouse < closest_dist):
			closest_dist = distanceToMouse
			closest = idx

	return closest

def updateScatterAnnot(label, x, y):

	# Update position
	scatterAnnot.xy = (x, y)

	# Format annotation text
	total_sales = dollar(y, "")
	total_area = metersquared(x, "")
	price_per_area = "{:1.1f}".format(y/x)

	scatterAnnot.set_text(
		f"{label}\n"+
		f"Total of sales: {total_sales}\n"+
		f"Total of area: {total_area}\n"+
		f"Property value: ${price_per_area} / m²\n"
		"click to show on map"
	)

def updateScatterColors(idx):
	global clickedScatterIndex

	# XOR True to flip the boolean value
	clickedScatterIndex[idx] ^= True
	
	colorArr = []
	for i in range(len(clickedScatterIndex)):
		colorArr.append(
			getColor(total_price_area_by_town['price'][i], total_price_area_by_town['area'][i], clickedScatterIndex[i])
		)
	scatterplot.set_color(colorArr)

def updateBarText(town):
	barTownText.set_text(
		town
	)
	barUnitsText.set_text(
		f"{completion_status_townDict[max_year][town]['under construction']} units under construction\n"+
		f"as of {max_year}\n"
	)

# Given a list of towns(planing areas), highlight
# them on the map with a different color
def updateMap(towns):
	for patch in patches:
		if patch.get_label() in towns:
			patch.set_facecolor(colors['map_selected'])
			patch.set_edgecolor(colors['map_selected_edge'])
		else:
			patch.set_facecolor(colors['map_unselected'])
			patch.set_edgecolor(colors['map_unselected_edge'])
	
# Update the hover annotations on the geometric map
def updateMapAnnotation(x, y, town):
	mapAnnot.set_text(
		f"{town}"
	)

	mapAnnot.set_fontfamily('monospace')
	mapAnnot.set_fontsize('small')
	mapAnnot.set_color(colors['map_selected'])

	mapAnnot.xy = (x, y)
	mapAnnot.set_visible(True)

# Given a list of town indexes, we want to display
# all those towns
def displayResult(town_indexes):
	global result_box

	# Remove all current results
	[box[0].remove() for box in result_box]

	# Hide all result texts by default
	[single_text.set_visible(False) for single_text in result_text]

	# We clear the result box
	result_box = []

	# Display the towns from the town_indexes we are given
	# This includes displaying the town names and the boxes 
	# that encase them
	# NOTE 'i' is the index of the town_indexes array whereas
	# town_idx is the index of that town from total_price_area_by_town['town]. 
	for i, town_idx in enumerate(town_indexes):
		
		town_name = total_price_area_by_town['town'][town_idx]

		# Note that this is 0 index
		updateResultText(i, town_name, clickedScatterIndex[town_idx])
		result_text[i].set_visible(True)

		# Note that generateBox is 1-indexed
		temp_box = generateBox(i+1)
		box = parameterAxes.fill(
			*temp_box.exterior.xy,
			color=colors['result_box_bg_blue'] if (clickedScatterIndex[town_idx]) else colors['result_box_bg_grey'],
			edgecolor=colors['result_box_edge'],
		)

		result_box.append(box)

# Given a string, we will search for that town name.
# If there are results, will display the results
# of our search.
def updateParameterSearch(textstring):

	# This list contains town names that include the
	# user's search term
	final_towns = []

	# Go through all towns and see if the search term
	# is inside a town name
	for i, town in enumerate(total_price_area_by_town['town']):

		if (len(textstring) > 0):
			# NOTE searching is NOT case sensitive
			if (textstring.lower() in town.lower()): 
				final_towns.append(i)

		# There is not enough space to display more than 4 towns
		if (len(final_towns) == 4):
			break

	if (len(final_towns) > 0):
		parameterResultClickGuideText.set_visible(True)
	else:
		parameterResultClickGuideText.set_visible(False)

	# Finally we display the results of our search
	displayResult(final_towns)
			

def onkey(event):

	# We only allow the user to type if the parameter is
	# 'in focus'; user must click on the parameter first
	# to begin typing.
	if (allowTyping):

		# Apply the user's pressed key
		parameterText.set_visible(True)
		parameterText.set_text(
			addToText(parameterText.get_text(), event.key)
		)

		if (len(parameterText.get_text()) == 0):
			parameterTypeGuideText.set_visible(True)
		else:
			parameterTypeGuideText.set_visible(False)

		# Search for town names given the user's new
		# search term
		updateParameterSearch(parameterText.get_text())

	fig.canvas.draw_idle()


def onclick(event):
	global allTowns, allowTyping, colorbar

	# If user clicks somewhere else other
	# than the parameter axes, we will
	# display the guides again
	if (event.inaxes != parameterAxes):

		allowTyping = False
		parameterClickGuideText.set_visible(True)
		parameterTypeGuideText.set_visible(False)
		updateParameterSearch("")
		parameterText.set_text("")

	# If user clicks in bar axes
	if (event.inaxes == barAxes):
		for i, patch in enumerate(barchart):
			if (patch.contains_point([event.x, event.y])):
				
				townname = under_construction['town'][i]
				updateBarText(townname)
				updateMap([townname])

	# If user clicks in scatter axes
	elif (event.inaxes == scatterAxes):

		containsMouse, point_dict = scatterplot.contains(event)

		if (containsMouse):

			# Of all the points that contain the mouse, choose
			# the index that points to the data point closest
			# to the mouse
			idx = getClosestIndex(event, point_dict['ind'])

			point_label = total_price_area_by_town['town'][idx]
			point_y = total_price_area_by_town['price'][idx]
			point_x = total_price_area_by_town['area'][idx]

			# Update the colors on the scatterplot
			updateScatterColors(idx)
			scatterAnnot.set_visible(True)

			# Update the map
			updateMap(total_price_area_by_town['town'][clickedScatterIndex])

			# Update the scatter annotation to the newest point
			updateScatterAnnot(point_label, point_x, point_y)
		else:
			scatterAnnot.set_visible(False)

	# If user clicks in tabs axes
	elif (event.inaxes == tabsAxes):
		if barTab.contains_point([event.x, event.y]):
			if (barAxes.get_visible() == False):

				# Changing tab colors
				barTab.set_color(colors['tab'])
				scatterTab.set_color(colors['not_tab'])

				# Displaying the correct axes
				barAxes.set_visible(True)
				barTextAxes.set_visible(True)
				scatterAxes.set_visible(False)
				parameterAxes.set_visible(False)
				colorbar.remove()

				# Changing tab text font
				barTabText.set_fontweight('bold')
				scatterTabText.set_fontweight('light')

				# Updating all towns
				allTowns = set(np.unique(under_construction['town']))
				refreshMap()

		elif scatterTab.contains_point([event.x, event.y]):
			if (scatterAxes.get_visible() == False):

				# Changing tab colors
				barTab.set_color(colors['not_tab'])
				scatterTab.set_color(colors['tab'])

				# Displaying the correct axes
				barAxes.set_visible(False)
				barTextAxes.set_visible(False)
				scatterAxes.set_visible(True)
				parameterAxes.set_visible(True)
				makeColorBar()

				# Changing tab text font
				scatterTabText.set_fontweight('bold')
				barTabText.set_fontweight('light')

				# Updating all towns
				allTowns = set(np.unique(resale_prices['town']))
				refreshMap()

	# If user clicks in parameter axes
	elif (event.inaxes == parameterAxes):
		if (not allowTyping):
			if textBox[0].contains_point([event.x, event.y]):
				parameterClickGuideText.set_visible(False)
				parameterTypeGuideText.set_visible(True)
				allowTyping = True
		else:
			for i, box in enumerate(result_box):
				if box[0].contains_point([event.x, event.y]):

					townname = result_text[i].get_text()

					town_idx = list(total_price_area_by_town['town']).index(townname)

					# Update the colors on the scatterplot
					updateScatterColors(town_idx)

					# Show the annotation of the newly selected point
					total_price = total_price_area_by_town['price'][town_idx]
					total_area = total_price_area_by_town['area'][town_idx]
					updateScatterAnnot(townname, total_area, total_price)

					# If the town is selected because of this click, show
					# the annotation for that town. clickedScatterIndex is
					# a list of booleans and as such the code below is valid
					scatterAnnot.set_visible(clickedScatterIndex[town_idx])

					# Update the map
					updateMap(total_price_area_by_town['town'][clickedScatterIndex])

					# Refresh the colors of the result box
					updateParameterSearch(parameterText.get_text())
				
	fig.canvas.draw_idle()

def onmove(event):

	# If the user moves out of an axes,
	# hide the annotations of that axes
	if (event.inaxes != barAxes):
		barAnnot.set_visible(False)
	if (event.inaxes != scatterAxes):
		scatterAnnot.set_visible(False)
	if (event.inaxes != mapAxes):
		mapAnnot.set_visible(False)

	# If user hovers over barchart axes
	if (event.inaxes == barAxes):

		containsMouse = False
		for i, patch in enumerate(barchart):
			if (patch.contains_point([event.x, event.y])):

				containsMouse = True
				
				# Update the bar annotation
				townname = under_construction['town'][i]
				barAnnot.xy = event.xdata, event.ydata
				barAnnot.set_text(
					townname
				)
				barAnnot.set_visible(True)
		
		# If the mouse is not on any patch, hide the bar annotation
		if (not containsMouse): barAnnot.set_visible(False)

	# If user hovers over scatterplot axes
	elif (event.inaxes == scatterAxes):

		containsMouse, point_dict = scatterplot.contains(event)

		if (containsMouse):

			# Of all the points that contain the mouse, choose
			# the index that points to the data point closest
			# to the mouse
			idx = getClosestIndex(event, point_dict['ind'])

			point_label = total_price_area_by_town['town'][idx]
			point_y = total_price_area_by_town['price'][idx]
			point_x = total_price_area_by_town['area'][idx]

			scatterAnnot.set_visible(True)
			updateScatterAnnot(point_label, point_x, point_y)

		# If none of the data points are hovered,
		# we hide the annotation
		else:
			scatterAnnot.set_visible(False)

	# If user hovers over the map axes
	elif (event.inaxes == mapAxes):

		# Update the map annotation
		containsMouse = False
		for patch in patches:
			if patch.contains_point([event.x, event.y]):
				containsMouse = True
				updateMapAnnotation(event.xdata, event.ydata, patch.get_label())
		
		# If the mouse is not on any patch, hide the map annotation
		if (not containsMouse): mapAnnot.set_visible(False)
	fig.canvas.draw_idle()

fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
fig.canvas.mpl_connect('key_press_event', onkey)
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmove)

plt.show()