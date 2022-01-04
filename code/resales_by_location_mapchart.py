# ====================================
#	Produces a map that allows users to compare the number of resales
# 	by locations.

# 	NOTE geopandas is needed for this Python script. Please follow
# 	the instructions in geopanda_setup.docx if you do not have geopandas
#	set up.
# ====================================

import geopandas as gpd;
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shapely;
from shapely.geometry.multipolygon import MultiPolygon;
from matplotlib.widgets import Slider

# Ensuring that the current directory is correct
os.chdir(
	(os.path.abspath(os.path.dirname(__file__)))
)

# Reading files
resale_prices = np.genfromtxt('../datasets/resale_flat_prices_all.csv', dtype=[('month', 'U32'), ('town', 'U32'), ('floor_area_sqm', np.int32), ('price', np.int32)], skip_header=1, delimiter=',')

# =============
# 	Cleaning
# =============

# Change month column to quarter column
year_column = np.array(list(
	map(

		# Change from 2020-03 to 2020-Q1
		lambda x: x.split('-')[0],
		[row['month'] for row in resale_prices]
	)
))

# Create a new data frame with only the relevant columns: quarter, town and price
clean_resale_prices = np.array(
	list(zip(year_column, resale_prices['town'], resale_prices['price'])),
	dtype=[('year', 'U32'), ('town', 'U32'), ('price', np.int32)]
)

# Set of all town names
# NOTE: This will be used to check if a town is inside the HDB dataset. set
# is used due to it being more efficient than a list
allTowns = set(np.unique(clean_resale_prices['town']))

# ----- Number of Resales by Quarter By Town -----
# This 2d dictionary contains the total number of sales as
# the value, for a given key of 'quarter' and 'town'
resales_by_yearTownDict = {
	singleYear: dict() for singleYear in clean_resale_prices['year']
}

# Counting number of sales for each year and town
for row in clean_resale_prices:
	if resales_by_yearTownDict[row['year']].get(row['town']):
		resales_by_yearTownDict[row['year']][row['town']] += 1
	else:
		resales_by_yearTownDict[row['year']][row['town']] = 1

# This 2d list contains
# 	[<year>, <town>, <sales>]
# as its elements
resales_by_yearTown = []

# Filling up resales_by_quarterTown
for year in resales_by_yearTownDict.keys():
	for town in resales_by_yearTownDict[year].keys():
		resales_by_yearTown.append((year, town, resales_by_yearTownDict[year][town]))

resales_by_yearTown = np.array(resales_by_yearTown, dtype=[('year', np.int32), ('town', 'U32'), ('sales', np.int32)])


# -- Shape file ---
# Read the shapes file. This is retrieved from
# https://data.gov.sg/dataset/planning-area-census2010?resource_id=076de3a7-eaa6-426e-8331-557469dd7ce8
Singapore = gpd.read_file('../datasets/planning_area_shape/Planning_Area_Census2010.shp');

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
	"Number of resales by town",
	verticalalignment='center',
	fontsize='xx-large'
)
fig.set_size_inches(10, 6)
fig.set_dpi(120)

colors = {
	'sea_color': (127/255, 205/255, 187/255, 0.8),
	'map_edgecolor': (178/255, 178/255, 178/255, 1),
	'map_hoveredgecolor': (0,0,0),
	'sparkline_title_intro': '#BBB',
	'sparkline_color': '#888',
}

# ==========================
# 	Creating subplots
# ==========================

sliderAndSparklineSubplot = fig.subplots(5, 2, gridspec_kw={'width_ratios': [2.2, 1], 'height_ratios': [3, 0.25, 0.7, 4,1]})
colorBarSubplot = fig.subplots(3, 2, gridspec_kw={'width_ratios': [2.8, 1], 'height_ratios': [1, 7, 1]})
mapSubplot = fig.subplots(3, 2, gridspec_kw={'width_ratios': [2.8, 1], 'height_ratios': [1, 7, 1]})

# ==========================
# 	Creating the map
# ==========================

# Defining color map that is used
cmap = plt.cm.Blues

# This variable contains the year that is set
# by the slider parameter
yearParameter = 2020

# This list will contain all patches from the map
patches = []

# Given year and town, calculate what color
# should be returned for gradient
def getColorForMap(year, town):
	current_sales = (resales_by_yearTownDict[str(year)][town])

	# Note that the color is determined across all years.
	maximum_sales = max(resales_by_yearTown['sales'])
	return cmap(current_sales / maximum_sales)


# Hiding axes that we are not using.
for i, blankArea in enumerate(mapSubplot.flatten()):
	if (not(i == 2)):
		blankArea.set_visible(False)

mapAxes = mapSubplot[1][0]

for i in range(0, len(clean_Singapore['geometry'])):
	areaname = clean_Singapore['area'][i]
	item = clean_Singapore['geometry'][i]

	for geom in item.geoms:

		# We only fill areas that are part of the resale_flat_prices_all dataset.
		if (areaname in allTowns):
			patch, = mapAxes.fill(*geom.exterior.xy, label=str(areaname), color=getColorForMap(yearParameter, areaname), edgecolor=colors['map_edgecolor'], linewidth=0.8);
			patches.append(patch)
		# Any other area(such as inhabitable areas) are grayed out
		else:
			patch, = mapAxes.fill(*geom.exterior.xy, label='blank', color='#EEE', edgecolor=(0,0,0,0), linestyle="None", linewidth=0);


# --- Map hover annotation ---
mapAnnot = mapAxes.annotate(
	"",
	xy=(0, 0),
	xytext=(10, 10),
	textcoords="offset points",
	bbox=dict(boxstyle="round", facecolor='w'),
)

mapAnnot.set_visible(False)



mapAxes.axis('equal');

mapAxes.spines['top'].set_color((0,0,0,0))
mapAxes.spines['bottom'].set_color((0,0,0,0))
mapAxes.spines['left'].set_color((0,0,0,0))
mapAxes.spines['right'].set_color((0,0,0,0))

mapAxes.set_facecolor((0,0,0,0))

mapAxes.set_xticks([])
mapAxes.set_yticks([])

# ==========================
# 	Creating the colorbar
# ==========================

for i, blankArea in enumerate(colorBarSubplot.flatten()):
	if (not(i == 2)):
		blankArea.set_visible(False)
colorBarAxes = colorBarSubplot[1][0]

# We first make an image from the resales dataset. This image is used in producing the colorbar
colorBarImageData = resales_by_yearTown['sales'].reshape(len(resales_by_yearTown), 1)
colorBarImage = colorBarAxes.imshow(colorBarImageData, cmap='Blues', interpolation="None", origin='lower', vmin=0)
colorBarImage.set_visible(False)

fig.colorbar(
	colorBarImage,
	ax=colorBarAxes,
	location="left",
	fraction=0.015,
	anchor=(0.0,1),
	ticks=[0, max(resales_by_yearTown['sales'])],
	format=matplotlib.ticker.FormatStrFormatter("%d resales")
)

colorBarAxes.set_facecolor((0,0,0,0))

colorBarAxes.spines['top'].set_color((0,0,0,0))
colorBarAxes.spines['bottom'].set_color((0,0,0,0))
colorBarAxes.spines['left'].set_color((0,0,0,0))
colorBarAxes.spines['right'].set_color((0,0,0,0))

colorBarAxes.set_xticks([])
colorBarAxes.set_yticks([])

# ==========================
# 	Creating the slider 
# ==========================

for i, blankArea in enumerate(sliderAndSparklineSubplot.flatten()):
	if (not(i == 1 or i == 3 or i == 7)):
		blankArea.set_visible(False)

sliderTextAxes = sliderAndSparklineSubplot[0][1]
sliderAxes = sliderAndSparklineSubplot[1][1]
sparklineAxes = sliderAndSparklineSubplot[3][1]

year_slider = Slider(
	ax=sliderAxes,
	label='',
	valmin=min(resales_by_yearTown['year']),
	valmax=max(resales_by_yearTown['year']),
	valinit=max(resales_by_yearTown['year']),
	valstep=resales_by_yearTown['year']
)

year_slider.valtext.set_visible(False)

# This function when user moves the slider.
def updateSlider(val):
	global yearParameter
	yearParameter = val
	updateSliderText()
	updateMap()

	# NOTE that we need to draw here.
	fig.canvas.draw_idle()

year_slider.on_changed(updateSlider)

# ==========================
# 	Creating the slider text
# ==========================



sliderTextAxes.spines['top'].set_color((0,0,0,0))
sliderTextAxes.spines['bottom'].set_color((0,0,0,0))
sliderTextAxes.spines['left'].set_color((0,0,0,0))
sliderTextAxes.spines['right'].set_color((0,0,0,0))

sliderTitle = sliderTextAxes.text(
	0.5,
	0.2,
	"Year\n",
	horizontalalignment='center',
	verticalalignment='center',
	fontfamily='monospace',
	fontsize='xx-large',
	fontweight='medium'
)

sliderText = sliderTextAxes.text(
	0.5,
	0,
	"move the slider",
	horizontalalignment='center',
	verticalalignment='center',
	fontfamily='monospace',
	fontsize='xx-large',
	fontweight='light',
	fontstyle='italic',
	color=(0,0,0,0.4)
)

sliderTextAxes.set_xticks([])
sliderTextAxes.set_yticks([])

# ==========================
# 	Creating the sparkline
# ==========================

sparkline = sparklineAxes.plot(
	[],
	marker='',
	color=colors['sparkline_color']
)

# Hide ticks on load
sparklineAxes.set_xticks([])

sparklineAxes.set_title(
	"Click on a planning area\non the map\n<-",
	y=0.5,
	loc='center',
	fontdict={
		'color':colors['sparkline_title_intro']
	}
)

sparklineMinMarker = sparklineAxes.plot(
	[],
	marker='o',
	linestyle='',
	markersize=3.5,
	color=cmap(1/5),
	markeredgecolor='black'
)

sparklineMaxMarker = sparklineAxes.plot(
	[],
	marker='o',
	linestyle='',
	markersize=3.5,
	color=cmap(4/5),
	markeredgecolor='black'
)

sparklineMinText = sparklineAxes.text(
	0,
	0,
	'',
	verticalalignment='top'
)

sparklineMaxText = sparklineAxes.text(
	0,
	0,
	'',
	verticalalignment='top'
)

sparklineHoverAnnot = sparklineAxes.annotate(
	"",
	xy=(0, 0),
	xytext=(0, 10),
	textcoords="offset points",
	horizontalalignment='center',
	zorder=4,
	bbox=dict(boxstyle="round", facecolor='w'),
)

sparklineHoverMarker = sparklineAxes.plot(
	[],
	marker='x',
	linestyle='',
	markersize=5,
	color=(0,0,0,0),
	markeredgecolor='black'
)

sparklineAxes.spines['top'].set_color((0,0,0,0))
sparklineAxes.spines['bottom'].set_color((0,0,0,0.5))
sparklineAxes.spines['left'].set_color((0,0,0,0))
sparklineAxes.spines['right'].set_color((0,0,0,0))

sparklineAxes.set_yticks([])

# ==========================
# 	Adding interactivity
# ==========================

# 'cur' stores the information about the nearest point on the sparkline where
# the distance between the point and mouse is less than
# what is stored in the variable 'distanceLimit'
cur = []
# cur[0] stores the x coord of the point
# cur[1] stores the y coord of the point
# cur[2] stores the distance between the point and the mouse

# This variable stores the distance needed
# between the mouse and a point, for the
# point to be considered 'hovered'.
distanceLimit = 15

# Update the geometric map
def updateMap():
	for patch in patches: 
		patch.set_color(getColorForMap(yearParameter, patch.get_label()))
		patch.set_edgecolor(colors['map_edgecolor'])

# Update the hover annotations on the geometric map
def updateMapAnnotation(x, y, town):
	mapAnnot.set_text(
		f"{town}\n"+
		f"{resales_by_yearTownDict[str(yearParameter)][town]} resales in {yearParameter}"
	)
	mapAnnot.xy = (x, y)
	mapAnnot.set_visible(True)

# Update the text display that's above the slider
def updateSliderText():
	sliderText.set_text(f"{yearParameter}")
	sliderText.set_fontweight('bold')
	sliderText.set_fontstyle('normal')
	sliderText.set_color((0,0,0,1))

# Show the text when user hovers over a point in the
# sparkline graph
def sparklineHover(year_idx, sales):

	# Hover marker
	sparklineHoverMarker[0].set_data((year_idx, sales))
	sparklineHoverMarker[0].set_visible(True)

	# Hover annotations
	year = np.unique(resales_by_yearTown['year'])[int(year_idx)]
	sparklineHoverAnnot.set_text(
		f"{int(sales)} resales\n"+
		f"{year}"
	)
	sparklineHoverAnnot.xy = (year_idx, sales)
	sparklineHoverAnnot.set_visible(True)

# Update the sparkline graph when user clicks on a town
def updateSparklineClick(town):

	filtered_sales = resales_by_yearTown['sales'][resales_by_yearTown['town'] == town]
	filtered_years = resales_by_yearTown['year'][resales_by_yearTown['town'] == town]
	sparkline[0].set_xdata(np.arange(0, len(filtered_sales)))
	sparkline[0].set_ydata(filtered_sales)

	# Changing limits of sparkline
	sparklineAxes.set_xlim(-1,10)
	range_filtered_sales = max(filtered_sales) - min(filtered_sales)
	padding = range_filtered_sales / 5
	sparklineAxes.set_ylim(min(filtered_sales)-padding,max(filtered_sales)+padding)
	sparklineAxes.set_xticks([0, 9],labels=[min(filtered_years), max(filtered_years)])

	# Get the maximum and minimum points
	min_point = (list(filtered_sales).index(min(filtered_sales)), min(filtered_sales))
	max_point = (list(filtered_sales).index(max(filtered_sales)), max(filtered_sales))

	# Updating the text points with new coordinates
	text_displacement = range_filtered_sales / 12
	sparklineMinText.set_x(min_point[0])
	sparklineMinText.set_y(min_point[1]-text_displacement)
	sparklineMaxText.set_x(max_point[0])
	sparklineMaxText.set_y(max_point[1]-text_displacement)

	# Updating text points with new data
	sparklineMinText.set_text(f"{min_point[1]} resales")
	sparklineMaxText.set_text(f"{max_point[1]} resales")

	# Update the position of the markers
	sparklineMinMarker[0].set_data(min_point)
	sparklineMaxMarker[0].set_data(max_point)

	# Changing title for sparkline 
	sparklineAxes.set_title(town, y=0.9, loc='center', fontdict = {
		'color': cmap(4.5/5),
		'fontfamily': 'monospace',
		'fontweight': 'bold',
		'fontsize': 'large',
	})

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

# This function runs when button_press_event triggers
def onclick(event):

	# If user click is inside the map axes
	if (event.inaxes == mapAxes):

		# When user clicks on a town, update the
		# sparkline such that it displays that information
		# of that town
		for patch in patches:
			if patch.contains_point([event.x, event.y]):
				updateSparklineClick(patch.get_label())

	fig.canvas.draw_idle()
			
# This function runs when any mouse movement is detected
def onmove(event):

	# If user hovers over the map axes
	if (event.inaxes == mapAxes):

		mouseInPatch = False
		for patch in patches:
			if patch.contains_point([event.x, event.y]):
				patch.set_edgecolor(colors['map_hoveredgecolor'])

				# We need to set zorder to a higher number, as
				# if zorder is the same, the edge color of hovered
				# patches do not display properly
				patch.set_zorder(2)
				mouseInPatch = True
				updateMapAnnotation(event.xdata, event.ydata, patch.get_label())
			else:
				patch.set_edgecolor(colors['map_edgecolor'])
				patch.set_zorder(1)
		
		# If the mouse is not on any patch, hide the map annotation
		if (not mouseInPatch):
			mapAnnot.set_visible(False)

	# If user hovers over the sparkline axes
	elif (event.inaxes == sparklineAxes):

		global cur

		# We need to clear cur for every move,
		# else the hover point will stay stuck until
		# the distance between mouse and a point becomes even
		# smaller than the currently stored closest distance, cur[2]
		cur = []

		for i, [point_x, point_y] in enumerate(sparkline[0].get_xydata()):

			# Transform data coordinates to pixel coordinates
			pixel_point_x, pixel_point_y = toPixel(sparklineAxes, point_x, point_y)

			event_x, event_y = toPixel(sparklineAxes, event.xdata, event.ydata)

			distanceMouseAndPoint = euclideanDistance(pixel_point_x, pixel_point_y, event_x, event_y)
			if (distanceMouseAndPoint <= distanceLimit):

				# If there is a point that is closer to the mouse, update the current point
				# to that closer point
				if (len(cur) == 0 or cur[2] > distanceMouseAndPoint):
					cur = point_x, point_y, distanceMouseAndPoint

		# If there is a point that meets the distance limit, we will consider
		# that point to be hovered
		if (len(cur) != 0):

			# cur[0], cur[1] stores x and y data points, while cur[3] stores the town's name
			sparklineHover(cur[0], cur[1])
		else:

			# If there is no point that is close enough to the mouse,
			# we will make sure the hover effects are hidden
			sparklineHoverAnnot.set_visible(False)
			sparklineHoverMarker[0].set_visible(False)

	fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmove)

plt.show()


