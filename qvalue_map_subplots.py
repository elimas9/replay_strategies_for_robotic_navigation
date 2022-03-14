import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.spatial as scispa
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib_latex_bridge as mlb
import matplotlib
import json

mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=11)


deterministic_case = False  # choose to plot the determistic case or the stochastic one


def create_voronoid(ax=None, replay_type=None):
	"""
	Plots on the "ax" axes the voronoi structure of the environment fill with the "replay_type" max Q-values for
	each state.

	"ax" must be the matplotlib.pyplot.figure.subplots axes for the figure.
	"replay_type" must be a string describing thr replay_type, i.e. the key of the dictionary contained the
	q-values.

	Returns the plt.cm.ScalarMappable instance plotted on the figure.
	"""
	# coordinates of the centre of all the states
	states_ = [(0.00320002, 0.0059351), (0.310727, 0.0241474), (0.593997, 0.152759), (0.724483, -0.118826),
			   (0.956215, 0.0761724), (-0.0546479, 0.308712), (-0.359653, 0.3537), (-0.465333, 0.651306),
			   (-0.636187 ,0.920479), (-0.325808 ,0.96138), (-0.0262775, 0.936864), (-0.12853 ,0.608485),
			   (0.275655, 0.947168), (0.164439 ,1.22664), (-0.588397, 1.21822), (-0.779392, 0.652651),
			   (-0.669064, 0.253532), (-0.863761 ,0.0191329), (-0.924563, -0.277856), (-1.08771, 0.22059),
			   (-1.23553, 0.490936), (-1.09893 ,0.758705), (-0.293732, 1.30145), (0.208853, 0.454286),
			   (0.522142, 0.755471), (0.69987, 0.513679), (-0.0588719, -0.289781), ( 0.0889117 ,-0.583929),
			   (-0.120961, -0.804133), (-0.939397, 1.014), ( 0.367168, -0.274708), ( -0.329299 ,-0.156936),
			   (0.39444, -0.660872), (-0.539525, -0.381784), (-1.22956, -0.263396), (-0.504464 ,-0.834182)]

	if deterministic_case:
		# open the file saving the Q values of trial 3, individual 50, of the deterministic case
		with open("data_files/q_values_detTrue_ind100_alpha0.78_trial_3_ind_50.json", "rb") as qvs:
			q_values_det_sing = json.load(qvs)
	else:
		# open the file saving the Q values of trial 3, individual 50, of the stochastic case
		with open("data_files/q_values_detFalse_ind100_alpha0.78_trial_3_ind_50.json", "rb") as qvs:
			q_values_det_sing = json.load(qvs)

	# save the maximum values of these q-values just opened for normalisation later
	max_q_value = {}
	for tr in q_values_det_sing.keys():  # replay types
		max_q_value[tr] = {}
		for st in range(len(q_values_det_sing[tr])):  # states
			max_q_value[tr][str(st)] = max(q_values_det_sing[tr][st])

	centre_states = np.array(states_)
	centre_states = np.append(centre_states, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)  # add 4
	# distant dummy points

	# create Voronoi
	vor = scispa.Voronoi(centre_states)

	# Normalize the max Q-value for each state
	qvalues = max_q_value[replay_type]
	qvalues_min = 0
	qvalues_max = 1
	normalizer = matplotlib.colors.Normalize(vmin=qvalues_min, vmax=qvalues_max)

	# Fill each state with a color corresponding to the normalized Q-value
	for r in range(len(vor.point_region)):
		region = vor.regions[vor.point_region[r]]
		if not -1 in region:
			polygon = [vor.vertices[i] for i in region]
			ax.fill(*zip(*polygon), color=plt.cm.viridis(normalizer(qvalues[str(r)])), zorder=0)
	
	# states and legend
	plot_states = np.array(states_)
	ax.scatter(plot_states[35, 0], plot_states[35, 1], c="green", label="35 - initial state")
	ax.scatter(plot_states[22, 0], plot_states[22, 1], c="purple", label="22 - first reward state")
	ax.scatter(plot_states[4, 0], plot_states[4, 1], c="orange", label="4 - second reward state")

	index_state = []
	for i in range(0, 36):
		if i != 4 and i != 22 and i != 35:
			index_state.append(i)

	return plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=normalizer)


def mask_outside_polygon(poly_verts, ax=None):
	"""
	Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
	all areas outside of the polygon specified by "poly_verts" are masked.  

	"poly_verts" must be a list of tuples of the verticies in the polygon in
	counter-clockwise order.

	Returns the matplotlib.patches.PathPatch instance plotted on the figure.
	"""
	if ax is None:
		ax = plt.gca()

	# Get current plot limits
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	# Verticies of the plot boundaries in clockwise order
	bound_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]), (xlim[1], ylim[1]), (xlim[1], ylim[0]), (xlim[0], ylim[0])]

	# A series of codes (1 and 2) to tell matplotlib whether to draw a line or 
	# move the "pen" (So that there's no connecting line)
	bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
	poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

	# Plot the masking patch
	path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
	patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none', zorder=2)
	patch = ax.add_patch(patch)

	# Reset the plot limits to their original extents
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	return patch


def create_map(map_path="map1.pgm", scale=1.0, offset=np.zeros(2), ax=None):
	"""
	Plots the map of the environment, given the contour "map_path" .pgm, the "scale" and the "offset", on "ax" axes.
	"""

	img = mpimg.imread(map_path)

	# get list of coordinates for the border
	coords = []
	h, w = img.shape
	for i in range(h):
		for j in range(w):
			if img[i,j] == 0:
				coords.append((i*scale, j*scale))
	coords = np.array(coords)

	# center data and adjust offset
	coords[:, 0] -= np.mean(coords[:, 0])
	coords[:, 1] -= np.mean(coords[:, 1])
	coords = coords + offset

	# compute the contour
	hull = ConvexHull(coords)
	hull_points_x = [coords[v, 0] for v in hull.vertices]
	hull_points_x.append(coords[hull.vertices[0], 0])
	hull_points_y = [coords[v, 1] for v in hull.vertices]
	hull_points_y.append(coords[hull.vertices[0], 1])

	# draw contour and white patch outside
	hull_points = list(zip(hull_points_x, hull_points_y))
	mask_outside_polygon(hull_points, ax=ax)
	ax.plot(hull_points_x, hull_points_y, 'k-')
	ax.axis("equal")
	ax.axis("off")


def main():
	# set the relay types
	replay_types = ["no replay", "backward replay", "random replay"]
	replay_titles = ["MF-RL no replay", "MF-RL backward replay", "MF-RL shuffled replay"]

	# set the figure parameters
	fig = mlb.figure_textwidth(height=2)
	axs = fig.subplots(1, 3)

	# for each replay type, plot the figure
	for idx, rt in enumerate(replay_types):
		ax = axs[idx]
		colormap = create_voronoid(ax=ax, replay_type=rt)
		create_map(map_path="data_files/map1.pgm", scale=0.08, offset=np.array([-0.2, 0.2]), ax=ax)
		ax.set_title(replay_titles[idx])

		# fix range of plot
		ax.set_xlim(- 2, 1.5)
		ax.set_ylim(- 1.5, 2)

	# set a common colorbar
	cbar = fig.colorbar(colormap, ax=axs)
	cbar.set_label('maximum Q-value\nin each state (a.u.)')

	plt.show()


if __name__ == '__main__':
	main()
