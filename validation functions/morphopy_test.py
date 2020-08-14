#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:59:32 2020

@author: user
"""



# # To delete when installation works
import sys
sys.path.append("../")

from morphopy.computation import file_manager as fm
from morphopy.neurontree import NeuronTree as nt
import pandas as pd
import networkx as nx
import numpy as np


filename = '1to1pair_b_series_t1_seed_end_1_input_0_output.swc'
s_path = '../(49) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/train with 1e6 after here/TEST_inference_196446/'
N = fm.load_swc_file(s_path + filename)


#filename = '1to1pair_b_series_t1_eLOSTpairbbslnannot-005_reformated_all_ax.swc'
#s_path = '/media/user/storage/Data/(1) snake seg project/Traces files/swc files/reformated_all_ax/'
#N = fm.load_swc_file(s_path + filename)


import seaborn as sns
import matplotlib.pyplot as plt


# fig = plt.figure(figsize=(10,5))
# ax1 = plt.subplot(121)
# N.draw_2D(fig, ax=ax1, projection='xy')


# ax2= plt.subplot(122)
# N.draw_2D(fig, ax=ax2, projection='xz')


# fig = plt.figure(figsize=(10,5))
# ax1 = plt.subplot(121)
# N.draw_2D(fig, ax=ax1, projection='xy')


# ax2= plt.subplot(122)
# N.draw_2D(fig, ax=ax2, projection='xz')



from morphopy.neurontree.plotting import show_threeview

fig = plt.figure(figsize=(10,10))
show_threeview(N, fig)


"""
Plotting axon and dendrites and neurites independently

In the plot above, the axon is large and entangled with the dendrites, so you might want to look at axons and dendrites independently. This can be done through the get_axonal_tree() and get_dendritic_tree() methods.
"""

Axon = N.get_axonal_tree()
Dendrites = N.get_dendritic_tree()

dendrite_fig = plt.figure(figsize=(10,10))
show_threeview(Dendrites, dendrite_fig)


axon_fig = plt.figure(figsize=(10,10))
show_threeview(Axon, axon_fig)


"""
get_neurites() returns a list of all neurites that extend from the soma. This can be used, for example, to examine each neurite individually.
"""

# neurites = Dendrites.get_neurites()

# fig = plt.figure(figsize=(16,10))
# for k in range(len(neurites)):
#     ax = plt.subplot(2,4, k+1)
#     neurites[k].draw_2D(fig, ax, projection='xy')





"""
Compute Morphometric Statistics

MorphoPy offers a default selection of 28 single-valued morphometric statistics.
"""

from morphopy.computation.feature_presentation import compute_morphometric_statistics

# morph_wide = compute_morphometric_statistics(N)
# morph_wide['filename'] = filename
# morph_wide.set_index('filename')

morph_long = compute_morphometric_statistics(N, format='long')
morph_long['filename'] = filename
morph_long





"""

Computing Morphometric Distributions

MorphoPy offers a range of different morphometric distributions via the get_histogram(key, dist_measure=None, **kwargs) method. If no dist_measure is passed the distributions are typically one-dimensional. **kwargs allows to pass parameters to the numpy.histogramdd method.

Possible keys for statistics are:

- branch orders
- Strahler order
- branch angles
- path angles
- root angles
- thickness
- segment lengths
- path length to soma
- radial distance
"""



# show morphometric distributions for axon and dendrites separately. 
sns.set_context('notebook')
statistics = ['branch_order', 'strahler_order', 
              'branch_angle', 'path_angle', 'root_angle', 
              'thickness', 'segment_length', 'path_length', 'radial_dist']

hist_widths = [2,.2, 10, 10, 10, .05, 20, 80, 30]
limits = [35, 5, 180, 180, 180, 0.5, 600, 2500, 900]
A = Axon.get_topological_minor()
D = Dendrites.get_topological_minor()

plt.figure(figsize=(12,12))
k = 1
for stat, lim, w in zip(statistics,limits, hist_widths):
    plt.subplot(3,3,k)
    for Trees, c in [[(Axon, A), 'darkgreen'], [(Dendrites, D), 'darkgrey']]:
        if stat in ['segment_length', 'path_length', 'radial_dist']:
            bins = np.linspace(0,lim, 20)
        else:
            bins= np.linspace(0,lim, 10)
        if stat in ['branch_order', 'strahler_order', 'root_angle']:
            dist, edges = Trees[1].get_histogram(stat, bins=bins)
        else:    
            dist, edges = Trees[0].get_histogram(stat, bins=bins) # you can pass options to the histogram method

        
        plt.bar(edges[1:], dist, width=w, color=c, alpha=.7)
        sns.despine()
        xlabel = stat.replace("_", " ").capitalize()
        if xlabel.find('length') > -1 or xlabel.find('dist') > -1 or xlabel == 'Thickness':
            xlabel += ' (um)'
        elif xlabel.find('angle') > -1:
            xlabel += ' (deg)'
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')

    k+=1

plt.legend(['Axon', 'Dendrites'])
plt.suptitle('1-D morphometric distributions', weight='bold')




"""
All 1D distributions can also be queried with a distance function via the parameter dist_measure. This is a good way to see their spatial progression with distance to the soma, the returned distribution then becomes two-dimensional.

"""
    
dist_both, edges_both = Axon.get_histogram('branch_angle', dist_measure='radial_dist', density=True)
dist_ba , edges_ba = Axon.get_histogram('branch_angle', density=True)
dist_r , edges_r = Axon.get_histogram('radial_dist', density=True)

sns.set_context('talk')
plt.figure(figsize=(8,8))
ax1 = plt.subplot2grid((4, 4), (0, 1), rowspan=3, colspan=3)
ax2 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=1)
ax3 = plt.subplot2grid((4, 4), (3, 1), rowspan=1, colspan=3)


ax1.imshow(dist_both) 
ax1.set_xticks(range(len(edges_both[1])-1))
ax1.set_xticklabels('')
ax1.set_yticks(range(len(edges_both[0])-1))
ax1.set_yticklabels('')
ax1.invert_yaxis()

ax3.bar(edges_ba[1:], dist_ba, width=15)
sns.despine()
ax3.set_xlabel('Branch angle (deg)')

ax2.barh(edges_r[1:], dist_r, height=80)
ax2.set_ylabel('Radial distance (um)')





"""
Sholl intersection profiles

A special distribution not mentioned yet is the Sholl intersection profile. It counts how often a 2D projection of the neural arbors intersects with concentric circles of growing radius (idea developed here). The center is usually placed at the soma. In our implementation one can also choose to use the centroid of the arbors' convex hull as a center point.
"""



# from matplotlib.patches import Circle

# # get the sholl intersection profileof the xy projection using 10 circles and the soma as center. 
# counts, radii = Dendrites.get_sholl_intersection_profile(proj='xy', steps=10, centroid='soma')

# fig = plt.figure(figsize=(12,5))
# ax1 = plt.subplot(121)
# Dendrites.draw_2D(fig, projection='xy')
# sns.despine()
# ax1.set_xlabel('X (um)')
# ax1.set_ylabel('Y (um)')

# for r in radii:
    
#     circ = Circle((0,0),r, edgecolor='k', facecolor=None, fill=False, linewidth=1)
#     ax1.add_artist(circ)

# plt.title('Depiction of the Sholl intersection procedure')    

    
# ax2 = plt.subplot(122)
# ax2.bar(radii[:-1], counts, color='darkgrey', width=30)
# sns.despine()
# ax2.set_xlabel('Radial distance (um)')
# ax2.set_ylabel('#intersections')
# plt.title('Corresponding intersection profile')






"""
Computing the Persistence Diagram

By default MorphoPy implements four different distance functions for persistence diagrams: radial distance to soma, path length to soma, height to soma, and branch order (to be found in computations.persistence_functions).
"""
    
from morphopy.computation.persistence_functions import path_length, radial_distance, height, branch_order
from morphopy.computation.feature_presentation import get_persistence

filter_function = path_length

df = get_persistence(N.get_topological_minor(), f=filter_function) 
# we pass the topological minor here since persistence only operates on branch points anyways
df.head()


"""

However, one can also provide a custom distance function. It only needs to follow the form function(networkx.DiGraph(), node_id_end, node_id_start) and return the distance between start node and end node.
"""

import numpy as np
def custom_distance(G, u, v):
    """
    Returns a distance between nodes u and v, which both are part of the graph given in G.
    """
    if np.float(nx.__version__) < 2: 
        n = G.node[u]['pos']
        r = G.node[v]['pos']
    else:
        n = G.nodes[u]['pos']
        r = G.nodes[v]['pos']
    return np.dot(n - r, [0,0,1])



df = get_persistence(N.get_topological_minor(), f=custom_distance) 
# we pass the topological minor here since persistence only operates on branch points anyways
df.head()




