import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges
from networkx.algorithms import community
from bokeh.palettes import Blues8, Reds8, Purples8
from bokeh.palettes import Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine,HoverTool, BoxZoomTool, ResetTool, PanTool, Plot
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.io import output_notebook
import streamlit as st
import folium
import numpy as np
import geopandas as gpd
from shapely.ops import cascaded_union
from geovoronoi import voronoi_regions_from_coords, points_to_coords
from streamlit_folium import folium_static

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Network Visualization of Marvel Characters')
st.sidebar.title("Network View")
add_selectbox = st.sidebar.radio("",("Marvel Network", "Karate Network", "Facebook Network", "QC Voronoi Plot"))

if add_selectbox == "Marvel Network":
    sb_communities = st.selectbox("Communities",(0, 1, 2, 3))
    marvel_df = pd.read_csv('marvel-unimodal-edges.csv')

    G = nx.from_pandas_edgelist(marvel_df, 'Source', 'Target', 'Weight')

    ### responsive highlighting

    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name='degree', values=degrees)

    ### box-cox
    size = 0.5
    l = 0.5

    def box_cox_normalization(node_size):
        from math import ceil
        from math import pow

        compressed_point = (pow(node_size, l) - 1) / l
        return ceil(size*compressed_point)

    # number_to_adjust_by = 0.5
    adjusted_node_size = dict(map(lambda node: (node[0],    box_cox_normalization(node[1]))
                         , dict(G.degree).items()))

    nx.set_node_attributes(G, name='adjusted_node_size',
                                values=adjusted_node_size)

    communities = community.greedy_modularity_communities(G)
    modularity_class = {}
    modularity_color = {}

    for community_number, community in enumerate(communities):
        for name in community:
            modularity_class[name] = community_number
            modularity_color[name] = Spectral8[community_number]

    for (name, community_number) in modularity_class.items():
        if community_number == sb_communities:
            modularity_class[name] = community_number

    filtered_dict = {k:v for (k,v) in modularity_class.items() if   v==sb_communities}

    nx.set_node_attributes(G, modularity_class, 'modularity_class')
    nx.set_node_attributes(G, modularity_color, 'modularity_color')

    node_highlight_color = 'white'
    edge_highlight_color = 'black'

    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'modularity_color'

    color_palette = Blues8

    title = 'Marvel Network'

    HOVER_TOOLTIPS = [("Character", "@index"),
                      ("Degree", "@degree"),
                      ("Modularity Class", "@modularity_class"),
                      ("Modularity Color", "$color[swatch]  :modularity_color")]

    plot = figure(tooltips = HOVER_TOOLTIPS,
                  tools = "pan,wheel_zoom,save,reset",
                  active_scroll = 'wheel_zoom',
                  x_range=Range1d(-10.1,10.1), 
                  y_range=Range1d(-10.1,10.1),
                  title = title)

    subnetwork = G.subgraph(filtered_dict)

    network_graph = from_networkx(subnetwork, 
                                  nx.spring_layout, 
                                  scale=10, 
                                  center=(0,0))

    network_graph.node_renderer.glyph = Circle  (size=size_by_this_attribute, fill_color=color_by_this_attribute)

    network_graph.node_renderer.hover_glyph = Circle    (size=size_by_this_attribute, fill_color=node_highlight_color,line_width=2)

    network_graph.node_renderer.selection_glyph = Circle    (size=size_by_this_attribute, fill_color=node_highlight_color,
    line_width=2)

    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5,
    line_width=1)

    network_graph.edge_renderer.hover_glyph = MultiLine(
    line_color=edge_highlight_color,
    line_width=2)

    network_graph.edge_renderer.selection_glyph = MultiLine(
    line_color=edge_highlight_color,
    line_width=2)

    network_graph.selection_policy = NodesAndLinkedEdges()

    network_graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(network_graph)
    st.bokeh_chart(plot)

elif add_selectbox == "Karate Network":
    sb_color = st.selectbox("Colors",('r', 'b', 'g'))
    G = nx.karate_club_graph()
    
    node_color = dict()
    communities = nx.community.girvan_newman(G)
    next(communities)
    community = next(communities)
    for node in community[0]:
        node_color[node] = 'r'

    for node in community[1]:
        node_color[node] = 'b'

    for node in community[2]:
        node_color[node] = 'g'

    #for node in community[3]:
    #    node_color[node] = 'c'

    node_color = [node_color[node] for node in sorted(node_color)]

    # filtered_list = [color for color in node_color if color==sb_color]
    
    dictionary = {}
    for (number, color) in enumerate(node_color):
        dictionary[number] = color

    filt_dict = {k:v for (k,v) in dictionary.items() if v==sb_color}
    subnetwork = G.subgraph(filt_dict)
    pos = nx.spring_layout(subnetwork)
    fig = nx.draw(subnetwork, pos=pos, node_color = sb_color, with_labels=True)
    st.pyplot(fig)

elif add_selectbox == "Facebook Network":

    sb_communities = st.selectbox("Communities",('blue', 'orange', 'red'))
    G = nx.read_adjlist('facebook_social_graph.adjlist')

    size = 0.5
    l = 0.5

    def box_cox_normalization(node_size):
        from math import ceil
        from math import pow
    
        compressed_point = (pow(node_size, l) - 1) / l
        return ceil(size*compressed_point)

    def get_influence_colour(num_connections):
        if num_connections > 500:
            return 'red'
        elif 50 < num_connections < 500:
            return 'orange'
        else:
            return 'blue'
    
    new_sizes = dict(map(lambda node: (node[0], box_cox_normalization(node[1])), dict(G.degree).items()))
    colors = dict(map(lambda node: (node[0], get_influence_colour(node[1])) , dict(G.degree).items()))

    filtered_dict = {k:v for (k,v) in colors.items() if   v==sb_communities}

    nx.set_node_attributes(G, dict(G.degree), 'connections')
    nx.set_node_attributes(G, new_sizes, 'node_size')
    nx.set_node_attributes(G, colors, 'node_color')

    plot = Plot(plot_width=700, plot_height=500,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

    plot.title.text = "Facebook Social Network"

    plot.add_tools(PanTool(), BoxZoomTool(), ResetTool(), HoverTool (tooltips=[("connections", "@connections")]))

    subnetwork = G.subgraph(filtered_dict)

    graph_renderer = from_networkx(subnetwork, nx.spring_layout, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size='node_size',   fill_color='node_color')

    graph_renderer.edge_renderer.glyph = MultiLine(
        line_alpha=0.4, 
        line_width=0.4
    )

    plot.renderers.append(graph_renderer)
    st.bokeh_chart(plot)

else:

    ph_2 = gpd.read_file('./gadm36_PHL_shp/gadm36_PHL_2.shp')
    df_final_point = pd.read_csv("df_final_point.csv")
    df_final_point['geometry'] = gpd.GeoSeries.from_wkt(df_final_point['geometry'])
    qc_area = ph_2[ph_2['NAME_2'] == 'Quezon City']
    boundary_shape = cascaded_union(qc_area.geometry)
    coords = points_to_coords(df_final_point.geometry)

    poly_shapes, poly_to_pt_assignments = voronoi_regions_from_coords(coords, boundary_shape)

    m = folium.Map([14.67428, 121.05750], zoom_start=11, tiles='cartodbpositron')

    #draw the voronoi diagram within coverage area
    for x in range(len(poly_shapes)):
        folium.GeoJson(poly_shapes[x]).add_to(m)

    #draw the data points
    points = [[geom.xy[1][0], geom.xy[0][0]] for geom in df_final_point.geometry]
    locs = points
    for location in locs:
        folium.CircleMarker(location=location, 
        color = "#4925a2", radius=0.01).add_to(m)

    folium_static(m)

    
