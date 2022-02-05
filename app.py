import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges
from networkx.algorithms import community
from bokeh.palettes import Blues8, Reds8, Purples8
from bokeh.palettes import Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.io import output_notebook
import streamlit as st

st.title('Network Visualization of Marvel Characters')
st.sidebar.title("Network View")
add_selectbox = st.sidebar.radio("",("1", "2", "3"))

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
adjusted_node_size = dict(map(lambda node: (node[0], box_cox_normalization(node[1]))
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
                  ("Modularity Color", "$color[swatch]:modularity_color")]

plot = figure(tooltips = HOVER_TOOLTIPS,
              tools = "pan,wheel_zoom,save,reset",
              active_scroll = 'wheel_zoom',
              x_range=Range1d(-10.1,10.1), 
              y_range=Range1d(-10.1,10.1),
              title = title)

network_graph = from_networkx(G, 
                              nx.spring_layout, 
                              scale=10, 
                              center=(0,0))

network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, 
                                           fill_color=color_by_this_attribute)

network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, 
                                           fill_color=node_highlight_color,
                                                line_width=2)

network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, 
                                           fill_color=node_highlight_color,
                                                line_width=2)

network_graph.edge_renderer.glyph = MultiLine(
                                                line_alpha=0.5,
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