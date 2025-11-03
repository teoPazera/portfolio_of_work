import json
from shapely.geometry import shape, Point
from shapely import to_geojson
from networkx.readwrite import json_graph
import networkx as nx
import folium


def graph_to_geojson(G, path) -> None:
    """
    Serialize NetworkX graph G (with Shapely geometries in node attrs)
    to a JSON file at 'path', converting geometries via to_geojson.
    """
    # 1. Extract node-link data
    data = json_graph.node_link_data(G)
    # 2. Replace each geometry attribute with a GeoJSON dict
    for node in data["nodes"]:
        geom = node.get("geometry")
        if geom is not None:
            # to_geojson returns a JSON string; parse it
            geojson_str = to_geojson(geom, indent=None)
            node["geometry"] = json.loads(geojson_str)
    # 3. Dump to file
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def graph_from_geojson(path, directed=False, multigraph=True) -> nx.Graph:
    """
    Read a JSON file at 'path' produced by graph_to_geojson and
    reconstruct the original NetworkX graph with Shapely geometries.
    """
    # 1. Load the raw node-link dict
    with open(path) as f:
        data = json.load(f)

    # 2. Convert each node's GeoJSON dict back into a Shapely geometry
    for node_obj in data.get("nodes", []):
        geom_dict = node_obj.get("geometry")
        if isinstance(geom_dict, dict) and "type" in geom_dict:
            node_obj["geometry"] = shape(geom_dict)  # inverse of to_geojson :contentReference[oaicite:1]{index=1}

    # 3. (Optional) If you stored edge geometries similarly, undo those too
    for edge_obj in data.get("links", data.get("edges", [])):
        geom_dict = edge_obj.get("geometry")
        if isinstance(geom_dict, dict) and "type" in geom_dict:
            edge_obj["geometry"] = shape(geom_dict)

    # 4. Rebuild the NetworkX graph (with all attrs, including restored geometries)
    G = json_graph.node_link_graph(
        data,
        directed=directed,
        multigraph=multigraph
    )  # rebuilds Graph from node-link format :contentReference[oaicite:2]{index=2}

    return G


def largest_connected_component(G: nx.Graph) -> nx.Graph:
    # keep only the largest component of G
    components = nx.connected_components(G)
    largest_component_nodes = max(components, key=len)
    return G.subgraph(largest_component_nodes).copy()

def graph_to_folium(G: nx.Graph, country: str) -> folium.Map:
    # visualizes G on the map
    geographical_centers = {
        "svk": [48.7, 19.5],
        "cze": [49.75, 15.5]
    }

    m = folium.Map(location=geographical_centers[country], zoom_start=7)

    for node1, node2 in G.edges():
        if node1 in G.nodes and node2 in G.nodes and 'geometry' in G.nodes[node1] and 'geometry' in G.nodes[node2]:
            geom1 = G.nodes[node1]['geometry']
            geom2 = G.nodes[node2]['geometry']
            if isinstance(geom1, Point) and isinstance(geom2, Point):
                lat1, lon1 = geom1.y, geom1.x
                lat2, lon2 = geom2.y, geom2.x
                folium.PolyLine([[lat1, lon1], [lat2, lon2]], color='blue', weight=1).add_to(m)
            else:
                print(f"Warning: Skipping edge between {node1} and {node2} due to non-Point geometry.")
        else:
            print(f"Warning: Skipping edge between {node1} and {node2} because one or both nodes are missing geometry information.")

    # Mark the nodes (stations) with red dots
    for _, data in G.nodes(data=True):
        if 'geometry' in data and isinstance(data['geometry'], Point):
            folium.CircleMarker(
                location=[data['geometry'].y, data['geometry'].x],
                radius=1,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7
            ).add_to(m)
    
    return m
