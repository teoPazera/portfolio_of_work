import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps
import matplotlib.colors as colors
import networkx as nx
import numpy as np
import seaborn as sns
import sqlite3
from src.util import graph_from_geojson, largest_connected_component, graph_to_folium
from shapely.geometry import shape, Point
import folium
import warnings


def normalized_kirchhoff_index(G: nx.Graph) -> float:
    # kirchhoff index is infinity for disconnected graphs
    if not nx.is_connected(G):
        return float("inf")

    # since distance is related to resistance, we need its inverse for Laplacian
    H: nx.Graph = G.copy()
    for _, _, d in H.edges(data=True):
        d["conductance"] = 1.0 / (d["distance"] / 1000)

    # compute Laplacian using conductance as weight
    L = nx.laplacian_matrix(H, weight="conductance").todense()
    eigenvalues = np.linalg.eigvalsh(L)

    # get nonzero eigenvalues
    nonzero_eigenvalues = np.sort(eigenvalues)[1:]

    # compute the normalized kirchhoff coefficient
    n = H.number_of_nodes()
    return (2 / (n - 1)) * np.sum(1.0 / nonzero_eigenvalues)


def effective_edge_resistance_centrality(u: str, v: str, G: nx.Graph, k_G: float) -> float:
    # remove u-v edge
    G_tmp: nx.Graph = G.copy()
    G_tmp.remove_edge(u, v)

    # compute kirchhoff coefficient for altered graph
    k: float = normalized_kirchhoff_index(G_tmp) 

    # return resistance centrality
    if k == float("inf"):
        return float("inf")
    return (k - k_G) / k_G


def effective_vertex_resistance_centrality(n: str, G: nx.Graph, k_G: float) -> float:
    # remove node n (and all incident edges)
    G_tmp: nx.Graph = G.copy()
    G_tmp.remove_node(n)

    # compute kirchhoff coefficient for altered graph
    k: float = normalized_kirchhoff_index(G_tmp) 

    # return resistance centrality
    if k == float("inf"):
        return float("inf")
    return (k - k_G) / k_G


def main() -> None:
    # load railway networks, keep only the largest component
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        G_svk: nx.Graph = largest_connected_component(
            graph_from_geojson("graphs/svk-railroad-network.json")
            )
        G_cze: nx.Graph = largest_connected_component(
            graph_from_geojson("graphs/cze-railroad-network.json")
        )
    
    # compute kirchhoff indexes
    k_G_svk: float = normalized_kirchhoff_index(G_svk)
    k_G_cze: float = normalized_kirchhoff_index(G_cze)
    print(f"Kirchhoff coefficient of Slovak railway network: {k_G_svk}")    # 92.29
    print(f"Kirchhoff coefficient of Czech railway network: {k_G_cze}")     # 65.99
    
    # get the distributions of kirchhoff indexes of randomized networks
    with sqlite3.connect("src/random_networks_data/networks.sqlite") as conn:
        cur: sqlite3.Cursor = conn.cursor()
        k_G_svk_random: list[float] = [
            r[0] for r in cur.execute(
                "SELECT kirchhoff FROM graphs WHERE country = \"svk\""
                ).fetchall()
            ]
        k_G_cze_random: list[float] = [
            r[0] for r in cur.execute(
                "SELECT kirchhoff FROM graphs WHERE country = \"cze\""
                ).fetchall()
            ]
    
    # plot the average kirchhoff index of randomized networks 
    # as well as percentage of greater indexes from randomized networks
    print(f"Kirchhoff coefficient of randomized Slovak railway network: {np.mean(k_G_svk_random)}")     # 96.35
    print(f"Kirchhoff coefficient of randomized Czech railway network: {np.mean(k_G_cze_random)}")      # 67.16
    p_svk: float = sum(
        [1 if k_G_svk < k_G_svk_random[i] else 0 for i in range(len(k_G_svk_random))]
        ) / len(k_G_svk_random)
    p_cze: float = sum(
        [1 if k_G_cze < k_G_cze_random[i] else 0 for i in range(len(k_G_cze_random))]
        ) / len(k_G_cze_random)
    print(f"Percentage of randomized Slovak networks with greater Kirchhoff coefficient that the actual one: {p_svk}")  # 0.99
    print(f"Percentage of randomized Czech networks with greater Kirchhoff coefficient that the actual one: {p_cze}")   # 0.97
    
    # plot the KDEs of these distributions along with the values of the actual network
    fig, ax = plt.subplots()
    sns.kdeplot(k_G_svk_random, color="tab:blue", label=r"svk: rozdelenie $\mathit{Kf}_n(G)$ modifikovaných sietí", ax=ax)
    sns.kdeplot(k_G_cze_random, color="tab:orange", label=r"cze: rozdelenie $\mathit{Kf}_n(G)$ modifikovaných sietí", ax=ax)
    ax.axvline(k_G_svk, linestyle="--", color="tab:blue", label=r"svk: skutočný $\mathit{Kf}_n(G)$")
    ax.axvline(k_G_cze, linestyle="--", color="tab:orange", label=r"cze: skutočný $\mathit{Kf}_n(G)$")

    ax.legend()
    ax.set_xlabel("Kirchhoffov index")
    ax.set_ylabel("hustota")
    plt.tight_layout()
    plt.savefig("src/visualizations/kde_kirchhoff_randomized.png")
    plt.close(fig)

    # open the figure for plotting distribution of centralities
    fig, ax = plt.subplots()

    # compute vertex resistance centrality for both networks, visualize it
    for G, country, k_G, color in zip(
        [G_svk, G_cze], ["svk", "cze"], [k_G_svk, k_G_cze], ["tab:blue", "tab:orange"]
        ):  
        resistance_centrality = {}
        
        # compute the resistance centrality for each node
        for n in G.nodes():
            k = effective_vertex_resistance_centrality(n, G, k_G)

            # if resistance centrality is infinity (disconnects graph), do not store it, will be visualized differently
            if k != float("inf"):
                resistance_centrality[n] = k

        # plot the distributions of centralities
        ax.hist(list(resistance_centrality.values()), bins=10, density=True, color=color,
                 label=rf"{country}: distribúcia $R(v_i, G)$", alpha=0.7)
        ax.axvline(
            np.mean(list(resistance_centrality.values())), color=color,
            label=rf"{country}: priemerná $R(v_i, G)$", linestyle="--"
            )

        # map the values to colors
        norm: colors.Normalize = colors.Normalize(
            vmin=min(resistance_centrality.values()), vmax=max(resistance_centrality.values())
            )
        cmap: colors.Colormap = colormaps["inferno"]
        color_dict: dict[str, str] = {
            node: colors.to_hex(cmap(norm(value))) 
            for node, value in resistance_centrality.items()
            }

        # plot the map, overwrite the drawn nodes
        m: folium.Map = graph_to_folium(G, country)
        for node, data in G.nodes(data=True):
            if node not in resistance_centrality:  # node has infinite resistance centrality, plot blue dot
                folium.CircleMarker(
                    location=[data["geometry"].y, data["geometry"].x], radius=1, 
                    color="#00BFFF", fill=True, fill_color="#00BFFF", fill_opacity=1
                ).add_to(m)
            else:           # color the node according to its centrality
                folium.CircleMarker(
                    location=[data["geometry"].y, data["geometry"].x], radius=3,
                    color=color_dict[node], fill=True, fill_color=color_dict[node], fill_opacity=1
                ).add_to(m)

        # save the centrality colored map
        m.save(f"src/visualizations/{country}_vertex_resistance.html")
    
    # save the visualized distributions
    ax.legend()
    ax.set_xlabel("$R(v_i, G)$")
    ax.set_ylabel("hustota")
    plt.savefig("src/visualizations/vertex_resistance_distribution.png")
    plt.close(fig)

    # open the figure for plotting distribution of centralities
    fig, ax = plt.subplots()
    
    # compute edge resistance centrality for both networks, visualize it
    for G, country, k_G, color in zip(
        [G_svk, G_cze], ["svk", "cze"], [k_G_svk, k_G_cze], ["tab:blue", "tab:orange"]
        ):      
        resistance_centrality = {}

        # compute centrality for all edges
        for u, v in G.edges():
            k = effective_edge_resistance_centrality(u, v, G, k_G)

            # if resistance centrality is infinity (disconnects graph), do not store it, will be visualized differently
            if k != float("inf"):
                resistance_centrality[(u, v)] = k
        
        # plot the distributions of centralities
        ax.hist(list(resistance_centrality.values()), bins=10, density=True, color=color,
                 label=rf"{country}: distribúcia $R(e_{{i,j}}, G)$", alpha=0.7)
        ax.axvline(
            np.mean(list(resistance_centrality.values())), color=color,
            label=rf"{country}: priemerná $R(e_{{i,j}}, G)$", linestyle="--"
            )
        
        # map the centralities to colors
        norm: colors.Normalize = colors.Normalize(
            vmin=min(resistance_centrality.values()), vmax=max(resistance_centrality.values())
            )
        cmap: colors.Colormap = colormaps["inferno"]        
        edge_color_dict: dict[tuple[str, str], str] = {
            edge: colors.to_hex(cmap(norm(value))) 
            for edge, value in resistance_centrality.items()
            }

        # plot the map, overwrite the drawn edges
        m = graph_to_folium(G, country)
        for u, v in G.edges():
            if (u, v) not in resistance_centrality:
                folium.PolyLine(
                    [[G.nodes[u]["geometry"].y, G.nodes[u]["geometry"].x],
                     [G.nodes[v]["geometry"].y, G.nodes[v]["geometry"].x]], 
                    color="#00BFFF", weight=1
                    ).add_to(m)
            else:
                folium.PolyLine(
                    [[G.nodes[u]["geometry"].y, G.nodes[u]["geometry"].x],
                     [G.nodes[v]["geometry"].y, G.nodes[v]["geometry"].x]], 
                    color=edge_color_dict[(u, v)], weight=3
                    ).add_to(m)

        m.save(f"src/visualizations/{country}_edge_resistance.html")  
    
    # save the visualized distributions
    ax.legend()
    ax.set_xlabel(r"$R(e_{i,j}, G)$")
    ax.set_ylabel("hustota")
    plt.savefig("src/visualizations/edge_resistance_distribution.png")
    plt.close(fig)

if __name__ == "__main__":
    main()