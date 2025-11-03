import math
from multiprocessing import Pool
import random
import sqlite3
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from src.util import graph_from_geojson, largest_connected_component, graph_to_folium
from src.kirchhoff import normalized_kirchhoff_index


db_name: str = "src/generated_data/random_networks/networks.sqlite"


def connected_double_edge_swap_distance(G: nx.Graph, max_swaps: int) -> int:
    # function to change the passed graph G by selecting nodes that form 
    # an almost square with two parallel edges and rotating them, 
    # while keeping the graph connected
    # u--v            u  v
    #        becomes  |  |
    # x--y            x  y

    def edge_length(G: nx.Graph, u: str, v: str) -> float:
        # calculate the length of the u-v edge in meters
        return geodesic(
            (G.nodes[u]['geometry'].coords[0][1], G.nodes[u]['geometry'].coords[0][0]), 
            (G.nodes[v]['geometry'].coords[0][1], G.nodes[v]['geometry'].coords[0][0])
            ).meters


    def is_almost_square(G: nx.Graph, a: str, b: str, c: str, d: str, rel_tol: float) -> bool:
        # compute whether abcd is a square with relative tolerance `rel_tol`
        ab: float = G[a][b]["distance"]
        cd: float = G[c][d]["distance"]
        ad: float = edge_length(G, a, d)
        bc: float = edge_length(G, b, c)
        
        return (
            math.isclose(ab, cd, rel_tol=rel_tol) and
            math.isclose(ad, bc, rel_tol=rel_tol) and
            math.isclose(ab, ad, rel_tol=rel_tol)
        )


    def sample_almost_square(G: nx.Graph, max_tries: int=1000, rel_tol: float=0.25) -> tuple[str, str, str, str] | None:
        # try to find four nodes that form an almost square (try it `max_tries times`)
        edges: list[tuple[str, str]] = list(G.edges())

        # try to find a square
        for _ in range(max_tries):
            # sample two edges
            (a, b), (c, d) = random.sample(edges, 2)

            # if there are less than four distinct nodes 
            # or the potential square does not have exactly two parallel edges, ignore it
            if len({a, b, c, d}) < 4 or (a, d) in edges or (d, a) in edges or (b, c) in edges or (c, b) in edges:
                continue

            # if abcd form a square, return those nodes
            if is_almost_square(G, a, b, c, d, rel_tol):
                return (a, b, c, d)
        
        # no square found
        return None
    
    # count the number of swaps done
    swaps: int = 0

    # try to do `max_swaps` swaps, each tries to find a square 1000 times
    for _ in range(max_swaps):
        square: tuple[str, str, str, str] | None = sample_almost_square(G)
        
        if not square:  # if no square found, try again
            continue
        
        # square was found, try to swap the edges
        a, b, c, d = square
        G_tmp = G.copy()
        G_tmp.remove_edge(a, b)
        G_tmp.remove_edge(c, d)
        G_tmp.add_edge(a, d)
        G_tmp.add_edge(b, c)

        # if graph stays connected even after swap, perform it
        if nx.is_connected(G_tmp):
            G.remove_edge(a, b)
            G.remove_edge(c, d)
            G.add_edge(a, d)
            G.add_edge(b, c)

            # calculate the distances of the new edges
            for u, v in zip([a, b], [d, c]):
                G[u][v]['distance'] = edge_length(G, u, v)

            # successful swap
            swaps += 1
    
    return swaps


def generate_random_graphs(country: str, n: int=10, max_swap: int=100) -> None:
    # generate `n` random graphs by swapping railway network of `country`
    # store the statistics of the generated graph into the database

    # load the graph
    G: nx.Graph = largest_connected_component(graph_from_geojson(f"graphs/{country}-railroad-network.json"))

    # open the connection to the database
    with sqlite3.connect(db_name) as conn:
        # create `n` new graphs
        for i in range(n):
            print(f"{i+1}/{n}") # print progress
            
            # create a new graph by swapping edges of G
            H: nx.Graph = G.copy()
            swaps: int = connected_double_edge_swap_distance(H, max_swaps=max_swap)
            
            # if no swaps were performed, try again
            while swaps == 0:
                swaps = connected_double_edge_swap_distance(H, max_swaps=max_swap)
            
            # compute the graph's statistics and write them into the database
            connected: bool = nx.is_connected(H)
            k_H: float = normalized_kirchhoff_index(H)
            sum_distances: float = sum([d["distance"] for _, _, d in H.edges(data=True)])

            conn.execute(
                """
                INSERT INTO graphs(country, swaps, connected, kirchhoff, sum_distances)
                VALUES (?, ?, ?, ?, ?)
                """,
                (country, swaps, connected, k_H, sum_distances)
            )
            conn.commit()


def main() -> None:
    """processes: int = 0      # 10
    with Pool(processes=processes) as pool:
        pool.map(generate_random_graphs, ["svk"]*processes)
    with Pool(processes=processes) as pool:
        pool.map(generate_random_graphs, ["cze"]*processes)"""
    
    
if __name__ == "__main__":
    main()