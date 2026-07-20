import sys
import os

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')
import networkx as nx
import pandas as pd

# Add workspace root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.graph.graph_query import find_common_movies_of_entities
from chatbot.graph.build_movie_graph import load_or_build_graph
from chatbot.data_loader import load_data

def test_aggregation_returns_only_movie_nodes():
    df = load_data()
    G = load_or_build_graph(df)
    
    # Steven Spielberg (Director) and Tom Hanks (Actor)
    movies = find_common_movies_of_entities(G, ["Steven Spielberg"], ["Tom Hanks"])
    
    # Assert every item returned is indeed a Movie node
    for m in movies:
        node_id = f"Movie:{m}"
        assert G.has_node(node_id), f"Node {node_id} does not exist in graph"
        assert G.nodes[node_id].get("type") == "Movie", f"Node {node_id} is not of type Movie"

def test_aggregation_tom_hanks_spielberg():
    df = load_data()
    G = load_or_build_graph(df)
    
    movies = find_common_movies_of_entities(G, ["Steven Spielberg"], ["Tom Hanks"])
    print(f"Tom Hanks + Spielberg co-collaborations found: {movies}")
    
    # Expected movies
    expected = {"Saving Private Ryan", "Catch Me If You Can", "The Terminal", "Bridge of Spies", "The Post"}
    found_expected = expected.intersection(set(movies))
    
    assert len(found_expected) > 0, f"Expected some of {expected} to be found, but got {movies}"
    
    # Ensure no node types other than Movie are returned
    for m in movies:
        assert not m.startswith("Actor:"), f"Returned actor node: {m}"
        assert not m.startswith("Director:"), f"Returned director node: {m}"
        assert not m.startswith("Genre:"), f"Returned genre node: {m}"
        assert not m.startswith("Country:"), f"Returned country node: {m}"

def test_no_common_movies_returns_empty_not_error():
    df = load_data()
    G = load_or_build_graph(df)
    
    # Query two entities that have never collaborated (e.g. Quentin Tarantino and some random actor)
    movies = find_common_movies_of_entities(G, ["Quentin Tarantino"], ["Ryan Reynolds"])
    assert movies == [], f"Expected no collaborations, but got {movies}"

if __name__ == "__main__":
    print("Running test_aggregation_returns_only_movie_nodes...")
    test_aggregation_returns_only_movie_nodes()
    print("test_aggregation_returns_only_movie_nodes PASSED.")
    
    print("Running test_aggregation_tom_hanks_spielberg...")
    test_aggregation_tom_hanks_spielberg()
    print("test_aggregation_tom_hanks_spielberg PASSED.")
    
    print("Running test_no_common_movies_returns_empty_not_error...")
    test_no_common_movies_returns_empty_not_error()
    print("test_no_common_movies_returns_empty_not_error PASSED.")
    print("All tests completed successfully!")
