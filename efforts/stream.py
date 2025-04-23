import networkx as nx
import time
import tracemalloc
import random
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import os
import numpy as np # Needed for spectral sparsifier and AMS sketch
from pathlib import Path
import csv # For saving results

# --- Directory Setup ---
OUTPUT_DIR = Path("./final_output")
PLOT_DIR = Path("./plots")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# --- Graph Generation ---
def create_graph_edges(n, m, seed=42, weighted=True):
    """Creates a graph with exactly n nodes and m edges."""
    print(f"Creating graph G (n={n}, m={m})...")
    if m > n * (n - 1) // 2:
        raise ValueError(f"Cannot create graph with {n} nodes and {m} edges. Max edges is {n*(n-1)//2}")
    if m < n - 1:
        print(f"Warning: Graph with n={n}, m={m} might not be connected.")

    G = nx.Graph()
    G.add_nodes_from(range(n))
    random.seed(seed)
    np.random.seed(seed) # Also seed numpy for weight generation

    edges_added = 0
    while edges_added < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and not G.has_edge(u, v):
            weight = np.random.uniform(1.0, 10.0) if weighted else 1.0
            G.add_edge(u, v, weight=weight)
            edges_added += 1

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# --- UnionFind (Unchanged) ---
class UnionFind:
    """Simple Union-Find implementation."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.num_nodes = n # Store n for clarity

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

    def num_sets(self):
        """Returns the number of disjoint sets."""
        return sum(1 for i, p in enumerate(self.parent) if i == p)

# --- Memory Measurement (Unchanged) ---
def measure_memory(func, *args, **kwargs):
    """Measures peak memory usage of a function call using tracemalloc."""
    tracemalloc.start()
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = end_time - start_time
    # Peak memory is more indicative of the algorithm's space requirement
    # Convert peak to MiB for readability
    peak_memory_mib = peak / (1024 * 1024)
    return result, elapsed_time, peak_memory_mib

# --- Algorithm Implementations ---

# 1. Connectivity (Streaming) - OK
def run_connectivity(stream, n):
    """
    Constructs a spanning forest using Union-Find. Streaming.
    Space: O(n)
    Time: O(m * alpha(n))
    """
    uf = UnionFind(n)
    spanning_forest_edges = []
    num_edges_processed = 0
    for u, v, _ in stream: # Ignore weight for connectivity
        num_edges_processed += 1
        # Ensure nodes are within expected range [0, n-1]
        if not (0 <= u < n and 0 <= v < n):
             print(f"Warning: Edge ({u},{v}) has node outside range [0, {n-1}]. Skipping.")
             continue
        if uf.union(u, v):
            spanning_forest_edges.append(tuple(sorted((u, v))))

    # Create the resulting graph object
    H = nx.Graph()
    H.add_nodes_from(range(n))
    H.add_edges_from(spanning_forest_edges)
    # Calculate number of connected components
    num_components = uf.num_sets()
    return H, {'edge_count': len(spanning_forest_edges), 'components': num_components}

# 2. Spanner Construction (Naive Streaming Baseline) - Inefficient
def run_spanner(stream, n, alpha):
    """
    Constructs an alpha-spanner (naive baseline). Streaming technically, but slow.
    Space: O(|E_H|) where |E_H| can be large (up to O(n^(1+eps))).
    Time: O(m * (|V_H|+|E_H|)) per edge using BFS/Dijkstra naively. Very slow.
    """
    H = nx.Graph()
    H.add_nodes_from(range(n))
    edge_count = 0

    for u, v, data in stream:
         # Ensure nodes are within expected range [0, n-1]
        if not (0 <= u < n and 0 <= v < n):
             print(f"Warning: Edge ({u},{v}) has node outside range [0, {n-1}]. Skipping.")
             continue

        # Add nodes if they aren't already (should be added by add_nodes_from, but safe check)
        # No need to add nodes explicitly if add_nodes_from(range(n)) was called

        try:
            # Calculate shortest path distance *using unweighted edges* in the current spanner H
            # Note: If graph is weighted, use nx.shortest_path_length(H, source=u, target=v, weight='weight')
            # Using unweighted for simplicity based on common definition, adjust if needed.
            distance = nx.shortest_path_length(H, source=u, target=v) # Unweighted distance
            if distance > alpha:
                H.add_edge(u, v, **data) # Add with original data (like weight)
                edge_count += 1
        except nx.NetworkXNoPath:
            # If no path exists, the distance is effectively infinite > alpha
            H.add_edge(u, v, **data)
            edge_count += 1
        # NodeNotFound should not happen if nodes 0..n-1 exist

    return H, {'edge_count': edge_count}

# 3. MST (Offline Kruskal Baseline) - Not Streaming
def run_mst_offline_kruskal(stream, n):
    """
    Kruskal MST using Union-Find. Requires sorting all edges first. OFFLINE Baseline.
    Space: O(m) to store and sort edges + O(n) for UF.
    Time: O(m log m) or O(m log n) for sorting + O(m * alpha(n)) for UF.
    """
    H = nx.Graph()
    H.add_nodes_from(range(n))

    # Store and Sort the edges by weight - VIOLATES STREAMING
    # Convert generator to list if needed
    stream_list = list(stream)
    # Handle missing weights gracefully
    sorted_stream = sorted(stream_list, key=lambda x: x[2].get('weight', 1))

    uf = UnionFind(n)
    mst_weight = 0
    edge_count = 0

    for u, v, data in sorted_stream:
        # Ensure nodes are within expected range [0, n-1]
        if not (0 <= u < n and 0 <= v < n):
             # Should not happen if graph generation is correct
             print(f"Warning: Edge ({u},{v}) has node outside range [0, {n-1}]. Skipping.")
             continue

        weight = data.get('weight', 1)
        if uf.union(u, v):  # Only add if it doesn't form a cycle
            H.add_edge(u, v, weight=weight)
            mst_weight += weight
            edge_count += 1
            # Optimization: stop if n-1 edges are added for a connected graph
            # if edge_count == n - uf.num_sets(): # Stops when forest is complete
            #     break

    num_components = uf.num_sets()
    return H, {'edge_count': edge_count, 'total_weight': mst_weight, 'components': num_components}

# 4. MST (Cycle Property Streaming) - Potentially Slow
def run_mst_cycle(stream, n):
    """
    Constructs an MST/MSF using the cycle property. Streaming, but potentially slow cycle find.
    Space: O(n) edges stored in H potentially.
    Time: Potentially O(m * n) or worse due to find_cycle.
    """
    H = nx.Graph()
    H.add_nodes_from(range(n))
    edge_count = 0
    total_weight = 0

    for u, v, data in stream:
        # Ensure nodes are within expected range [0, n-1]
        if not (0 <= u < n and 0 <= v < n):
             print(f"Warning: Edge ({u},{v}) has node outside range [0, {n-1}]. Skipping.")
             continue

        weight = data.get('weight', 1)

        # If edge already exists with potentially different weight, skip or update?
        # Current networkx behavior might update. Assume simple graphs (no parallel edges).
        if H.has_edge(u,v): # Avoid adding parallel edges explicitly
             # Decide on policy: ignore, update weight? Ignoring for now.
             continue

        H.add_edge(u, v, weight=weight)

        try:
            # Check if adding the edge created a cycle
            # orientation='ignore' needed for undirected cycle finding > nx 2. Cycles are edge lists.
            cycle_edges = nx.find_cycle(H, source=u, orientation='ignore')

            # If a cycle is found
            max_weight_edge = None
            max_weight_in_cycle = -float('inf')
            edge_to_remove = None

            # Iterate through edges in the cycle path
            for edge_in_cycle in cycle_edges:
                u_c, v_c = edge_in_cycle[0], edge_in_cycle[1]
                # Ensure edge exists (should always in undirected)
                if H.has_edge(u_c, v_c):
                    w_c = H[u_c][v_c].get('weight', 1)
                    if w_c > max_weight_in_cycle:
                        max_weight_in_cycle = w_c
                        # Store edge canonically for removal
                        edge_to_remove = tuple(sorted((u_c, v_c)))
                # else: # Should not happen with ignore orientation
                #      print(f"Warning: Edge {edge_in_cycle} from cycle not found in H.")

            if edge_to_remove:
                 u_rem, v_rem = edge_to_remove
                 if H.has_edge(u_rem, v_rem):
                    H.remove_edge(u_rem, v_rem)
                 # else: # Should not happen
                 #     print(f"Warning: Trying to remove non-existent edge {edge_to_remove}")

        except nx.NetworkXNoCycle:
            # No cycle formed by adding the edge, keep it.
            pass
        # NodeNotFound should not happen

    # Calculate final stats
    edge_count = H.number_of_edges()
    total_weight = sum(d.get('weight', 1) for _, _, d in H.edges(data=True))
    num_components = nx.number_connected_components(H)

    return H, {'edge_count': edge_count, 'total_weight': total_weight, 'components': num_components}


# 5. Greedy Matching (Streaming) - OK
def run_greedy_matching(stream, n):
    """
    Constructs a 2-approximate maximum cardinality matching greedily. Streaming.
    Space: O(n)
    Time: O(m)
    """
    matched_nodes = set()
    M = [] # List of edge tuples (optionally store weights if needed later)

    for u, v, data in stream:
        # Ensure nodes are within expected range [0, n-1]
        if not (0 <= u < n and 0 <= v < n):
             print(f"Warning: Edge ({u},{v}) has node outside range [0, {n-1}]. Skipping.")
             continue
        weight = data.get('weight', 1) # Store weight even if not used by logic

        if u not in matched_nodes and v not in matched_nodes:
            M.append( (u, v, weight) ) # Store edge with weight
            matched_nodes.add(u)
            matched_nodes.add(v)

    # Create graph object
    H = nx.Graph()
    H.add_nodes_from(range(n))
    total_weight = 0
    for u, v, weight in M:
        H.add_edge(u, v, weight=weight)
        total_weight += weight

    return H, {'edge_count': len(M), 'total_weight': total_weight}

# 6. Greedy Weighted Matching (Streaming) - OK
def run_greedy_weighted_matching(stream, n, gamma=0.001): # Use small gamma > 0
    """
    Constructs an approx maximum weight matching greedily with replacement. Streaming.
    Approximation factor depends on gamma.
    Space: O(n)
    Time: O(m)
    """
    if gamma <= 0:
        # print("Warning: Gamma must be > 0 for Greedy Weighted Matching. Using 1e-9.")
        gamma = 1e-9 # Use a small positive value

    node_to_edge_in_M = {} # node -> edge tuple (u,v)
    edge_weights_M = {}    # edge tuple (u,v) -> weight
    M_edges = set()         # set of edge tuples (u,v)

    for u, v, data in stream:
        # Ensure nodes are within expected range [0, n-1]
        if not (0 <= u < n and 0 <= v < n):
             print(f"Warning: Edge ({u},{v}) has node outside range [0, {n-1}]. Skipping.")
             continue

        edge = tuple(sorted((u, v)))
        weight = data.get('weight', 1) # Default weight 1 if missing

        conflicting_edges_tuples = set()
        if u in node_to_edge_in_M:
            conflicting_edges_tuples.add(node_to_edge_in_M[u])
        if v in node_to_edge_in_M:
            conflicting_edges_tuples.add(node_to_edge_in_M[v])

        # Ensure we only consider edges currently in M (might have been removed by prior conflicts)
        w_C = sum(edge_weights_M[c_edge] for c_edge in conflicting_edges_tuples if c_edge in edge_weights_M)

        # Replacement condition w_e >= (1 + gamma) * w_C
        if weight >= (1 + gamma) * w_C:
            # Remove conflicting edges *that are still in M*
            edges_to_remove_now = {c_edge for c_edge in conflicting_edges_tuples if c_edge in M_edges}

            for c_edge in edges_to_remove_now:
                # Remove from M structures
                M_edges.remove(c_edge)
                del edge_weights_M[c_edge]
                # Clean up node_to_edge map for the endpoints of the removed edge
                c_u, c_v = c_edge
                if c_u in node_to_edge_in_M and node_to_edge_in_M[c_u] == c_edge:
                    del node_to_edge_in_M[c_u]
                if c_v in node_to_edge_in_M and node_to_edge_in_M[c_v] == c_edge:
                    del node_to_edge_in_M[c_v]

            # Add the new edge (only if it doesn't conflict with itself, which is impossible here)
            M_edges.add(edge)
            edge_weights_M[edge] = weight
            node_to_edge_in_M[u] = edge
            node_to_edge_in_M[v] = edge

    # Create graph object
    H = nx.Graph()
    H.add_nodes_from(range(n))
    total_weight = 0
    for edge, weight in edge_weights_M.items():
        H.add_edge(edge[0], edge[1], weight=weight)
        total_weight += weight

    return H, {'edge_count': len(M_edges), 'total_weight': total_weight}

# 7. Triangle Count (Exact Offline Baseline) - Not Streaming
def exact_triangle_count(G):
    """
    Compute the exact number of triangles in G. OFFLINE Baseline.
    """
    # Make sure graph is simple (no self-loops, no parallel edges) for accurate triangle count
    if not nx.is_simple(G):
        G_simple = nx.Graph(G) # Create a simple copy
    else:
        G_simple = G
    tri_per_node = nx.triangles(G_simple)
    # Each triangle is counted 3 times (once for each node)
    return sum(tri_per_node.values()) // 3

# 8. Triangle Count Estimation (Moments Offline Baseline) - Not Streaming
def estimate_triangles_moment_offline(G):
    """
    Estimate number of triangles using F0, F1, F2 computed offline. OFFLINE Baseline.
    Time Complexity: O(n^3)
    Space Complexity: O(1) if G is given, O(n+m) otherwise.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 3: return 0.0

    F1 = m * (n - 2)
    if F1 < 0: F1 = 0 # Handle n=0,1,2 case

    F0 = 0
    F2 = 0

    nodes = list(G.nodes())
    adj = G.adj # Adjacency view for faster lookups

    # Iterate over all triples of nodes - O(N^3) - Expensive!
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                u, v, w = nodes[i], nodes[j], nodes[k]
                # count edges among the triple
                cnt = 0
                if v in adj[u]: cnt += 1
                if w in adj[u]: cnt += 1
                if w in adj[v]: cnt += 1

                if cnt > 0:
                    F0 += 1
                F2 += cnt * cnt # cnt^2

    if F0 == 0 and F1 == 0 and F2 == 0: return 0.0 # Avoid potential division by zero issues if graph is empty

    # Formula T = F0 - 1.5*F1 + 0.5*F2
    # Potential issue: Can result in negative estimates for sparse graphs
    estimate = F0 - 1.5 * F1 + 0.5 * F2
    return max(0.0, estimate) # Return non-negative estimate


# --- AMS Sketch Helper Class ---
class AMS_Sketch:
    """Basic AMS sketch for F2 estimation."""
    def __init__(self, k, domain_size):
        """
        k: number of hash functions (determines accuracy/probability)
        domain_size: maximum possible value (e.g., n for nodes, or hash range)
        """
        self.k = k
        self.domain_size = domain_size
        self.counters = [0] * k
        # Generate k 4-wise independent hash functions (using random seeds for simplicity here)
        # A better implementation would use explicit hash function families.
        self.seeds = [(random.randint(1, sys.maxsize), random.randint(1, sys.maxsize)) for _ in range(k)]

    def _hash(self, item, i):
        """Simple hash function mapping item to {-1, +1}. Returns hash index and sign."""
        # Use Python's built-in hash with seeds for pseudo-randomness
        # Map to -1, +1 based on parity
        hash_val = hash((item, self.seeds[i][0]))
        sign = 1 if hash((hash_val, self.seeds[i][1])) % 2 == 0 else -1
        return sign

    def update(self, item):
        """Process an item from the stream."""
        for i in range(self.k):
            sign = self._hash(item, i)
            self.counters[i] += sign

    def estimate_f2(self):
        """Estimate F2 = sum(frequency(item)^2)."""
        # Average of counter squares
        return sum(c*c for c in self.counters) / self.k

    def estimate_triangles(self, f0_est, f1_est):
        """Estimate triangles from estimated moments."""
        f2_est = self.estimate_f2()
        estimate = f0_est - 1.5 * f1_est + 0.5 * f2_est
        return max(0.0, estimate)


# 9. Triangle Count Estimation (Streaming AMS) - Streaming
def run_streaming_triangle_estimation_ams(stream, n, k=100): # k: number of sketch counters
    """
    Estimates triangles using an AMS sketch for F2. Streaming.
    Space: O(k * log(n)) - k counters, size depends on stream length but bounded.
    Time: O(m * k) - k hash computations per edge.
    Needs estimates for F0 and F1 separately (F1 is easy, F0 needs another sketch like HyperLogLog).
    For simplicity here, we will *assume* F0 and F1 are known or estimated elsewhere.
    A more complete solution would integrate F0 estimation.
    We'll compute F1 exactly from the stream and F0 naively (count distinct triples seen).
    """
    if n < 3: return 0.0

    # --- F1 Calculation (Exact from stream) ---
    m = 0 # Count edges as they arrive
    for _ in stream: # Need to consume the stream once to get m
        m += 1
    f1_exact = m * (n - 2)
    if f1_exact < 0 : f1_exact = 0

    # --- F0 and F2 Estimation (Requires another pass or buffering/sampling) ---
    # Problem: The F0/F2 moments are over the *triples*, not edges.
    # The basic AMS sketch processes individual items. To use it for triples,
    # we'd need to generate/sample triples from the edge stream, which is complex.

    # --- Alternative: Estimate F2 of edge stream (less direct for triangles) ---
    # Let's implement the edge F2 sketch as an example of AMS, though its direct use
    # for the triangle formula is less standard than sketching the triple vector.
    ams_sketch_edges = AMS_Sketch(k=k, domain_size=n*n) # Domain is potential edges
    # Reset stream if it was a generator
    # This highlights a challenge: streaming algos often need the stream passed again or stored.
    # Assuming 'stream' can be iterated multiple times for this example:
    distinct_triples_seen = set() # Naive F0 estimate for triples requires storing triples
    f2_estimator_triples = defaultdict(int) # Naive F2 estimate for triples

    # This part becomes NON-STREAMING if we iterate through triples generated from edges
    # Let's stick to a simple edge-based F2 sketch example instead, acknowledging it doesn't directly fit the triangle formula.

    print("Streaming Triangle Estimation (AMS): Note: This implementation is simplified.")
    print("It calculates F1 exactly and demonstrates AMS sketch for edge F2,")
    print("but properly estimating F0/F2 for *triples* in a stream is more complex.")

    # --- Simplified Approach: Track edge occurrences (Doesn't directly give triangle count) ---
    # This is just to demonstrate the sketch mechanics.
    ams_sketch_edge_stream = AMS_Sketch(k=k, domain_size=n*n) # Domain ~ m or n^2
    edge_stream_list = list(stream) # Convert to list to iterate multiple times if needed
    for u, v, _ in edge_stream_list:
         edge_tuple = tuple(sorted((u,v)))
         # Update sketch based on edge arrivals
         ams_sketch_edge_stream.update(edge_tuple)

    f2_edges_est = ams_sketch_edge_stream.estimate_f2()

    # Cannot compute triangle estimate without F0/F2 over triples. Return NaN or 0.
    return np.nan, {'F1_exact': f1_exact, 'F2_edges_est': f2_edges_est, 'Note': 'Triangle estimate requires F0/F2 over triples, not edges.'}


# 10. Spectral Sparsifier (Offline Baseline) - Not Streaming
def spectral_sparsify_offline(G, gamma):
    """
    Compute a (1+gamma)-spectral sparsifier via effective resistance sampling. OFFLINE Baseline.
    Requires numpy. Computationally expensive (Laplacian pseudoinverse).
    Space: O(n^2) for matrices.
    Time: O(n^omega) or O(m*n) depending on pseudoinverse method, plus O(m*n) for resistances.
    """
    n = G.number_of_nodes()
    if n == 0: return nx.Graph(), {'edge_count': 0}
    if G.number_of_edges() == 0: return G.copy(), {'edge_count': 0} # Handle empty graph

    is_weighted = nx.is_weighted(G)
    weight_key = 'weight' if is_weighted else None

    # Ensure nodes are 0 to n-1 for matrix indexing if not already
    node_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    G_idx = nx.relabel_nodes(G, node_map, copy=True)

    try:
        # Build Laplacian matrix L
        L = nx.linalg.laplacian_matrix(G_idx, nodelist=range(n), weight=weight_key).toarray()

        # Check for disconnected graph - pseudoinverse works but interpretation differs
        if not nx.is_connected(G_idx):
             print("Warning: Graph is disconnected. Computing pseudoinverse for spectral sparsifier.")
             # Add small regularization to avoid issues with zero eigenvalues for disconnected components
             # L += np.eye(n) * 1e-9

        # Pseudoinverse of L - Computationally expensive!
        L_pinv = np.linalg.pinv(L, rcond=1e-10) # Adjust rcond if needed

    except np.linalg.LinAlgError as e:
        print(f"Error computing pseudoinverse: {e}. Check graph connectivity/Laplacian.")
        # Return an empty graph or handle error appropriately
        H_sparse = nx.Graph()
        H_sparse.add_nodes_from(G.nodes())
        return H_sparse, {'edge_count': 0, 'error': 'LinAlgError'}
    except MemoryError:
        print(f"MemoryError computing pseudoinverse for n={n}. Skipping.")
        H_sparse = nx.Graph()
        H_sparse.add_nodes_from(G.nodes())
        return H_sparse, {'edge_count': 0, 'error': 'MemoryError'}


    # Sampling parameter q >= O(n log n / gamma^2) expected edges
    # BSS sampling probability: p_e = min(1, q * R_e * w_e)
    # Spielman-Srivastava uses q = O(log n / gamma^2) samples *with replacement* and reweighting.
    # Let's use the simpler effective resistance sampling probability from BSS/generic Spielman-Teng work.
    # Need total effective resistance sum or sample count q.
    q = 8 * np.log(n) / (gamma ** 2) if n > 1 else 1 # Number of samples needed approx

    H_sparse_idx = nx.Graph()
    H_sparse_idx.add_nodes_from(range(n))
    edges_added = 0
    total_weight_sparse = 0

    # Compute effective resistance and sample edges
    for u_idx, v_idx, data in G_idx.edges(data=True):
        w = data.get('weight', 1.0)
        # indicator vector difference e_uv
        e_uv = np.zeros(n)
        e_uv[u_idx] = 1
        e_uv[v_idx] = -1

        # Effective resistance R_uv = e_uv^T L_pinv e_uv
        R_uv = e_uv @ L_pinv @ e_uv
        R_uv = max(R_uv, 1e-12) # Ensure positive resistance for probability calc

        # Sampling probability
        p_uv = min(1.0, q * w * R_uv)

        if random.random() < p_uv:
            # Reweight edge for sparsifier
            w_new = w / p_uv
            H_sparse_idx.add_edge(u_idx, v_idx, weight=w_new)
            edges_added += 1
            total_weight_sparse += w_new

    # Relabel nodes back to original labels
    inv_node_map = {i: node for node, i in node_map.items()}
    H_sparse = nx.relabel_nodes(H_sparse_idx, inv_node_map, copy=True)

    return H_sparse, {'edge_count': edges_added, 'total_weight': total_weight_sparse}


# --- Experiment Runner ---
def run_experiments(algorithms, n_values, m_values_func, fixed_param, vary_param_name):
    """Runs algorithms and collects time/memory data."""
    results = defaultdict(lambda: defaultdict(list)) # results[algo_name][metric] = [values]

    param_values = n_values if vary_param_name == 'nodes' else m_values_func(fixed_param)

    for i, val in enumerate(param_values):
        if vary_param_name == 'nodes':
            n = val
            m = fixed_param # Fixed number of edges
            # Ensure m is feasible for n
            if m > n * (n - 1) // 2:
                print(f"Skipping n={n}, m={m}: Too many edges for nodes. Max is {n*(n-1)//2}")
                # Pad results with NaN or previous value? Padding with NaN.
                for algo_name in algorithms:
                    results[algo_name]['time'].append(np.nan)
                    results[algo_name]['memory'].append(np.nan)
                    results[algo_name]['param_value'].append(val)
                continue
            if m < n - 1 and i > 0: # Avoid disconnected graphs for larger N if M is small
                 print(f"Warning: n={n}, m={m} likely disconnected. Results might vary.")

        else: # vary_param_name == 'edges'
            n = fixed_param # Fixed number of nodes
            m = val
            if m > n * (n - 1) // 2:
                print(f"Skipping n={n}, m={m}: Too many edges requested. Max is {n*(n-1)//2}")
                # Pad results
                for algo_name in algorithms:
                    results[algo_name]['time'].append(np.nan)
                    results[algo_name]['memory'].append(np.nan)
                    results[algo_name]['param_value'].append(val)
                continue

        print(f"\n--- Running for {vary_param_name}={val} ({'n' if vary_param_name=='edges' else 'm'}={fixed_param}) ---")
        G = create_graph_edges(n, m, seed=42, weighted=True)
        # Create a list view of edges once, pass to functions
        # This simulates the stream arriving. data=True includes weights.
        edge_stream = list(G.edges(data=True))
        print(f"Generated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")


        for algo_name, func_info in algorithms.items():
            func = func_info['func']
            params = func_info.get('params', {})
            is_streaming = func_info.get('streaming', True) # Assume streaming unless specified

            print(f"Running {algo_name}...")
            try:
                # For offline algos needing the graph object:
                if algo_name in ['exact_triangle_count', 'estimate_triangles_moment_offline', 'spectral_sparsify_offline']:
                     (result_graph, stats), elapsed_time, peak_memory = measure_memory(func, G, **params)
                # For Kruskal, pass the stream but it sorts internally
                elif algo_name == 'run_mst_offline_kruskal':
                     (result_graph, stats), elapsed_time, peak_memory = measure_memory(func, edge_stream, n, **params)
                # For streaming triangle count, pass stream and n
                elif algo_name == 'run_streaming_triangle_estimation_ams':
                     (result_val, stats), elapsed_time, peak_memory = measure_memory(func, edge_stream, n, **params)
                # For standard streaming algos pass stream and n
                else:
                    (result_graph, stats), elapsed_time, peak_memory = measure_memory(func, edge_stream, n, **params)

                print(f"  Time: {elapsed_time:.4f}s, Memory: {peak_memory:.4f} MiB")
                if isinstance(stats, dict) and 'edge_count' in stats:
                    print(f"  Result Edges: {stats['edge_count']}")
                elif algo_name == 'exact_triangle_count':
                     print(f"  Exact Triangles: {result_graph}") # result_graph holds the count here
                elif algo_name == 'estimate_triangles_moment_offline':
                     print(f"  Estimated Triangles (Moment Offline): {result_graph:.2f}")
                elif algo_name == 'run_streaming_triangle_estimation_ams':
                     print(f"  Estimated F1: {stats['F1_exact']:.2f}, Estimated Edge F2: {stats['F2_edges_est']:.2f}")


                results[algo_name]['time'].append(elapsed_time)
                results[algo_name]['memory'].append(peak_memory)
                results[algo_name]['param_value'].append(val)
                # Store other stats if needed (e.g., edge counts, weights)
                for stat_key, stat_val in stats.items():
                    results[algo_name][stat_key].append(stat_val)


            except Exception as e:
                print(f"  ERROR running {algo_name}: {e}")
                # Add NaN for this point to keep array lengths consistent
                results[algo_name]['time'].append(np.nan)
                results[algo_name]['memory'].append(np.nan)
                results[algo_name]['param_value'].append(val)
                # Add NaN for other stats as well
                # Find expected keys from a previous successful run if possible, otherwise skip
                if i > 0 and len(results[algo_name].keys()) > 3: # Check if stats keys exist
                    for key in results[algo_name].keys():
                        if key not in ['time', 'memory', 'param_value']:
                            results[algo_name][key].append(np.nan)


    return results

# --- Plotting Function ---
def plot_results(results, fixed_param, vary_param_name, plot_dir):
    """Plots time and memory usage."""
    param_label = f"Number of {vary_param_name.capitalize()}"
    fixed_label = 'Nodes' if vary_param_name == 'edges' else 'Edges'
    title_suffix = f"(Fixed {fixed_label} = {fixed_param})"

    metrics = ['time', 'memory']
    y_labels = {'time': 'Execution Time (s)', 'memory': 'Peak Memory Usage (MiB)'}

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for algo_name, data in results.items():
            if data[metric]: # Check if there is data to plot
                # Use the recorded param_value for x-axis
                param_values = data.get('param_value', [])
                metric_values = data[metric]
                # Ensure lengths match, filter NaNs for plotting
                valid_indices = ~np.isnan(metric_values) & ~np.isnan(param_values) # Ensure both are valid
                if np.any(valid_indices): # Only plot if there are valid points
                    plt.plot(np.array(param_values)[valid_indices], np.array(metric_values)[valid_indices], marker='o', linestyle='-', label=algo_name)

        plt.xlabel(param_label)
        plt.ylabel(y_labels[metric])
        plt.title(f"{y_labels[metric]} vs {param_label} {title_suffix}")
        plt.legend()
        plt.grid(True)
        plot_filename = plot_dir / f"{metric}_vs_{vary_param_name}_{fixed_label}{fixed_param}.png"
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

# --- Main Execution ---
if __name__ == "__main__":

    # --- Define Algorithms to Run ---
    # Include parameters and indicate if streaming or offline baseline
    # Add gamma for spanner/weighted matching, epsilon for sparsifier
    algorithms_to_run = {
        # Streaming
        "Connectivity":        {'func': run_connectivity, 'streaming': True},
        "Spanner (alpha=3)":   {'func': run_spanner, 'params': {'alpha': 3}, 'streaming': True, 'slow': True}, # Naive, slow
        "MST (Cycle)":         {'func': run_mst_cycle, 'streaming': True, 'slow': True}, # Potentially slow
        "Greedy Matching":     {'func': run_greedy_matching, 'streaming': True},
        "Greedy W-Matching":   {'func': run_greedy_weighted_matching, 'params': {'gamma': 0.1}, 'streaming': True},
        # Streaming Triangle Estimation Placeholder
        "Triangle Est (AMS)":  {'func': run_streaming_triangle_estimation_ams, 'params': {'k': 50}, 'streaming': True, 'note': 'Simplified'},

        # Offline Baselines
        "MST (Kruskal Offline)": {'func': run_mst_offline_kruskal, 'streaming': False},
        "Exact Triangles":       {'func': exact_triangle_count, 'streaming': False},
        "Triangle Est (Moment Offline)": {'func': estimate_triangles_moment_offline, 'streaming': False, 'slow': True}, # O(n^3)
        # "Spectral Sparsify (Offline)": {'func': spectral_sparsify_offline, 'params': {'gamma': 0.5}, 'streaming': False, 'slow': True, 'needs_numpy': True}, # Very slow/memory intensive
    }

    # --- 2. Run on Specific Graph (200 Nodes, 1000 Edges) ---
    print("\n--- Running on Specific Graph (N=200, M=1000) ---")
    N_SPECIFIC = 200
    M_SPECIFIC = 1000
    specific_graph = create_graph_edges(N_SPECIFIC, M_SPECIFIC, seed=42, weighted=True)
    specific_stream = list(specific_graph.edges(data=True))

    results_specific = {}
    for name, info in algorithms_to_run.items():
        print(f"\nRunning {name} on specific graph...")
        try:
             # Adjust call based on whether it needs G or stream
            if name in ['exact_triangle_count', 'estimate_triangles_moment_offline', 'spectral_sparsify_offline']:
                 res, t, mem = measure_memory(info['func'], specific_graph, **info.get('params', {}))
            elif name == 'run_mst_offline_kruskal':
                 res, t, mem = measure_memory(info['func'], specific_stream, N_SPECIFIC, **info.get('params', {}))
            elif name == 'run_streaming_triangle_estimation_ams':
                 res, t, mem = measure_memory(info['func'], specific_stream, N_SPECIFIC, **info.get('params', {}))
            else:
                 res, t, mem = measure_memory(info['func'], specific_stream, N_SPECIFIC, **info.get('params', {}))

            print(f"  Time: {t:.4f}s, Memory: {mem:.4f} MiB")
            results_specific[name] = {'result': res[0] if isinstance(res, tuple) else res, # Store graph or value
                                      'stats': res[1] if isinstance(res, tuple) and len(res) > 1 else {},
                                      'time': t, 'memory': mem}
            # Print key stats
            if isinstance(res, tuple) and isinstance(res[1], dict):
                 print(f"  Stats: {res[1]}")
            elif name == 'exact_triangle_count':
                 print(f"  Exact Triangles: {res}")


        except Exception as e:
            print(f"  ERROR running {name}: {e}")
            results_specific[name] = {'error': str(e)}

    # Save the original graph image
    print("\nSaving visualization of the specific graph (N=200, M=1000)...")
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(specific_graph, seed=42, k=0.2) # Adjust layout params
    nx.draw(specific_graph, pos, node_size=50, width=0.5, with_labels=False)
    plt.title("Input Graph (N=200, M=1000)")
    graph_filename = OUTPUT_DIR / "graph_n200_m1000.png"
    plt.savefig(graph_filename, dpi=300)
    print(f"Saved graph image: {graph_filename}")
    plt.close()

    # Optionally, save one result graph, e.g., MST
    mst_result = results_specific.get("MST (Kruskal Offline)", {}).get('result')
    if isinstance(mst_result, nx.Graph):
        plt.figure(figsize=(12, 12))
        pos_mst = nx.spring_layout(mst_result, seed=42, k=0.2)
        nx.draw(mst_result, pos_mst, node_size=50, width=0.5, with_labels=False, edge_color='red')
        plt.title("Result: MST (Kruskal Offline) for N=200, M=1000")
        mst_filename = OUTPUT_DIR / "result_mst_kruskal_n200_m1000.png"
        plt.savefig(mst_filename, dpi=300)
        print(f"Saved MST image: {mst_filename}")
        plt.close()


    # --- 3. Complexity Analysis: Vary Edges (Fixed Nodes) ---
    print("\n--- Complexity Analysis: Varying Edges ---")
    N_FIXED = 100 # Keep nodes constant
    # M_VALUES = [100, 200, 500, 1000, 2000, 4000] # Up to N*(N-1)/2 = 4950
    M_VALUES = list(np.linspace(100, min(4000, N_FIXED*(N_FIXED-1)//2), num=8, dtype=int)) # Generate edge values
    print(f"Node count fixed: {N_FIXED}")
    print(f"Edge counts to test: {M_VALUES}")

    results_vary_m = run_experiments(algorithms_to_run, [], lambda n: M_VALUES, N_FIXED, 'edges')
    plot_results(results_vary_m, N_FIXED, 'edges', PLOT_DIR)

    # Save raw data
    with open(OUTPUT_DIR / f'results_vary_edges_n{N_FIXED}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        header = ['algorithm', 'metric'] + [f'm={m}' for m in M_VALUES]
        writer.writerow(header)
        for algo, metrics_data in results_vary_m.items():
            for metric, values in metrics_data.items():
                 if metric != 'param_value': # Exclude the redundant param list
                    writer.writerow([algo, metric] + values)
    print(f"Saved raw data for varying edges to {OUTPUT_DIR}")


    # --- 4. Complexity Analysis: Vary Nodes (Fixed Edges) ---
    print("\n--- Complexity Analysis: Varying Nodes ---")
    M_FIXED = 500 # Keep edges constant
    # N_VALUES = [50, 100, 150, 200, 300, 400]
    # Ensure M_FIXED is feasible for min N (e.g., n=50 -> max edges = 50*49/2 = 1225)
    N_VALUES = list(np.linspace(40, 400, num=8, dtype=int)) # Start N high enough for M
    print(f"Edge count fixed: {M_FIXED}")
    print(f"Node counts to test: {N_VALUES}")

    results_vary_n = run_experiments(algorithms_to_run, N_VALUES, None, M_FIXED, 'nodes')
    plot_results(results_vary_n, M_FIXED, 'nodes', PLOT_DIR)

     # Save raw data
    with open(OUTPUT_DIR / f'results_vary_nodes_m{M_FIXED}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        header = ['algorithm', 'metric'] + [f'n={n}' for n in N_VALUES]
        writer.writerow(header)
        for algo, metrics_data in results_vary_n.items():
            for metric, values in metrics_data.items():
                 if metric != 'param_value':
                    writer.writerow([algo, metric] + values)
    print(f"Saved raw data for varying nodes to {OUTPUT_DIR}")


    # --- 5. Other Evaluation Methods ---
    print("\n--- Other Evaluation Considerations ---")
    print("1. Approximation Quality:")
    print("   - For Matching: Compare greedy results size/weight to offline max weight matching (nx.max_weight_matching).")
    print("   - For Spanner: Calculate stretch factor (max ratio of spanner_dist/original_dist) - computationally expensive.")
    print("   - For MST: Compare streaming MST weight (if implemented) or run_mst_cycle weight to Kruskal's MST weight.")
    # Example: Compare Greedy Matching vs Kruskal MST weight on the specific graph
    if "Greedy Matching" in results_specific and "MST (Kruskal Offline)" in results_specific:
         greedy_match_stats = results_specific["Greedy Matching"].get('stats', {})
         mst_stats = results_specific["MST (Kruskal Offline)"].get('stats', {})
         if 'total_weight' in greedy_match_stats:
              print(f"   - Greedy Matching Weight (N=200, M=1000): {greedy_match_stats['total_weight']:.2f}")
         if 'total_weight' in mst_stats:
              print(f"   - Kruskal MST Weight (N=200, M=1000): {mst_stats['total_weight']:.2f}")

    print("\n2. Order Dependence:")
    print("   - Shuffle the edge stream (e.g., `random.shuffle(specific_stream)`) and re-run algorithms.")
    print("   - Compare output quality (e.g., matching size/weight, spanner size) across different orders.")
    # You would add a loop here, shuffling specific_stream and re-running algos to test this.

    print("\n3. Comparison to Theoretical Bounds:")
    print("   - The plots generated visualize the empirical scaling.")
    print("   - Compare slopes on log-log plots (if generated) to theoretical exponents (e.g., O(n), O(m)).")

    print("\n--- Analysis Complete ---")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOT_DIR}")