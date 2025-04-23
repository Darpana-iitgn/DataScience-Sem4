# --- START OF (MODIFIED) FILE project1.0.py ---

# Original classes from project1.0.py go here...
# (L0Sampler, GraphConnectivitySketch, GraphLinearMeasurements, AdaptiveGraphSketch,
#  ModifiedBoruvkaSketching, SimpleSetSketch, CountMinSketch, gSketch,
#  GraphVisualSketchHierarchy)
# ...
# I will paste the original classes here for completeness, with minor fixes/imports.

import random
import math
import time
import sys
import hashlib # Added for CountMinSketch, gSketch
from collections import defaultdict
import heapq # Corrected from 'heap' import

# --- L0Sampler Class ---
class L0Sampler:
    """
    A very simple 'L0-sampler' over a dynamic multiset of items:
      - supports insert(item) and delete(item)
      - sample() returns a uniformly random item among those currently present (or None)
    Note: deletion implementation here swaps with the end.
    """
    def __init__(self):
        self.items = []
        self.pos = defaultdict(set)

    def insert(self, item):
        i = len(self.items)
        self.items.append(item)
        self.pos[item].add(i)

    def delete(self, item):
        if not self.pos[item]:
            # Item not present or already removed its last instance
            # Check if it exists anywhere else in pos, indicating multiple copies were inserted
            # and we are trying to delete one that doesn't match the last known position.
            # This basic implementation doesn't handle multiset counts perfectly on deletion.
            # For robust multiset deletion, counts or a different structure would be needed.
             return # Or raise an error if deletion of non-existent is critical

        try:
            i = self.pos[item].pop() # Get one position of the item
        except KeyError:
             # This can happen if the item was already fully removed via other means
             # or if defaultdict logic has issues.
             return

        if i >= len(self.items):
             # This indicates an inconsistent state, potentially due to prior errors
             # or complex interactions not fully handled by this simple structure.
             # Silently return or log an error.
             # Let's try to ensure the set is clean for the item if its position is invalid.
             if item in self.pos and not self.pos[item]:
                 del self.pos[item]
             return


        # If 'i' is the last element's position, just pop
        if i == len(self.items) - 1:
            self.items.pop()
            # If the set for 'item' becomes empty after popping 'i', remove the item key
            if item in self.pos and not self.pos[item]:
                 del self.pos[item]
            return

        # Otherwise, swap with the last element
        last_item = self.items.pop() # Remove last element first
        last_item_original_pos = len(self.items) # Its original position was the new length

        self.items[i] = last_item # Place last_item into the vacated spot 'i'

        # Update position tracking for the moved last_item
        if last_item in self.pos:
             # Remove its old position (end of the list)
             self.pos[last_item].discard(last_item_original_pos)
             # Add its new position 'i'
             self.pos[last_item].add(i)

        # Clean up the entry for the deleted item if its position set is now empty
        if item in self.pos and not self.pos[item]:
            del self.pos[item]


    def sample(self):
        """Return a random item, or None if empty."""
        if not self.items:
            return None
        # Ensure sampling only happens if items list is not empty
        try:
            return random.choice(self.items)
        except IndexError:
            # This could happen if items becomes empty between the check and choice
            # due to concurrency, though unlikely in this single-threaded context.
            # Or if self.items somehow became non-list temporarily (unlikely).
            return None

    def get_memory_estimate(self):
        # Basic memory estimation
        items_mem = sys.getsizeof(self.items) + sum(sys.getsizeof(x) for x in self.items)
        pos_mem = sys.getsizeof(self.pos)
        for k, v in self.pos.items():
            pos_mem += sys.getsizeof(k) + sys.getsizeof(v) + sum(sys.getsizeof(x) for x in v)
        return items_mem + pos_mem

# --- GraphConnectivitySketch Class ---
class GraphConnectivitySketch:
    """
    Maintains a dynamic undirected graph on n nodes via one L0Sampler per node.
    Supports edge insertions/deletions in O(1) time (amortized),
    and can extract a spanning forest in O(n) time by sampling each node once.
    """
    def __init__(self, n):
        self.n = n
        # Ensure n is non-negative
        if n < 0:
            raise ValueError("Number of nodes 'n' cannot be negative.")
        self.samplers = [L0Sampler() for _ in range(n)]
        self.edges = set() # Track edges to avoid double adding/deleting effects

    def _encode_edge(self, u, v):
        """Encode an undirected edge as an ordered pair for deduplication."""
        # Add boundary checks
        if not (0 <= u < self.n and 0 <= v < self.n):
             raise ValueError(f"Vertices {u}, {v} out of range [0, {self.n-1}]")
        if u == v:
             # Decide how to handle self-loops. Often ignored in connectivity.
             return None # Or raise an error if needed
        return tuple(sorted((u, v)))

    def add_edge(self, u, v):
        e = self._encode_edge(u, v)
        if e is None or e in self.edges: # Handle self-loops or existing edges
            return
        self.edges.add(e)

        # Check bounds before accessing samplers
        if 0 <= u < self.n and 0 <= v < self.n:
             self.samplers[u].insert(v)
             self.samplers[v].insert(u)
        else:
             # This case should ideally be caught by _encode_edge, but double-check.
             print(f"Warning: Attempted to add edge ({u}, {v}) with out-of-bounds node.")

    def del_edge(self, u, v):
        e = self._encode_edge(u, v)
        if e is None or e not in self.edges:
            return
        self.edges.remove(e)
        # Check bounds before accessing samplers
        if 0 <= u < self.n and 0 <= v < self.n:
             self.samplers[u].delete(v)
             self.samplers[v].delete(u)
        else:
             print(f"Warning: Attempted to delete edge ({u}, {v}) with out-of-bounds node.")


    def spanning_forest(self):
        """
        Extract a spanning forest using one L0 sample per node.
        This is like one round of the Ahn–Guha–McGregor algorithm:
        for each node u, sample a random neighbour v; if u and v are in
        different tree‐components, add (u,v) to the forest.
        Uses Union-Find data structure for efficiency.
        """
        if self.n == 0: return [] # Handle empty graph case

        parent = list(range(self.n))
        num_edges = 0
        # Path compression find
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i]) # Path compression
            return parent[i]

        # Union by rank/size is not strictly needed here, basic union works
        def union(i, j):
            nonlocal num_edges
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i # Simple union
                # Returning True indicates a successful merge that added an edge to the conceptual forest
                return True
            return False # Nodes were already in the same component

        forest_edges = []
        for u in range(self.n):
            if u >= len(self.samplers): # Safety check
                 print(f"Warning: Node index {u} out of bounds for samplers.")
                 continue
            v = self.samplers[u].sample()
            if v is None:
                continue
            # Ensure sampled neighbor v is within valid node range
            if not (0 <= v < self.n):
                print(f"Warning: Sampler for node {u} returned invalid neighbor {v}. Skipping.")
                continue

            if union(u, v):
                forest_edges.append(tuple(sorted((u, v)))) # Store edges consistently
                num_edges += 1


        return forest_edges


    def is_connected(self):
        """
        Check if the whole graph is likely connected by seeing if the extracted
        spanning forest has exactly n-1 edges.
        Requires n >= 1. Note: This is probabilistic, not guaranteed deterministic.
        """
        if self.n <= 0: return False # Or True if n=0 is considered connected
        if self.n == 1: return True

        # The AGM single-round approach doesn't guarantee finding a full spanning
        # tree even if one exists. It finds a spanning forest. Connectivity check
        # based *solely* on len(forest) == n-1 from *one round* is heuristic and
        # might give false negatives.
        # A more robust check would involve running the forest finding multiple times
        # or using the full Union-Find state after processing all samples.

        # We run the forest extraction and check the number of edges.
        # This matches the original code's logic but inherits its probabilistic nature.
        forest = self.spanning_forest()
        return len(forest) == self.n - 1

        # Alternative (more deterministic check using the final state of union-find):
        # Run spanning_forest to populate the parent array via union operations.
        # Then, count the number of distinct roots remaining.
        # parent = list(range(self.n)) # Need to re-init or use the one from spanning_forest
        # def find(i): ... # Define find again or pass parent array around
        # roots = {find(i) for i in range(self.n)}
        # return len(roots) == 1 if self.n > 0 else True


    def get_memory_estimate(self):
        base_mem = sys.getsizeof(self.n) + sys.getsizeof(self.edges) + sys.getsizeof(self.samplers)
        samplers_mem = sum(s.get_memory_estimate() for s in self.samplers)
        edges_mem = sum(sys.getsizeof(e) for e in self.edges)
        return base_mem + samplers_mem + edges_mem

# --- GraphLinearMeasurements Class ---
class GraphLinearMeasurements:
    """
    Uses m = O(n · polylog n) random linear measurements of the incidence stream.
    Supports edge insertions/deletions in O(m) time per update.
    NOTE: Decoding routines (connectivity, MST etc.) are placeholders.
    """
    def __init__(self, n, c=4):
        if n < 0: raise ValueError("Number of vertices n cannot be negative.")
        self.n = n

        if n <= 1:
             log_n_sq = 1 # Avoid log(0) or log(1) issues
        else:
             # Use max(1, ...) to ensure log_n_sq is at least 1 if n is small but > 1
             log_n_sq = max(1, math.log(n, 2)) ** 2

        # Ensure m is at least 1
        self.m = max(1, int(c * n * log_n_sq))

        # Using defaultdict for signs simplifies adding new edges encountered
        # Default factory assigns a random sign {+1, -1} to unseen edges
        self.signs = [defaultdict(lambda: random.choice([1, -1])) for _ in range(self.m)]
        self.y = [0.0] * self.m # The sketch vector

    def _encode_edge(self, u, v):
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError(f"Vertices {u}, {v} out of range [0, {self.n-1}]")
        if u == v:
            # Standard linear sketches often ignore self-loops or handle them specifically.
            # Raise error or return None to signal it shouldn't be processed.
            # raise ValueError("Self-loops are not typically handled in this basic model.")
             return None # Silently ignore self-loops
        return tuple(sorted((u, v)))

    def update(self, u, v, delta=1):
        """Process edge update (u, v) with change delta."""
        if delta == 0: return

        e = self._encode_edge(u, v)
        if e is None: return # Skip if self-loop or invalid edge

        for i in range(self.m):
            # Accessing signs[i][e] will create the entry with a random sign if it doesn't exist
            sign_i_e = self.signs[i][e]
            self.y[i] += delta * sign_i_e


    # --- Placeholder Decoding Functions ---
    # These require complex algorithms not implemented here.
    # They will return dummy values for the benchmark.

    def _placeholder_decode_sketch(self):
        """Placeholder: Real decoding is complex."""
        print("[WARN] Using placeholder decode_sketch. Returning empty candidate set.")
        # Simulate some work based on sketch size 'm' for timing? No, keep it fast.
        # time.sleep(0.001 * self.m / self.n if self.n > 0 else 0) # Avoid realistic simulation
        return [] # Cannot return a valid forest

    def _placeholder_decode_bipartiteness(self):
        """Placeholder: Real decoding needed."""
        print("[WARN] Using placeholder decode_bipartiteness. Returning default.")
        return True # Cannot determine bipartiteness

    def extract_spanning_forest(self):
        """Uses placeholder decoder."""
        print(f"[INFO] Calling placeholder decode_sketch for forest extraction...")
        candidate_edges = self._placeholder_decode_sketch() # Gets []

        # The rest of the function attempts to build a forest from candidates.
        # Since candidates is empty, it will return an empty forest.
        if self.n == 0: return []
        parent = list(range(self.n))
        # Simplified find without path compression for clarity in placeholder context
        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        def union(i, j, edge_data): # Added edge_data to potentially use weight 'w'
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i
                return True
            return False

        forest = []
        num_edges_in_forest = 0
        # Assuming candidate_edges format is (u, v, w) or (u, v)
        # Sort candidates if weights matter (e.g., for MST approximation)
        # sorted_candidates = sorted(candidate_edges, key=lambda x: x[2] if len(x) > 2 else 1)

        for edge_data in candidate_edges: # This loop will not run if placeholder returns []
            if len(edge_data) < 2: continue # Skip malformed entries
            u, v = edge_data[0], edge_data[1]
            # Basic validation
            if not (0 <= u < self.n and 0 <= v < self.n):
                 print(f"[WARN] Skipping invalid edge from decoder: ({u}, {v})")
                 continue

            if union(u, v, edge_data):
                forest.append(edge_data) # Append the original tuple/list
                num_edges_in_forest += 1
                # If building MST and candidates were sorted, could stop early
                # if num_edges_in_forest == self.n - 1: break # Only if n > 0

        # Returns empty list due to placeholder decoder
        return forest

    def is_connected(self):
        """Checks connectivity using placeholder decoder. Will likely return False."""
        if self.n <= 0: return False # Consistent with n=0 case
        if self.n == 1: return True
        # Relies on extract_spanning_forest, which uses the placeholder.
        # The placeholder returns [], so len(forest) will be 0.
        # Thus, this will return False unless n=1.
        forest = self.extract_spanning_forest()
        # Check number of edges found by the (placeholder) decoder
        num_edges_in_forest = len(forest)
        # The theoretical check is if the *true* MST has n-1 edges.
        # Our check is based on what the (placeholder) decoder returned.
        # This is NOT a reliable connectivity test with the placeholder.
        print("[WARN] is_connected result relies on placeholder decoder.")
        return num_edges_in_forest == self.n - 1


    def is_bipartite(self):
        """Checks bipartiteness using placeholder decoder."""
        print(f"[INFO] Calling placeholder decode_bipartiteness...")
        print("[WARN] is_bipartite result relies on placeholder decoder.")
        return self._placeholder_decode_bipartiteness() # Returns True currently

    def approx_mst_weight(self):
        """Approximates MST weight using placeholder decoder. Returns 0."""
        # This relies on extract_spanning_forest which uses the placeholder.
        # Placeholder returns [], so the sum will be 0.
        forest = self.extract_spanning_forest()
        # Assumes forest edges are tuples like (u, v, w) where w is weight
        # If weights are not present, assumes weight 1 or requires modification
        total_weight = sum(edge[2] for edge in forest if len(edge) > 2) # Sum weights if available
        # If format is just (u,v), the sum is 0. If (u,v,w) is returned by decoder, it sums w.
        # Since placeholder returns [], sum is 0.
        print("[WARN] approx_mst_weight result relies on placeholder decoder.")
        return total_weight

    def get_memory_estimate(self):
        # Estimate memory for the sketch vector y and the sign dictionaries
        y_mem = sys.getsizeof(self.y) + sum(sys.getsizeof(x) for x in self.y)
        signs_mem = sys.getsizeof(self.signs)
        for sign_dict in self.signs:
            signs_mem += sys.getsizeof(sign_dict)
            # Estimate memory of keys (edges) and values (signs) within each dict
            # This can be expensive if dicts are large, do a sample or approximation
            # For simplicity, just add size of dict object itself
            # A more detailed sum:
            # for k, v in sign_dict.items():
            #    signs_mem += sys.getsizeof(k) + sys.getsizeof(v)
        return y_mem + signs_mem

# --- DSU Class (for Boruvka, Kruskal) ---
class DSU:
    # ... (DSU class implementation from original code) ...
    def __init__(self, n):
        if n < 0: raise ValueError("Number of elements n cannot be negative.")
        self.parent = list(range(n))
        self.size = [1] * n # Used for union by size/rank heuristic
        self.num_sets = n # Number of disjoint sets

    def find(self, i):
        # Find with path compression
        if i < 0 or i >= len(self.parent):
             raise IndexError("DSU index out of bounds.")
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i]) # Path compression
        return self.parent[i]

    def union(self, i, j):
        # Union by size
        if i < 0 or i >= len(self.parent) or j < 0 or j >= len(self.parent):
             raise IndexError("DSU index out of bounds.")
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Merge smaller tree into larger tree
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i # Swap roots
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.num_sets -= 1
            return True # Return True if merge happened
        return False # Return False if already in the same set

    def get_num_sets(self):
        return self.num_sets

# --- ModifiedBoruvkaSketching Class ---
class ModifiedBoruvkaSketching:
    """
    Conceptual: Leverages sketches to find minimum weight edge out of components.
    Actual Implementation: Uses a fallback simple Boruvka on the full edge list
                           stored in a heap, as sketch integration is not implemented.
    """
    def __init__(self, num_nodes):
        if num_nodes < 0: raise ValueError("Number of nodes cannot be negative.")
        self.num_nodes = num_nodes
        # Store edges in a list or heap if processing offline after adding all edges
        self.edges = [] # Using a list, will convert to heap or sort later if needed
        print(f"Initialized ModifiedBoruvkaSketching for {self.num_nodes} nodes.")
        # Placeholder for conceptual sketches (not used in current find_spanning_tree)
        # self.sketches_per_node = [...]

    def add_edge(self, u, v, weight):
        """Adds edge (for later processing)."""
        if not (0 <= u < self.num_nodes and 0 <= v < self.num_nodes):
             print(f"Warning: Edge ({u}, {v}) ignored, nodes out of range [0, {self.num_nodes-1}]")
             return
        if u == v: return # Ignore self-loops for standard MST
        # Store edge with weight first for easy sorting/heap processing
        heapq.heappush(self.edges, (weight, u, v))

    def delete_edge(self, u, v, weight):
        """Conceptual deletion. Not supported by the current fallback implementation."""
        print(f"[WARN] delete_edge not supported in fallback Boruvka. Edge ({u},{v},{weight}) ignored.")
        # To support deletion, would need to rebuild the edge list/heap or use complex structures.


    def find_spanning_tree(self):
        """
        Computes MST using the fallback Boruvka's algorithm on the stored edges.
        Does NOT use the conceptual 'sketches' mentioned in comments/theory.
        """
        if self.num_nodes == 0: return [], 0
        dsu = DSU(self.num_nodes)
        mst_edges = []
        mst_weight = 0
        # Make a copy or work with the heap directly if edge stream was large
        # For simplicity, assume self.edges contains all edges now.
        # Convert the list `self.edges` into a structure suitable for Boruvka rounds.
        # A simple list that we iterate through each round works, but is inefficient.
        # A copy of the edges allows filtering as the algorithm progresses.
        edge_list_for_rounds = sorted(self.edges) # Sort once if iterating multiple times

        print(f"\nStarting Fallback Boruvka's Algorithm on {len(edge_list_for_rounds)} edges...")
        num_components = self.num_nodes

        while num_components > 1:
            # cheapest_edge_map[component_root] = (min_weight, u, v)
            cheapest_edge_map = {}

            # Find the cheapest edge leaving each component
            # Iterate through *all* remaining edges (inefficient part of fallback)
            edges_processed_this_round = 0
            for weight, u, v in edge_list_for_rounds: # Inefficiently iterates all edges each round
                root_u = dsu.find(u)
                root_v = dsu.find(v)
                edges_processed_this_round += 1

                if root_u != root_v:
                    # Edge connects two different components
                    # Update cheapest edge for component root_u if this edge is cheaper
                    if root_u not in cheapest_edge_map or weight < cheapest_edge_map[root_u][0]:
                        cheapest_edge_map[root_u] = (weight, u, v)
                    # Update cheapest edge for component root_v if this edge is cheaper
                    if root_v not in cheapest_edge_map or weight < cheapest_edge_map[root_v][0]:
                        cheapest_edge_map[root_v] = (weight, u, v)

            print(f"Round: {self.num_nodes - num_components + 1}, Components: {num_components}, Edges considered: {edges_processed_this_round}, Cheapest edges found: {len(cheapest_edge_map)}")

            if not cheapest_edge_map:
                 # No edges found connecting components, graph must be disconnected
                 print("Warning: No connecting edges found. Graph may be disconnected.")
                 break # Exit loop

            num_merges_this_round = 0
            # Process the cheapest edges found for each component to merge them
            for root_node, edge_data in cheapest_edge_map.items():
                weight, u, v = edge_data
                # Perform union operation. DSU handles check if already merged.
                if dsu.union(u, v):
                    # print(f"  Merging components of {u} and {v} using edge ({u}, {v}, {weight})")
                    mst_edges.append((u, v, weight))
                    mst_weight += weight
                    num_merges_this_round += 1
                    num_components -= 1 # Decrement component count upon successful merge


            if num_merges_this_round == 0:
                # This happens if all cheapest edges connect components already merged
                # in *this same round* by another edge. Graph might be disconnected if num_components > 1.
                 if num_components > 1:
                      print("Warning: No merges occurred in this round, but multiple components remain.")
                 break # Avoid infinite loop

        print("Boruvka's algorithm finished.")
        if num_components == 1:
            print(f"Spanning tree found with total weight: {mst_weight:.2f}")
        else:
            print(f"Could not find a spanning tree. {num_components} components remain.")
            # Return the forest found so far
        return mst_edges, mst_weight


    def get_memory_estimate(self):
        # Memory is dominated by the stored edges heap/list
        base_mem = sys.getsizeof(self.num_nodes) + sys.getsizeof(self.edges)
        edges_mem = sum(sys.getsizeof(e) + sys.getsizeof(e[0]) + sys.getsizeof(e[1]) + sys.getsizeof(e[2]) for e in self.edges)
        return base_mem + edges_mem


# --- AdaptiveGraphSketch Class ---
class AdaptiveGraphSketch:
    """
    Implements adaptive sparsification via random sampling over multiple rounds.
    Computes exact MST and approximate Max-Weight Matching on the final sparsifier.
    """
    def __init__(self, n, gamma=0.1):
        if n < 0: raise ValueError("Number of vertices n cannot be negative.")
        if not (0 < gamma <= 1):
            raise ValueError("gamma must be between 0 (exclusive) and 1 (inclusive)")
        self.n = n
        self.gamma = gamma
        # Number of rounds, ensure at least 1
        self.rounds = max(1, int(math.ceil(1.0 / gamma)))

        # Store graph as adjacency list: {u: {v: weight, ...}}
        self.current_graph = defaultdict(dict)
        self.edge_count = 0 # Number of unique edges in current_graph

    def stream_update(self, u, v, w):
        """Add/update edge (u, v) with weight w."""
        if not (0 <= u < self.n and 0 <= v < self.n):
            # Silently ignore or raise error for out-of-bounds nodes
            # print(f"Warning: Skipping update for edge ({u},{v}), nodes out of range [0, {self.n-1}]")
            return
        if u == v: return # Ignore self-loops

        # Check if edge is new before adding to avoid double counting
        is_new_edge = v not in self.current_graph[u]

        # Update/add edge in both directions for undirected graph
        self.current_graph[u][v] = w
        self.current_graph[v][u] = w

        if is_new_edge:
            self.edge_count += 1

    def build_sparsifier(self):
        """Performs one round of random edge sampling sparsification."""
        if self.n <= 1 or self.edge_count == 0:
            # No sparsification needed for trivial graphs or if no edges
            return

        # Sampling probability p = n^{-gamma} (or 1 if n=1)
        # Ensure p is not zero and handle potential floating point issues
        p = self.n ** (-self.gamma) if self.n > 1 else 1.0
        # Clamp p to avoid issues if gamma is very large or n is huge
        p = max(min(p, 1.0), 1e-9) # Avoid p=0 division error, ensure p<=1

        new_graph = defaultdict(dict)
        new_edge_count = 0
        processed_edges = set() # Track processed edges to handle undirected nature

        for u, neighbors in self.current_graph.items():
            for v, w in neighbors.items():
                # Ensure we process each undirected edge only once
                edge = tuple(sorted((u, v)))
                if edge in processed_edges:
                    continue
                processed_edges.add(edge)

                # Sample the edge
                if random.random() < p:
                    # Keep the edge and rescale its weight
                    rescaled_w = w / p
                    new_graph[u][v] = rescaled_w
                    new_graph[v][u] = rescaled_w
                    new_edge_count += 1

        # Update graph and edge count
        self.current_graph = new_graph
        self.edge_count = new_edge_count


    def finalize(self):
        """Runs sparsification rounds and computes final results."""
        print(f"[INFO] AdaptiveSketch: Starting {self.rounds} rounds of sparsification (gamma={self.gamma})...")
        initial_edge_count = self.edge_count
        for r in range(self.rounds):
            round_start_edges = self.edge_count
            self.build_sparsifier()
            print(f"  Round {r+1}/{self.rounds}: Sparsified from {round_start_edges} to {self.edge_count} edges.")
            if self.edge_count == 0: break # Stop if no edges left

        print(f"[INFO] Final sparsifier has {self.edge_count} edges (started with {initial_edge_count}).")

        print("[INFO] Computing exact MST on the sparsifier...")
        start_time = time.time()
        mst_weight = self._exact_mst()
        mst_time = time.time() - start_time
        print(f"[INFO]   MST Weight on sparsifier: {mst_weight:.2f} (computed in {mst_time:.4f}s)")

        print("[INFO] Computing approximate max-weight matching on the sparsifier...")
        start_time = time.time()
        matching = self._approx_max_matching()
        matching_time = time.time() - start_time
        matching_weight = sum(w for _, _, w in matching)
        print(f"[INFO]   Approx Max-Weight Matching size: {len(matching)}, Total Weight: {matching_weight:.2f} (computed in {matching_time:.4f}s)")

        # Return results and timing info perhaps
        return mst_weight, matching, mst_time, matching_time

    def _get_edges_from_sparsifier(self):
        """Helper to extract unique edges (weight, u, v) for sorting."""
        edges = []
        processed = set()
        for u, neighbors in self.current_graph.items():
            for v, w in neighbors.items():
                edge = tuple(sorted((u, v)))
                if edge not in processed:
                    # Store weight first for easy sorting by weight
                    edges.append((w, u, v))
                    processed.add(edge)
        return edges

    def _exact_mst(self):
        """Computes exact MST weight on the current sparsifier using Kruskal."""
        if self.n == 0 or self.edge_count == 0: return 0.0

        edges = self._get_edges_from_sparsifier()
        edges.sort() # Sort by weight (ascending)

        dsu = DSU(self.n)
        mst_total_weight = 0.0
        mst_edge_count = 0
        max_edges_in_mst = self.n - 1 if self.n > 0 else 0

        for w, u, v in edges:
            if dsu.union(u, v):
                mst_total_weight += w
                mst_edge_count += 1
                if mst_edge_count == max_edges_in_mst:
                    # Found a spanning tree (if graph was connected)
                    break # Optimization

        # Check if a spanning tree was actually formed
        # if mst_edge_count != max_edges_in_mst and self.n > 1:
        #     # This can happen if the sparsifier became disconnected
        #     print(f"[WARN] Sparsifier may be disconnected; MST calculation formed a forest with {mst_edge_count} edges.")

        return mst_total_weight

    def _approx_max_matching(self):
        """Computes 1/2-approx max weight matching using greedy approach."""
        if self.n == 0 or self.edge_count == 0: return []

        edges = self._get_edges_from_sparsifier()
        # Sort by weight descending for greedy selection
        edges.sort(key=lambda x: x[0], reverse=True)

        matched = [False] * self.n # Track matched vertices
        matching_result = []

        for w, u, v in edges:
            # If both endpoints are available, add edge to matching
            if not matched[u] and not matched[v]:
                matched[u] = True
                matched[v] = True
                matching_result.append((u, v, w)) # Store edge info

        return matching_result


    def get_memory_estimate(self):
        # Memory is dominated by the current_graph structure
        graph_mem = sys.getsizeof(self.current_graph)
        edge_mem = 0
        for u, neighbors in self.current_graph.items():
            graph_mem += sys.getsizeof(u) + sys.getsizeof(neighbors)
            for v, w in neighbors.items():
                graph_mem += sys.getsizeof(v) + sys.getsizeof(w)
                # Edge data is stored twice (u->v and v->u), count size once roughly
                edge_mem += sys.getsizeof(v) + sys.getsizeof(w) # Approximate

        # Other small variables
        other_mem = sys.getsizeof(self.n) + sys.getsizeof(self.gamma) + \
                    sys.getsizeof(self.rounds) + sys.getsizeof(self.edge_count)

        # Return sum, graph_mem includes overhead of dicts/sets
        # A simple estimate might be proportional to n + edge_count
        estimated_size = sys.getsizeof(self.current_graph) + self.edge_count * (sys.getsizeof(0) * 2 + sys.getsizeof(0.0)) # Rough estimate
        return estimated_size


# --- SimpleSetSketch Class ---
class SimpleSetSketch:
    def __init__(self, sketch_size, num_hashes=3):
        if sketch_size <= 0: raise ValueError("Sketch size must be positive")
        if num_hashes <= 0: raise ValueError("Number of hashes must be positive")

        self.size = sketch_size
        self.num_hashes = num_hashes
        self.sketch_array = [0] * self.size
        # Use fixed seeds for reproducibility if needed, or random per instance
        self._hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(num_hashes)]

    def _get_hash_indices(self, element):
        """Calculates hash indices and integer representation for XORing."""
        indices = set()
        # Convert element to bytes consistently
        try:
            if isinstance(element, bytes):
                element_bytes = element
            elif isinstance(element, str):
                element_bytes = element.encode('utf-8')
            elif isinstance(element, int):
                 element_bytes = element.to_bytes((element.bit_length() + 7) // 8, 'big')
                 if element == 0: element_bytes = b'\x00' # Handle zero case
            elif isinstance(element, tuple):
                 # Hash tuple representation; assumes elements are hashable
                 element_bytes = str(element).encode('utf-8') # Simple string conversion
            else:
                # Fallback: use string representation
                element_bytes = str(element).encode('utf-8')
        except Exception as e:
             print(f"Warning: Could not convert element {element} to bytes: {e}. Using hash().")
             # Fallback if direct byte conversion fails
             element_bytes = str(hash(element)).encode('utf-8')


        # Use hash() for XOR value, try to make it consistent
        try:
            element_int_repr = hash(element)
        except TypeError:
            # If element is not hashable (e.g., list), use its string representation's hash
            element_int_repr = hash(str(element))

        # Generate hash indices using the seeds
        for i in range(self.num_hashes):
            # Combine element bytes and seed, then hash
            hasher = hashlib.sha256() # Use a standard hash like SHA-256
            hasher.update(element_bytes)
            hasher.update(self._hash_seeds[i].to_bytes(4, 'big')) # Mix in seed
            hash_val = int(hasher.hexdigest(), 16)
            indices.add(hash_val % self.size) # Map to sketch index

        return list(indices), element_int_repr

    def add(self, element):
        """Adds an element to the sketch using XOR."""
        indices, element_int_repr = self._get_hash_indices(element)
        for i in indices:
            if 0 <= i < self.size: # Bounds check
                 self.sketch_array[i] ^= element_int_repr

    def merge(self, other_sketch):
        """Merges another sketch into this one using XOR."""
        if self.size != other_sketch.size or self.num_hashes != other_sketch.num_hashes:
            raise ValueError("Sketches must have the same size and number of hashes.")
        # Also might want to check if hash seeds are the same if strict merging is required
        # if self._hash_seeds != other_sketch._hash_seeds:
        #     print("Warning: Merging sketches with different hash seeds.")

        for i in range(self.size):
            self.sketch_array[i] ^= other_sketch.sketch_array[i]

    def get_sketch(self):
        """Returns the current sketch array."""
        return list(self.sketch_array) # Return a copy

    def get_memory_estimate(self):
        # Memory is dominated by the sketch array
        array_mem = sys.getsizeof(self.sketch_array) + sum(sys.getsizeof(x) for x in self.sketch_array)
        seeds_mem = sys.getsizeof(self._hash_seeds) + sum(sys.getsizeof(x) for x in self._hash_seeds)
        other_mem = sys.getsizeof(self.size) + sys.getsizeof(self.num_hashes)
        return array_mem + seeds_mem + other_mem

# --- CountMinSketch Class ---
class CountMinSketch:
    def __init__(self, width, depth, seed=None):
        if not (width > 0 and depth > 0):
            raise ValueError("Width and depth must be positive integers")
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

        # Initialize hash seeds based on the provided seed for reproducibility
        if seed is not None:
             random.seed(seed)
        self._hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(depth)]

    def _get_hashes(self, item):
        """Get hash indices for the item for each row (depth)."""
        hashes = []
        # Convert item to bytes consistently for hashing
        try:
            item_str = str(item)
            item_bytes = item_str.encode('utf-8')
        except Exception as e:
             print(f"Warning: Could not encode item {item} to bytes: {e}. Using fallback.")
             item_bytes = b"fallback_item_representation"

        for i in range(self.depth):
            hasher = hashlib.sha256()
            hasher.update(item_bytes)
            hasher.update(self._hash_seeds[i].to_bytes(4, 'big')) # Mix in seed
            hash_val = int(hasher.hexdigest(), 16)
            hashes.append(hash_val % self.width) # Map to table width
        return hashes

    def add(self, item, count=1):
        """Increment the count for an item."""
        if count == 0: return
        hashes = self._get_hashes(item)
        for i in range(self.depth):
             if 0 <= hashes[i] < self.width: # Bounds check
                 self.table[i][hashes[i]] += count
             else: # Should not happen with modulo arithmetic, but safety first
                 print(f"Warning: Hash index {hashes[i]} out of bounds for width {self.width}.")

    def estimate(self, item):
        """Estimate the count (frequency) of an item."""
        hashes = self._get_hashes(item)
        min_count = float('inf')
        for i in range(self.depth):
             if 0 <= hashes[i] < self.width: # Bounds check
                 min_count = min(min_count, self.table[i][hashes[i]])
             else: # Should not happen
                 print(f"Warning: Hash index {hashes[i]} out of bounds during estimation.")
                 return 0 # Return 0 or raise error if index is invalid
        # Handle case where item was never added (min_count remains inf)
        return min_count if min_count != float('inf') else 0

    def get_memory_estimate(self):
        # Memory is dominated by the table
        table_mem = sys.getsizeof(self.table)
        for row in self.table:
            table_mem += sys.getsizeof(row) + sum(sys.getsizeof(x) for x in row)
        seeds_mem = sys.getsizeof(self._hash_seeds) + sum(sys.getsizeof(x) for x in self._hash_seeds)
        other_mem = sys.getsizeof(self.width) + sys.getsizeof(self.depth)
        return table_mem + seeds_mem + other_mem

# --- gSketch Class ---
class gSketch:
    """
    Simulates gSketch using partitioned CountMinSketches for localized queries
    (e.g., node degree estimation).
    """
    def __init__(self, num_partitions, cms_width, cms_depth, node_ids=None, query_workload=None):
        if num_partitions <= 0: raise ValueError("Number of partitions must be positive.")
        if cms_width <= 0 or cms_depth <= 0: raise ValueError("CMS width and depth must be positive.")

        self.num_partitions = num_partitions
        # Create separate CMS for each partition with unique seeds
        self.partitions = [CountMinSketch(cms_width, cms_depth, seed=i) for i in range(num_partitions)]
        # Cache node to partition mapping for efficiency
        self.node_to_partition = {}

        print(f"Initialized gSketch with {num_partitions} partitions.")
        print(f"Each partition uses a CountMinSketch(w={cms_width}, d={cms_depth}).")

        # Conceptual handling of query workload (not implemented)
        if query_workload:
            print("Conceptual: Query workload provided (details not used in this impl).")
            # Logic would go here to potentially adjust partitioning or CMS parameters
            pass # Placeholder

        # Pre-assign partitions if node IDs are known upfront
        if node_ids:
            print(f"Pre-assigning {len(node_ids)} nodes to partitions...")
            for node_id in node_ids:
                self._assign_partition(node_id) # Assigns and caches

    def _get_partition_index(self, node_id):
        """Determines partition index using a hash function."""
        # Use a stable hash based on the node ID
        try:
            node_str = str(node_id)
            node_bytes = node_str.encode('utf-8')
        except Exception as e:
             print(f"Warning: Could not encode node ID {node_id} to bytes: {e}. Using fallback.")
             node_bytes = b"fallback_node_id"

        hasher = hashlib.sha256()
        hasher.update(node_bytes)
        # Add a fixed salt/seed specific to gSketch partitioning if needed
        # hasher.update(b"gSketchPartitionSalt")
        hash_val = int(hasher.hexdigest(), 16)
        return hash_val % self.num_partitions

    def _assign_partition(self, node_id):
        """Assigns node to a partition if not already assigned. Returns partition index."""
        if node_id not in self.node_to_partition:
            part_idx = self._get_partition_index(node_id)
            self.node_to_partition[node_id] = part_idx
        return self.node_to_partition[node_id]

    def add_edge(self, u, v, weight=1):
        """
        Processes edge update for degree estimation.
        Increments degree count for both u and v in their respective partition sketches.
        """
        if weight == 0: return # No change

        # Assign nodes to partitions (uses cache if already assigned)
        part_idx_u = self._assign_partition(u)
        part_idx_v = self._assign_partition(v)

        # Update the CMS in the assigned partition for each node.
        # The item added to CMS is the node ID itself, representing its degree count.
        if 0 <= part_idx_u < self.num_partitions:
            self.partitions[part_idx_u].add(u, weight) # Increment degree count for u
        else: # Should not happen
             print(f"Warning: Invalid partition index {part_idx_u} for node {u}.")

        # Avoid double counting if u and v are the same node (self-loop)
        # and fall into the same partition (though self-loops often ignored)
        if u == v: return # Or handle self-loops based on requirements

        if 0 <= part_idx_v < self.num_partitions:
            self.partitions[part_idx_v].add(v, weight) # Increment degree count for v
        else: # Should not happen
             print(f"Warning: Invalid partition index {part_idx_v} for node {v}.")


    def estimate_degree(self, node_id):
        """Estimates degree of node_id using its assigned partition's CMS."""
        if node_id not in self.node_to_partition:
            # Node was never added via an edge, or not pre-assigned.
            # Assign it now to find its potential partition, but estimate will be 0.
            print(f"Info: Node {node_id} not seen before. Assigning partition. Degree estimated as 0.")
            self._assign_partition(node_id) # Assign for consistency, though degree is 0
            return 0

        part_idx = self.node_to_partition[node_id]
        if 0 <= part_idx < self.num_partitions:
            # Estimate the count of 'node_id' in its partition's CMS
            estimated_degree = self.partitions[part_idx].estimate(node_id)
            return estimated_degree
        else: # Should not happen
             print(f"Warning: Node {node_id} has invalid partition index {part_idx}. Returning 0.")
             return 0

    def get_memory_estimate(self):
        base_mem = sys.getsizeof(self.num_partitions) + sys.getsizeof(self.partitions) + \
                   sys.getsizeof(self.node_to_partition)
        partitions_mem = sum(p.get_memory_estimate() for p in self.partitions)
        # Estimate node_to_partition dict memory (crude)
        map_mem = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.node_to_partition.items())
        return base_mem + partitions_mem + map_mem


# --- GraphVisualSketchHierarchy Class ---
class GraphVisualSketchHierarchy:
    """Builds a multi-level graph hierarchy by iterative node aggregation."""
    def __init__(self, original_graph_adj):
        """Initializes with the original graph (Level 0)."""
        if not isinstance(original_graph_adj, dict):
            raise TypeError("original_graph_adj must be a dictionary (adjacency list).")

        # Ensure values are sets for consistent neighbor representation
        self.levels = {0: self._deep_copy_adj_to_sets(original_graph_adj)}
        self.max_level = 0
        # node_mapping[level][supernode_id] = {original_node_id1, ...}
        self.node_mapping = {0: {node: {node} for node in original_graph_adj}}

        print("Initialized Visual Sketch Hierarchy with Level 0.")

    def _deep_copy_adj_to_sets(self, adj):
        # Creates a deep copy ensuring neighbors are stored in sets
        copied_adj = {}
        for node, neighbors in adj.items():
            try:
                 copied_adj[node] = set(neighbors)
            except TypeError:
                 print(f"Warning: Neighbors of node {node} is not iterable. Setting to empty set.")
                 copied_adj[node] = set()
        return copied_adj

    def _create_next_level(self, current_level_adj, max_nodes_per_supernode=None):
        """
        Creates the next hierarchy level by merging connected nodes/supernodes.
        Simplified Strategy: Iteratively merge a node with one of its neighbors
                             if constraints (like max_nodes_per_supernode) are met.
                             This is a basic approach; more sophisticated methods exist
                             (e.g., using community detection, label propagation).
        """
        current_level = self.max_level
        print(f"\nAttempting to create Level {current_level + 1}...")
        if not current_level_adj:
             print("Current level is empty. Cannot create next level.")
             return None, None

        nodes = list(current_level_adj.keys())
        random.shuffle(nodes) # Process nodes in random order to avoid bias

        supernode_counter = 0
        new_adj = {} # Adjacency list for the next level
        new_node_mapping = {} # Mapping from new supernodes to original nodes
        node_to_supernode = {} # Maps nodes from current level to their new supernode ID
        merged_nodes = set() # Track nodes already merged into a supernode

        for node in nodes:
            if node in merged_nodes:
                continue # Skip node if already part of a supernode

            # Start a new supernode with the current node
            current_supernode_id = f"S{current_level + 1}_{supernode_counter}"
            # Get original nodes represented by this current level 'node'
            original_nodes_in_current = self.node_mapping[current_level].get(node, {node})
            supernode_members = {node} # Nodes from current level in this new supernode
            original_nodes_in_supernode = set(original_nodes_in_current) # Use set copy
            node_to_supernode[node] = current_supernode_id
            merged_nodes.add(node)

            # Try to merge neighbors into this supernode
            # Use a queue or list for neighbors to check for merging
            candidates_to_merge = list(current_level_adj.get(node, set()))
            processed_candidates = {node} # Track neighbors considered for merging with this supernode

            while candidates_to_merge:
                neighbor = candidates_to_merge.pop(0)

                if neighbor in merged_nodes or neighbor in supernode_members:
                    continue # Already merged or part of the current group

                # Check merge constraints
                should_merge = True
                neighbor_original_nodes = self.node_mapping[current_level].get(neighbor, {neighbor})

                # Constraint: Max original nodes per supernode
                if max_nodes_per_supernode and \
                   (len(original_nodes_in_supernode) + len(neighbor_original_nodes)) > max_nodes_per_supernode:
                    should_merge = False

                # Add other constraints here if needed (e.g., based on edge weights, community structure)

                if should_merge:
                    # Merge neighbor into the current supernode
                    supernode_members.add(neighbor)
                    original_nodes_in_supernode.update(neighbor_original_nodes)
                    node_to_supernode[neighbor] = current_supernode_id
                    merged_nodes.add(neighbor)

                    # Add neighbors of the merged node to the candidate list, if they aren't processed yet
                    for next_neighbor in current_level_adj.get(neighbor, set()):
                         if next_neighbor not in merged_nodes and next_neighbor not in processed_candidates:
                             candidates_to_merge.append(next_neighbor)
                             processed_candidates.add(next_neighbor) # Mark as considered for this merge group
                # Mark this neighbor as processed for the current supernode attempt
                processed_candidates.add(neighbor)


            # Finalize the new supernode
            if supernode_members: # Should always be true if loop started
                new_adj[current_supernode_id] = set() # Initialize neighbors set
                new_node_mapping[current_supernode_id] = original_nodes_in_supernode
                supernode_counter += 1

        # Check if any aggregation actually happened
        if len(new_adj) == len(current_level_adj) and len(merged_nodes) == len(current_level_adj):
            print("No significant aggregation occurred. Stopping hierarchy creation.")
            return None, None # Indicate no new level was effectively created

        print(f"Aggregated {len(current_level_adj)} nodes/supernodes into {len(new_adj)} new supernodes.")

        # Create edges between the new supernodes based on original connections
        print("Creating edges between new supernodes...")
        edges_added_count = 0
        for u_orig, neighbors in current_level_adj.items():
            # Find the supernode containing u_orig (if any)
            if u_orig not in node_to_supernode: continue
            u_super = node_to_supernode[u_orig]

            for v_orig in neighbors:
                # Find the supernode containing v_orig (if any)
                if v_orig not in node_to_supernode: continue
                v_super = node_to_supernode[v_orig]

                # If u_orig and v_orig are in different supernodes, add an edge between them
                if u_super != v_super:
                    # Add edge if not already present (using sets handles this)
                    if v_super not in new_adj[u_super]:
                         new_adj[u_super].add(v_super)
                         new_adj[v_super].add(u_super) # Assuming undirected
                         edges_added_count += 1

        print(f"Added {edges_added_count} edges between supernodes in Level {current_level + 1}.")
        return new_adj, new_node_mapping

    def build_hierarchy(self, max_levels=5, max_nodes_per_supernode=None):
        """Builds multiple levels of the hierarchy."""
        print(f"\nBuilding Visual Sketch Hierarchy up to {max_levels} levels...")
        if max_nodes_per_supernode:
            print(f"Constraint: Max {max_nodes_per_supernode} original nodes per supernode.")

        while self.max_level < max_levels - 1:
            current_adj = self.levels.get(self.max_level)
            if not current_adj:
                print(f"Stopping: Level {self.max_level} is empty.")
                break

            next_adj, next_mapping = self._create_next_level(
                current_adj,
                max_nodes_per_supernode=max_nodes_per_supernode
            )

            if next_adj is None:
                # No effective aggregation happened, stop building levels
                break

            # Successfully created a new level
            self.max_level += 1
            self.levels[self.max_level] = next_adj
            self.node_mapping[self.max_level] = next_mapping

        print(f"\nHierarchy built up to Level {self.max_level}.")

    def get_level_view(self, level):
        """Returns the graph adjacency list at a specific level."""
        if level < 0 or level > self.max_level:
             print(f"Warning: Level {level} does not exist (max level is {self.max_level}).")
             return None
        return self.levels.get(level)

    def get_original_nodes(self, level, supernode_id):
        """Gets the set of original node IDs within a supernode at a given level."""
        if level < 0 or level > self.max_level:
             print(f"Warning: Level {level} does not exist.")
             return None
        if level == 0:
             # At level 0, the ID is the original node ID itself
             # Check if it exists in the original mapping
             return {supernode_id} if supernode_id in self.node_mapping[0] else None

        # For levels > 0, look up the supernode ID in the mapping for that level
        if level in self.node_mapping and supernode_id in self.node_mapping[level]:
            return self.node_mapping[level][supernode_id]
        else:
             print(f"Warning: Supernode ID '{supernode_id}' not found at level {level}.")
             return None

    def get_memory_estimate(self):
        total_mem = sys.getsizeof(self.levels) + sys.getsizeof(self.node_mapping) + sys.getsizeof(self.max_level)
        # Estimate memory for levels dictionary
        for level, adj in self.levels.items():
            total_mem += sys.getsizeof(level) + sys.getsizeof(adj)
            for node, neighbors in adj.items():
                total_mem += sys.getsizeof(node) + sys.getsizeof(neighbors)
                total_mem += sum(sys.getsizeof(n) for n in neighbors) # Memory for neighbors themselves

        # Estimate memory for node_mapping dictionary
        for level, mapping in self.node_mapping.items():
            total_mem += sys.getsizeof(level) + sys.getsizeof(mapping)
            for supernode, original_nodes in mapping.items():
                total_mem += sys.getsizeof(supernode) + sys.getsizeof(original_nodes)
                total_mem += sum(sys.getsizeof(n) for n in original_nodes) # Memory for original node IDs

        return total_mem


# --- END OF ORIGINAL CLASSES (with modifications) ---


# --- Comparison Framework ---
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np # Needed for accuracy calculations

def generate_graph(n, m_target, model='ba', seed=None):
    """Generates a NetworkX graph."""
    print(f"\nGenerating graph: N={n}, Target Edges≈{m_target}, Model='{model}'...")
    if model == 'ba': # Barabási-Albert model
        # Parameter 'm' for BA is edges to attach from a new node.
        # Target edges M ≈ n * m (roughly). So m ≈ M / n.
        m_param = max(1, int(m_target / n))
        G = nx.barabasi_albert_graph(n, m_param, seed=seed)
        print(f"Generated BA graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (m_param={m_param}).")
    elif model == 'gnp': # Erdős-Rényi model
        # Probability p = M / (N*(N-1)/2)
        if n <= 1: p = 0.0
        else: p = min(1.0, m_target / (n * (n - 1) / 2.0))
        G = nx.gnp_random_graph(n, p, seed=seed)
        print(f"Generated GNP graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (p={p:.4f}).")
    else:
        raise ValueError(f"Unknown graph model: {model}")

    # Ensure graph is connected for certain tests (optional, might take time)
    # if not nx.is_connected(G):
    #     print("Graph is not connected. Trying to get largest component or add edges...")
    #     # Option 1: Use the largest connected component
    #     largest_cc = max(nx.connected_components(G), key=len)
    #     G = G.subgraph(largest_cc).copy()
    #     print(f"Using largest connected component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        # Option 2: Add edges to connect components (more complex)

    # Add random weights for algorithms that use them
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(1.0, 100.0)

    return G

def get_ground_truth(G):
    """Computes ground truth values for the graph G."""
    print("\nCalculating ground truth values...")
    truth = {}
    start_time = time.time()
    truth['is_connected'] = nx.is_connected(G)
    print(f"  Connectivity: {truth['is_connected']} (in {time.time() - start_time:.4f}s)")

    start_time = time.time()
    if truth['is_connected']:
         # Use networkx MST function which uses Kruskal/Prim depending on density
         mst = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal') # Or 'prim'
         truth['mst_weight'] = mst.size(weight='weight')
         truth['mst_edges'] = mst.number_of_edges()
    else:
         # For disconnected graphs, find MST weight of the forest
         truth['mst_weight'] = 0
         truth['mst_edges'] = 0
         num_components = 0
         for component_nodes in nx.connected_components(G):
             subgraph = G.subgraph(component_nodes)
             num_components += 1
             if subgraph.number_of_nodes() > 0:
                  mst_comp = nx.minimum_spanning_tree(subgraph, weight='weight')
                  truth['mst_weight'] += mst_comp.size(weight='weight')
                  truth['mst_edges'] += mst_comp.number_of_edges()
         truth['num_components'] = num_components

    print(f"  MST Weight (or Forest): {truth['mst_weight']:.2f} ({truth['mst_edges']} edges) (in {time.time() - start_time:.4f}s)")


    start_time = time.time()
    truth['degrees'] = dict(G.degree())
    print(f"  Calculated {len(truth['degrees'])} node degrees (in {time.time() - start_time:.4f}s)")

    # Bipartiteness check (can be slow for large graphs)
    # start_time = time.time()
    # truth['is_bipartite'] = nx.is_bipartite(G)
    # print(f"  Bipartite check: {truth['is_bipartite']} (in {time.time() - start_time:.4f}s)")


    # Max weight matching is computationally expensive (NP-hard for general graphs)
    # We can compute an approximation using nx.max_weight_matching for comparison
    # Note: nx.max_weight_matching returns a set of edges (tuples)
    # start_time = time.time()
    # try:
    #      # Set maxcardinality=False for max weight (True finds max cardinality matching)
    #      # This can still be slow. Might skip for very large graphs.
    #      approx_mwm = nx.max_weight_matching(G, maxcardinality=False, weight='weight')
    #      truth['approx_mwm_weight'] = sum(G[u][v]['weight'] for u, v in approx_mwm)
    #      truth['approx_mwm_size'] = len(approx_mwm)
    #      print(f"  Approx Max Weight Matching (nx): Weight={truth['approx_mwm_weight']:.2f}, Size={truth['approx_mwm_size']} (in {time.time() - start_time:.4f}s)")
    # except Exception as e:
    #      print(f"  Could not compute NetworkX max_weight_matching: {e}")
    #      truth['approx_mwm_weight'] = -1
    #      truth['approx_mwm_size'] = -1


    print("Ground truth calculation complete.")
    return truth

def run_comparison(G, ground_truth):
    """Runs the implemented algorithms on graph G and compares results."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    results = []

    print(f"\n--- Running Comparisons on Graph (N={n}, M={m}) ---")

    # 1. GraphConnectivitySketch (L0 Sampling based)
    print("\nAlgorithm: GraphConnectivitySketch")
    algo_name = "ConnectivitySketch (L0)"
    start_build = time.time()
    conn_sketch = GraphConnectivitySketch(n)
    for u, v in G.edges():
        conn_sketch.add_edge(u, v)
    build_time = time.time() - start_build
    mem_estimate = conn_sketch.get_memory_estimate()

    start_query = time.time()
    # Run is_connected multiple times for average? No, single shot as per paper concept.
    # The result is probabilistic. Let's run it a few times to see stability.
    conn_results = [conn_sketch.is_connected() for _ in range(5)]
    query_time = (time.time() - start_query) / 5
    # Majority vote or report range? Let's take the most common result.
    try:
        from collections import Counter
        conn_result = Counter(conn_results).most_common(1)[0][0]
    except IndexError:
        conn_result = None # Handle case where list is empty (e.g., n=0)

    accuracy = (conn_result == ground_truth['is_connected']) if conn_result is not None else "N/A"
    print(f"  Build Time: {build_time:.4f}s")
    print(f"  Query Time (is_connected avg 5 runs): {query_time:.6f}s")
    print(f"  Memory Estimate: {mem_estimate / 1024:.2f} KB")
    print(f"  Is Connected (Sketch): {conn_result} (vs Ground Truth: {ground_truth['is_connected']}) -> Accuracy: {accuracy}")
    results.append({
        'Algorithm': algo_name, 'Build Time (s)': build_time, 'Query Time (s)': query_time,
        'Memory (KB)': mem_estimate / 1024, 'Query': 'is_connected',
        'Accuracy': 1 if accuracy == True else 0 if accuracy == False else 0.5, # Numeric accuracy
        'Notes': f"Probabilistic (Result: {conn_result})"
    })

    # 2. GraphLinearMeasurements
    print("\nAlgorithm: GraphLinearMeasurements")
    algo_name = "LinearMeasurements"
    # Parameter c controls m = c * n * log^2(n)
    c_param = 4 # As used in the class default
    start_build = time.time()
    linear_sketch = GraphLinearMeasurements(n, c=c_param)
    for u, v in G.edges():
        linear_sketch.update(u, v, delta=1) # Assuming unweighted edges for simplicity
    build_time = time.time() - start_build
    mem_estimate = linear_sketch.get_memory_estimate()
    print(f"  Build Time: {build_time:.4f}s")
    print(f"  Memory Estimate: {mem_estimate / 1024:.2f} KB (m={linear_sketch.m} measurements)")

    # --- Queries (Using Placeholders) ---
    print("  Running queries (using PLACEHOLDER decoders)...")
    query_results = {}
    query_times = {}

    start_q = time.time()
    query_results['is_connected'] = linear_sketch.is_connected() # Placeholder
    query_times['is_connected'] = time.time() - start_q

    start_q = time.time()
    query_results['approx_mst_weight'] = linear_sketch.approx_mst_weight() # Placeholder
    query_times['approx_mst_weight'] = time.time() - start_q

    start_q = time.time()
    query_results['is_bipartite'] = linear_sketch.is_bipartite() # Placeholder
    query_times['is_bipartite'] = time.time() - start_q

    print(f"  Query Time (is_connected): {query_times['is_connected']:.6f}s -> Result: {query_results['is_connected']} (Placeholder!)")
    print(f"  Query Time (approx_mst_w): {query_times['approx_mst_weight']:.6f}s -> Result: {query_results['approx_mst_weight']} (Placeholder!)")
    print(f"  Query Time (is_bipartite): {query_times['is_bipartite']:.6f}s -> Result: {query_results['is_bipartite']} (Placeholder!)")

    # Record results, noting they are based on placeholders
    results.append({
        'Algorithm': algo_name, 'Build Time (s)': build_time, 'Query Time (s)': query_times['is_connected'],
        'Memory (KB)': mem_estimate / 1024, 'Query': 'is_connected (placeholder)',
        'Accuracy': 0.5, 'Notes': f"Placeholder decoder (Result: {query_results['is_connected']})"
    })
    results.append({
        'Algorithm': algo_name, 'Build Time (s)': build_time, 'Query Time (s)': query_times['approx_mst_weight'],
        'Memory (KB)': mem_estimate / 1024, 'Query': 'approx_mst_w (placeholder)',
        'Accuracy': 0.5, 'Notes': f"Placeholder decoder (Result: {query_results['approx_mst_weight']})"
    })

    # 3. AdaptiveGraphSketch
    print("\nAlgorithm: AdaptiveGraphSketch")
    algo_name = "AdaptiveSketch (Sparsify)"
    # Parameter gamma trades off sparsifier size/accuracy and time
    gamma_param = 0.3 # Smaller gamma -> denser sparsifier -> slower but potentially more accurate
    start_build = time.time()
    adaptive_sketch = AdaptiveGraphSketch(n, gamma=gamma_param)
    for u, v, data in G.edges(data=True):
        adaptive_sketch.stream_update(u, v, data.get('weight', 1.0))
    build_time = time.time() - start_build # Time to ingest edges (doesn't include finalize)
    mem_estimate_before_finalize = adaptive_sketch.get_memory_estimate()

    # Finalize runs sparsification and computes queries (MST, Matching)
    start_finalize = time.time()
    # finalize() returns: mst_weight, matching_list, mst_time, matching_time
    final_results = adaptive_sketch.finalize()
    finalize_time = time.time() - start_finalize
    mem_estimate_after_finalize = adaptive_sketch.get_memory_estimate() # Memory after sparsification

    mst_weight_sparse = final_results[0]
    matching_sparse = final_results[1]
    mst_query_time = final_results[2] # Time for MST computation on sparsifier
    match_query_time = final_results[3] # Time for Matching computation on sparsifier

    mst_accuracy = 1.0 - abs(mst_weight_sparse - ground_truth['mst_weight']) / ground_truth['mst_weight'] if ground_truth['mst_weight'] > 0 else 1.0
    # Matching accuracy is harder to quantify without ground truth Max Weight Matching

    print(f"  Build Time (Edge Ingestion): {build_time:.4f}s")
    print(f"  Finalize Time (Sparsify + Queries): {finalize_time:.4f}s")
    print(f"    Query Time (MST on sparsifier): {mst_query_time:.4f}s")
    print(f"    Query Time (Matching on sparsifier): {match_query_time:.4f}s")
    print(f"  Memory Estimate (After Sparsify): {mem_estimate_after_finalize / 1024:.2f} KB")
    print(f"  MST Weight (Sparse): {mst_weight_sparse:.2f} (vs Ground Truth: {ground_truth['mst_weight']:.2f}) -> Accuracy: {mst_accuracy:.4f}")
    # print(f"  Approx Max Weight Matching (Sparse): Size={len(matching_sparse)}, Weight={sum(w for _,_,w in matching_sparse):.2f}")

    results.append({
        'Algorithm': algo_name, 'Build Time (s)': build_time + finalize_time, # Total time
        'Query Time (s)': mst_query_time, # Just MST query time after finalize
        'Memory (KB)': mem_estimate_after_finalize / 1024, 'Query': 'MST Weight (Exact on Sparse)',
        'Accuracy': mst_accuracy, 'Notes': f"Gamma={gamma_param}, Sparsifier Edges={adaptive_sketch.edge_count}"
    })
    results.append({
        'Algorithm': algo_name, 'Build Time (s)': build_time + finalize_time,
        'Query Time (s)': match_query_time, # Matching query time
        'Memory (KB)': mem_estimate_after_finalize / 1024, 'Query': 'Approx Max Weight Matching',
        'Accuracy': 0.5, 'Notes': "Accuracy difficult to measure here"
    })

    # 4. ModifiedBoruvkaSketching (Fallback Implementation)
    print("\nAlgorithm: ModifiedBoruvkaSketching (Fallback)")
    algo_name = "Boruvka (Fallback Impl)"
    start_build = time.time()
    boruvka_sketch = ModifiedBoruvkaSketching(n)
    for u, v, data in G.edges(data=True):
        boruvka_sketch.add_edge(u, v, data.get('weight', 1.0))
    build_time = time.time() - start_build # Time to build edge heap
    mem_estimate = boruvka_sketch.get_memory_estimate() # Memory for edge heap

    start_query = time.time()
    mst_edges_boruvka, mst_weight_boruvka = boruvka_sketch.find_spanning_tree()
    query_time = time.time() - start_query # Time for MST computation

    mst_accuracy = 1.0 - abs(mst_weight_boruvka - ground_truth['mst_weight']) / ground_truth['mst_weight'] if ground_truth['mst_weight'] > 0 else 1.0

    print(f"  Build Time (Add Edges): {build_time:.4f}s")
    print(f"  Query Time (find_spanning_tree): {query_time:.4f}s")
    print(f"  Memory Estimate (Edge List/Heap): {mem_estimate / 1024:.2f} KB")
    print(f"  MST Weight (Boruvka): {mst_weight_boruvka:.2f} (vs Ground Truth: {ground_truth['mst_weight']:.2f}) -> Accuracy: {mst_accuracy:.4f}")
    results.append({
        'Algorithm': algo_name, 'Build Time (s)': build_time,
        'Query Time (s)': query_time,
        'Memory (KB)': mem_estimate / 1024, 'Query': 'MST Weight (Exact)',
        'Accuracy': mst_accuracy, 'Notes': f"Fallback impl (not sketched)"
    })


    # 5. gSketch (for Degree Estimation)
    print("\nAlgorithm: gSketch")
    algo_name = "gSketch (Degree Est.)"
    # Parameters
    num_partitions = 8 # More partitions -> more localized, maybe less contention
    cms_width = 128    # Width of internal CMS
    cms_depth = 5      # Depth of internal CMS
    start_build = time.time()
    # Provide node_ids upfront if known, helps partitioning consistency
    gsketch_instance = gSketch(num_partitions, cms_width, cms_depth, node_ids=list(G.nodes()))
    for u, v in G.edges():
        gsketch_instance.add_edge(u, v, weight=1) # Use weight=1 for degree counting
    build_time = time.time() - start_build
    mem_estimate = gsketch_instance.get_memory_estimate()

    # Query: Estimate degrees for a sample of nodes
    query_time_total = 0
    degree_errors = []
    nodes_to_query = random.sample(list(G.nodes()), min(100, n)) # Query degree for 100 nodes or all if < 100
    estimated_degrees = {}

    start_query_all = time.time()
    for node in nodes_to_query:
        start_q_node = time.time()
        est_degree = gsketch_instance.estimate_degree(node)
        query_time_total += (time.time() - start_q_node)
        estimated_degrees[node] = est_degree
        true_degree = ground_truth['degrees'].get(node, 0)
        degree_errors.append(abs(est_degree - true_degree))
    query_time_avg = query_time_total / len(nodes_to_query) if nodes_to_query else 0

    avg_abs_error = np.mean(degree_errors) if degree_errors else 0
    # Relative error is tricky if true degree is 0. Use Mean Absolute Percentage Error (MAPE) carefully or just abs error.
    # Accuracy score: 1 / (1 + avg_abs_error) -> ranges (0, 1], 1 is perfect.
    accuracy_score = 1.0 / (1.0 + avg_abs_error)

    print(f"  Build Time: {build_time:.4f}s")
    print(f"  Query Time (Estimate Degree avg over {len(nodes_to_query)} nodes): {query_time_avg:.8f}s")
    print(f"  Memory Estimate: {mem_estimate / 1024:.2f} KB")
    print(f"  Avg Absolute Degree Error: {avg_abs_error:.4f}")
    print(f"  Degree Accuracy Score: {accuracy_score:.4f}")

    results.append({
        'Algorithm': algo_name, 'Build Time (s)': build_time,
        'Query Time (s)': query_time_avg,
        'Memory (KB)': mem_estimate / 1024, 'Query': 'Estimate Degree',
        'Accuracy': accuracy_score, 'Notes': f"AvgAbsErr={avg_abs_error:.2f}, Parts={num_partitions}, W={cms_width}, D={cms_depth}"
    })
    # Store estimated degrees for potential plotting
    results[-1]['details'] = {'estimated_degrees': estimated_degrees, 'nodes_queried': nodes_to_query}


    # 6. GraphVisualSketchHierarchy (Qualitative / Structural)
    print("\nAlgorithm: GraphVisualSketchHierarchy")
    algo_name = "VisualSketchHierarchy"
    start_build = time.time()
    # Convert nx graph to simple dict adjacency list
    adj_dict = {node: set(neighbors) for node, neighbors in G.adjacency()}
    hierarchy = GraphVisualSketchHierarchy(adj_dict)
    # Build a few levels
    hierarchy.build_hierarchy(max_levels=4, max_nodes_per_supernode=max(10, n // 20)) # Limit supernode size
    build_time = time.time() - start_build
    mem_estimate = hierarchy.get_memory_estimate()

    # Query time is not typical; it's about retrieving level views
    start_query = time.time()
    level1_view = hierarchy.get_level_view(1)
    query_time = time.time() - start_query # Time to get one level view

    num_nodes_l0 = len(hierarchy.get_level_view(0))
    num_nodes_l1 = len(level1_view) if level1_view else 0
    reduction_factor = num_nodes_l1 / num_nodes_l0 if num_nodes_l0 > 0 else 0

    print(f"  Build Time (Hierarchy): {build_time:.4f}s")
    print(f"  Query Time (Get Level 1 View): {query_time:.6f}s")
    print(f"  Memory Estimate: {mem_estimate / 1024:.2f} KB")
    print(f"  Level 0 Nodes: {num_nodes_l0}, Level 1 Nodes: {num_nodes_l1} (Reduction: {reduction_factor:.2f})")

    results.append({
        'Algorithm': algo_name, 'Build Time (s)': build_time,
        'Query Time (s)': query_time,
        'Memory (KB)': mem_estimate / 1024, 'Query': 'Build Hierarchy / Get View',
        'Accuracy': reduction_factor, # Use reduction factor as a proxy metric
        'Notes': f"Built {hierarchy.max_level+1} levels. L1 Nodes={num_nodes_l1}"
    })
    # Store hierarchy object for potential visualization later
    results[-1]['details'] = {'hierarchy_object': hierarchy}


    print("\n--- Comparison Run Complete ---")
    return results, ground_truth # Return results and GT for plotting

# --- Visualization Functions ---

def plot_performance(results_df):
    """Plots Build Time, Query Time, and Memory Usage."""
    print("\n--- Generating Performance Plots ---")
    if results_df.empty:
        print("No results to plot.")
        return

    # Prepare data: Melt for easier plotting with seaborn
    perf_df = results_df.melt(id_vars=['Algorithm', 'Query', 'Notes'],
                              value_vars=['Build Time (s)', 'Query Time (s)', 'Memory (KB)'],
                              var_name='Metric', value_name='Value')

    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Build Time (Log scale often useful for time)
    plt.figure(figsize=(12, 6))
    build_data = perf_df[perf_df['Metric'] == 'Build Time (s)']
    # Use unique build times per algorithm (some algos listed multiple times for different queries)
    build_data_unique = build_data.drop_duplicates(subset=['Algorithm'])
    sns.barplot(data=build_data_unique, x='Algorithm', y='Value', palette='viridis')
    plt.title('Algorithm Build Time')
    plt.ylabel('Time (s) - Log Scale')
    plt.yscale('log') # Use log scale if times vary widely
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("plot_build_time.png")
    # plt.show()
    plt.close()


    # Plot Query Time (Log scale often useful) - Separate plot per query type might be clearer
    plt.figure(figsize=(14, 7))
    query_data = perf_df[perf_df['Metric'] == 'Query Time (s)']
    sns.barplot(data=query_data, x='Algorithm', y='Value', hue='Query', palette='magma', dodge=True)
    plt.title('Algorithm Query Time (Lower is Better)')
    plt.ylabel('Time (s) - Log Scale')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Query Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plot_query_time.png")
    # plt.show()
    plt.close()

    # Plot Memory Usage
    plt.figure(figsize=(12, 6))
    mem_data = perf_df[perf_df['Metric'] == 'Memory (KB)']
    # Use unique memory usage per algorithm instance
    mem_data_unique = mem_data.drop_duplicates(subset=['Algorithm'])
    sns.barplot(data=mem_data_unique, x='Algorithm', y='Value', palette='plasma')
    plt.title('Estimated Memory Usage')
    plt.ylabel('Memory (KB) - Log Scale')
    plt.yscale('log') # Use log scale if memory varies widely
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("plot_memory_usage.png")
    # plt.show()
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(14, 7))
    # Filter out placeholder results where accuracy is meaningless (0.5)
    acc_data = results_df[results_df['Accuracy'] != 0.5]
    if not acc_data.empty:
        sns.barplot(data=acc_data, x='Algorithm', y='Accuracy', hue='Query', palette='coolwarm', dodge=True)
        plt.title('Algorithm Accuracy (Higher is Better, Placeholders Excluded)')
        plt.ylabel('Accuracy Score (0 to 1)')
        plt.ylim(0, 1.1) # Set Y-axis limits
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Query Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("plot_accuracy.png")
        # plt.show()
    else:
        print("No non-placeholder accuracy results to plot.")
    plt.close()


    print("Performance plots generated and saved.")


def plot_degree_estimation(results_df, ground_truth):
    """Plots gSketch degree estimation accuracy."""
    print("\n--- Generating Degree Estimation Plot (gSketch) ---")
    gsketch_result = results_df[(results_df['Algorithm'] == 'gSketch (Degree Est.)') & ('details' in results_df.columns)].iloc[0]

    if not gsketch_result.empty and 'details' in gsketch_result and gsketch_result['details']:
        details = gsketch_result['details']
        estimated_degrees = details.get('estimated_degrees', {})
        nodes_queried = details.get('nodes_queried', [])

        if not estimated_degrees or not nodes_queried:
            print("No degree estimation data found for gSketch.")
            return

        true_degrees = [ground_truth['degrees'].get(node, 0) for node in nodes_queried]
        est_degrees_list = [estimated_degrees.get(node, 0) for node in nodes_queried]

        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=true_degrees, y=est_degrees_list, alpha=0.7)
        # Add y=x line for reference
        max_degree = max(max(true_degrees), max(est_degrees_list))
        plt.plot([0, max_degree], [0, max_degree], color='red', linestyle='--', label='Perfect Estimation (y=x)')
        plt.title('gSketch Degree Estimation Accuracy')
        plt.xlabel('True Degree')
        plt.ylabel('Estimated Degree (CMS)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plot_gsketch_degree_accuracy.png")
        # plt.show()
        plt.close()
        print("gSketch degree accuracy plot saved.")
    else:
        print("Could not find detailed gSketch results for plotting.")


def plot_graph_visualizations(G, results_df):
    """Visualizes original graph vs. sparsified/hierarchy."""
    print("\n--- Generating Graph Structure Visualizations ---")
    plt.figure(figsize=(18, 6))

    # 1. Original Graph (Sampled nodes/edges if too large)
    plt.subplot(1, 3, 1)
    pos = nx.spring_layout(G, seed=42, k=0.1) # Layout can be slow
    node_sample = random.sample(list(G.nodes()), min(100, G.number_of_nodes()))
    edge_sample = random.sample(list(G.edges()), min(200, G.number_of_edges()))
    G_sample = G.edge_subgraph(edge_sample).copy() # Create subgraph from sampled edges
    nodes_to_draw = list(G_sample.nodes()) # Draw nodes present in the edge sample
    nx.draw_networkx_nodes(G_sample, pos, nodelist=nodes_to_draw, node_size=10, alpha=0.6)
    nx.draw_networkx_edges(G_sample, pos, width=0.5, alpha=0.3)
    plt.title(f"Original Graph (Sample: {len(nodes_to_draw)}N, {len(edge_sample)}E)")
    plt.axis('off')

    # 2. Adaptive Sketch Sparsifier
    plt.subplot(1, 3, 2)
    adaptive_result = results_df[(results_df['Algorithm'] == 'AdaptiveSketch (Sparsify)') & (results_df['Query'] == 'MST Weight (Exact on Sparse)')]
    if not adaptive_result.empty:
         # Need the actual sparsified graph object. This requires modifying the run_comparison
         # or AdaptiveGraphSketch class to return/store the final graph.
         # Let's assume we can reconstruct it from the _get_edges_from_sparsifier logic (hacky)
         # This is inefficient and requires re-running part of the logic or storing the graph.
         # For demo, we'll just show fewer edges conceptually.
         # Proper way: Modify AdaptiveGraphSketch to store/return self.current_graph after finalize.
         # Hacky Demo: Draw original graph pos, but with fewer edges shown
         num_sparse_edges = int(adaptive_result['Notes'].iloc[0].split("=")[-1]) if 'Edges=' in adaptive_result['Notes'].iloc[0] else len(edge_sample)//2
         sparse_edge_sample = random.sample(edge_sample, min(len(edge_sample), num_sparse_edges))
         G_sparse_sample = G.edge_subgraph(sparse_edge_sample).copy()
         nodes_to_draw_sparse = list(G_sparse_sample.nodes())
         nx.draw_networkx_nodes(G_sparse_sample, pos, nodelist=nodes_to_draw_sparse, node_size=10, alpha=0.6, node_color='orange')
         nx.draw_networkx_edges(G_sparse_sample, pos, width=0.5, alpha=0.5, edge_color='orange')
         plt.title(f"Sparsifier (Conceptual, {num_sparse_edges} edges)")
         plt.axis('off')
    else:
         plt.title("Sparsifier (N/A)")
         plt.axis('off')


    # 3. Visual Hierarchy Level 1
    plt.subplot(1, 3, 3)
    hierarchy_result = results_df[(results_df['Algorithm'] == 'VisualSketchHierarchy') & ('details' in results_df.columns)]
    if not hierarchy_result.empty and 'details' in hierarchy_result.iloc[0] and hierarchy_result.iloc[0]['details']:
        hierarchy = hierarchy_result.iloc[0]['details'].get('hierarchy_object')
        if hierarchy:
            level1_view = hierarchy.get_level_view(1)
            if level1_view:
                 # Create a NetworkX graph from the level 1 view (supernodes)
                 G_level1 = nx.Graph()
                 for snode, neighbors in level1_view.items():
                     G_level1.add_node(snode)
                     for neighbor_snode in neighbors:
                         G_level1.add_edge(snode, neighbor_snode)

                 pos_l1 = nx.spring_layout(G_level1, seed=42)
                 nx.draw_networkx_nodes(G_level1, pos_l1, node_size=30, node_color='green', alpha=0.8)
                 nx.draw_networkx_edges(G_level1, pos_l1, width=1.0, alpha=0.6, edge_color='green')
                 plt.title(f"Visual Hierarchy Level 1 ({G_level1.number_of_nodes()} Supernodes)")
                 plt.axis('off')
            else:
                 plt.title("Visual Hierarchy L1 (Empty)")
                 plt.axis('off')
        else:
            plt.title("Visual Hierarchy L1 (N/A)")
            plt.axis('off')
    else:
        plt.title("Visual Hierarchy L1 (N/A)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("plot_graph_visualizations.png")
    # plt.show()
    plt.close()
    print("Graph structure visualizations saved.")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    NUM_NODES = 100      # Size of the graph (nodes)
    TARGET_EDGES = 1000  # Approximate number of edges
    GRAPH_MODEL = 'ba'   # 'ba' or 'gnp'
    RANDOM_SEED = 42     # For reproducibility

    # Set seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Generate Graph ---
    G = generate_graph(NUM_NODES, TARGET_EDGES, model=GRAPH_MODEL, seed=RANDOM_SEED)

    # --- Calculate Ground Truth ---
    ground_truth = get_ground_truth(G)

    # --- Run Algorithm Comparison ---
    comparison_results, _ = run_comparison(G, ground_truth)

    # --- Print Summary Table ---
    results_df = pd.DataFrame(comparison_results)
    # Remove complex objects before printing
    results_df_printable = results_df.drop(columns=['details'], errors='ignore')
    print("\n\n--- Performance Summary ---")
    print(results_df_printable.to_string())

    # --- Generate Plots ---
    if not results_df.empty:
         # Check if matplotlib is available
         try:
             import matplotlib.pyplot as plt
             import seaborn as sns
             plot_performance(results_df)
             plot_degree_estimation(results_df, ground_truth) # Specific plot for gSketch
             plot_graph_visualizations(G, results_df) # Structural plots
         except ImportError:
             print("\n[WARN] Matplotlib or Seaborn not found. Skipping plot generation.")
             print("Please install them: pip install matplotlib seaborn")
    else:
         print("\nNo results data frame generated. Skipping plots.")


    # --- Textual Comparison & Contrast ---
    print("\n\n--- Comparison and Contrast Summary ---")
    print(f"Graph: N={G.number_of_nodes()}, M={G.number_of_edges()}, Connected={ground_truth.get('is_connected', 'N/A')}, MST Weight={ground_truth.get('mst_weight', 0):.2f}")

    # Find best performers for different metrics (example)
    if not results_df.empty:
        # Fastest build time (use unique algorithm build times)
        build_times = results_df.drop_duplicates(subset=['Algorithm'])[['Algorithm', 'Build Time (s)']]
        fastest_build = build_times.loc[build_times['Build Time (s)'].idxmin()]
        print(f"\nFastest Build Time: {fastest_build['Algorithm']} ({fastest_build['Build Time (s)']:.4f}s)")

        # Lowest memory usage (use unique algorithm memory)
        mem_usage = results_df.drop_duplicates(subset=['Algorithm'])[['Algorithm', 'Memory (KB)']]
        lowest_mem = mem_usage.loc[mem_usage['Memory (KB)'].idxmin()]
        print(f"Lowest Memory Usage: {lowest_mem['Algorithm']} ({lowest_mem['Memory (KB)']:.2f} KB)")

        # Fastest query time for specific queries (e.g., Connectivity)
        conn_query_times = results_df[results_df['Query'].str.contains("is_connected", case=False, na=False)]
        if not conn_query_times.empty:
             fastest_conn_query = conn_query_times.loc[conn_query_times['Query Time (s)'].idxmin()]
             print(f"Fastest Connectivity Query: {fastest_conn_query['Algorithm']} ({fastest_conn_query['Query Time (s)']:.6f}s) - Note: May include placeholders/probabilistic.")

        # Highest accuracy for specific queries (e.g., MST)
        mst_accuracy = results_df[results_df['Query'].str.contains("MST", case=False, na=False) & (results_df['Accuracy'] != 0.5)] # Exclude placeholders
        if not mst_accuracy.empty:
             best_mst_acc = mst_accuracy.loc[mst_accuracy['Accuracy'].idxmax()]
             print(f"Highest MST Accuracy: {best_mst_acc['Algorithm']} ({best_mst_acc['Accuracy']:.4f})")

    print("\nKey Observations & Trade-offs:")
    print("- L0-Sampling (ConnectivitySketch): Very fast updates and queries, low memory, but probabilistic connectivity results.")
    print("- LinearMeasurements: Fast updates, potentially higher memory (O(n log^2 n)). Query performance/accuracy depends heavily on the *decoding algorithms* (which were placeholders here). Theoretical power for various queries.")
    print("- AdaptiveSketch (Sparsification): Slower 'build' (ingestion+sparsification), but computes exact MST/approx Matching on the *smaller* sparsified graph. Accuracy depends on gamma and sparsifier quality. Memory usage drops after sparsification.")
    print("- Boruvka (Fallback): Used as a baseline here. Its performance is standard MST computation O(M log N or M log* N), not sketch-based. Memory stores all edges.")
    print("- gSketch: Partitioned approach. Fast updates. Very fast degree estimation queries. Accuracy depends on CMS parameters (width, depth) and partitioning. Good for localized queries.")
    print("- SimpleSetSketch: Not directly compared on graph-wide tasks here, but demonstrated low-level set operations (XOR properties). Very space-efficient for small sets. Useful building block.")
    print("- VisualHierarchy: Primarily for structural abstraction, not typical query performance. Build time depends on aggregation complexity. Memory stores multiple graph levels.")

    print("\nChoosing an Algorithm:")
    print("- For **dynamic connectivity** on streams with low memory: L0-Sampling (ConnectivitySketch) is strong, accepting probabilistic results.")
    print("- For **static graph MST/Matching** where pre-processing is acceptable: AdaptiveSketch provides accuracy control via gamma, working on a smaller graph for queries.")
    print("- For **fast degree estimation** or other localized queries on streams: gSketch offers good performance through partitioning.")
    print("- For **diverse queries** on streams (if decoders implemented): LinearMeasurements offer theoretical breadth but require complex decoding.")
    print("- For **visual exploration** and multi-level analysis: VisualHierarchy.")

    print("\nNote: Performance depends significantly on graph size, structure, implementation details, and chosen parameters.")