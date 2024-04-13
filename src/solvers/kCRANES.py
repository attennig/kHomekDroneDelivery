import networkx as nx
import random
import numpy

numpy.random.seed(0)
random.seed(0)


class kCranesSolver:
    def __init__(self, V, initial_vertex, E, A, k):
        self.G = nx.Graph()
        self.G.add_nodes_from(V)
        self.G.add_edges_from(E)
        self.V = V
        self.initial_vertex = initial_vertex
        self.A = A
        self.k = k
        self.heads = [h for (t, h, _) in self.A]
        self.tails = [t for (t, h, _) in self.A]
        self.tail_name = 'src'
        self.head_name = 'dst'

    def get_arc_id(self, e):
        return int(e[0][len(self.tail_name):])

    def get_arc(self, node):
        for t, h, w in self.A:
            if node == t or node == h:
                return (t, h, w)
        return None

    def edge_cost(self, edge):
        if len(edge) == 3:
            return edge[2]['weight']
        return self.G[edge[0]][edge[1]]['weight']

    def preprocess(self):
        self.G.add_node(f'{self.initial_vertex}2')
        self.A += [(f'{self.initial_vertex}2', self.initial_vertex, {'weight': 0})]
        self.tails.append(f'{self.initial_vertex}2')
        self.heads.append(self.initial_vertex)
        edges = [(f'{self.initial_vertex}2', self.initial_vertex, {'weight': 0})] \
                + [(f'{self.initial_vertex}2', v, {'weight': self.G[self.initial_vertex][v]['weight']}) for v in
                   self.G.nodes if v not in [self.initial_vertex, f'{self.initial_vertex}2']]
        self.G.add_edges_from(edges)

    def constructTour(self, cycle, G):
        for i in range(len(cycle)):
            e = cycle[i]
            if e[0] == self.initial_vertex:
                break
        cycle = cycle[i:] + cycle[:i]
        # shortcutting
        tour, makespan = [], 0
        i = 0
        while i < len(cycle):
            if cycle[i] in self.A:
                (u, v, w) = cycle[i]
                if (u, v) == (f'{self.initial_vertex}2', self.initial_vertex):
                    if i+1<len(cycle):
                        tour[-1] = (tour[-1][0], cycle[i + 1][1])
                        i += 1
                    i += 1
                    continue
                tour += [(u, v, w)]
                makespan += w['weight']
                i += 1
            else:
                init_i = i
                u = cycle[i][0]
                while i < len(cycle) and len(cycle[i]) != 3:
                    i += 1
                if i >= len(cycle):
                    v = self.initial_vertex
                else:
                    v = cycle[i - 1][1]
                    if v == f'{self.initial_vertex}2':
                        if u == self.initial_vertex: continue
                        v = self.initial_vertex
                tour += [(u, v)]
                makespan += G[u][v]['weight']
        assert tour[0][0] == self.initial_vertex
        return tour, makespan

    def oriented_disjoint_cycles(self, min_matching):
        # orient matching
        oriented_matching = []
        for (u, v) in min_matching:
            if u in self.tails:
                assert v in self.heads
                oriented_matching.append((v, u))
            else:
                assert u in self.heads and v in self.tails
                oriented_matching.append((u, v))
        # extract oriented cycles
        cycles_graph = nx.Graph()
        cycles = {}
        nodes_count = 0
        arcs_for_cycles = self.A.copy()
        if oriented_matching:
            oriented_matching = list(dict.fromkeys(oriented_matching).keys())
        while oriented_matching:
            e = oriented_matching.pop()
            (u, v) = e[:2]
            cycle = [e]
            closure_node = u
            while True:
                for a in arcs_for_cycles:
                    src, dst = a[:2]
                    if src == v:
                        break
                arcs_for_cycles.remove(a)
                cycle += [a]
                if dst == closure_node:
                    cycles[f"n{nodes_count}"] = cycle
                    cycles_graph.add_node(f"n{nodes_count}")
                    nodes_count += 1
                    break
                u = dst
                v = [j for (i, j) in oriented_matching if i == u][0]
                oriented_matching.remove((u, v))
                cycle += [(u, v)]
        return cycles, cycles_graph

    def add_edges_to_cycles_graph(self, cycles, cycles_graph, G):
        for i_idx in range(cycles_graph.number_of_nodes()):
            for j_idx in range(i_idx + 1, cycles_graph.number_of_nodes()):
                internode_distance, min_u, min_v = float('inf'), None, None
                for e_u in cycles[f"n{i_idx}"]:
                    (u1, u2) = e_u[:2]
                    for e_v in cycles[f"n{j_idx}"]:
                        if len(e_v) == 3:
                            (v1, v2, _) = e_v
                        else:
                            (v1, v2) = e_v
                        if G[u1][v1]['weight'] < internode_distance:
                            internode_distance, min_u, min_v = G[u1][v1]['weight'], u1, v1
                        if G[u2][v2]['weight'] < internode_distance:
                            internode_distance, min_u, min_v = G[u2][v2]['weight'], u2, v2
                        if G[u1][v2]['weight'] < internode_distance:
                            internode_distance, min_u, min_v = G[u1][v2]['weight'], u1, v2
                        if G[u2][v1]['weight'] < internode_distance:
                            internode_distance, min_u, min_v = G[u2][v1]['weight'], u2, v1
                cycles_graph.add_edge(f"n{i_idx}", f"n{j_idx}", weight=internode_distance, min_edge=(min_u, min_v))
        return cycles_graph

    def merge_cycles(self, cycles, cycles_graph, min_spanning_tree):
        while min_spanning_tree.edges:
            (ni, nj) = list(min_spanning_tree.edges)[0]
            min_spanning_tree.remove_edge(ni, nj)

            (u, v) = cycles_graph[ni][nj]['min_edge']
            for k in range(len(cycles[ni])):
                (src_k, dst_k) = cycles[ni][k][:2]
                if src_k == u or src_k == v:
                    break

            for l in range(len(cycles[nj])):
                (src_l, dst_l) = cycles[nj][l][:2]
                if src_l == u or src_l == v:
                    break

            cycles[ni] = cycles[ni][k:] + cycles[ni][:k]
            cycles[nj] = cycles[nj][l:] + cycles[nj][:l]

            if u != cycles[ni][-1][1]:
                u, v = v, u
            c = cycles[ni] + [(u, v)] + cycles[nj] + [(v, u)]
            i, j = ni[1:], nj[1:]
            cycles[f"n({i},{j})"] = c
            cycles_graph.add_node(f"n({i},{j})")
            min_spanning_tree.add_node(f"n({i},{j})")

            for child, edge_attr in min_spanning_tree[ni].items():
                min_spanning_tree.add_edge(f"n({i},{j})", child, weight=edge_attr['weight'],
                                           min_edge=edge_attr['min_edge'])
                cycles_graph.add_edge(f"n({i},{j})", child, weight=edge_attr['weight'], min_edge=edge_attr['min_edge'])

            for child, edge_attr in min_spanning_tree[nj].items():
                min_spanning_tree.add_edge(f"n({i},{j})", child, weight=edge_attr['weight'],
                                           min_edge=edge_attr['min_edge'])
                cycles_graph.add_edge(f"n({i},{j})", child, weight=edge_attr['weight'], min_edge=edge_attr['min_edge'])

            min_spanning_tree.remove_node(ni)
            min_spanning_tree.remove_node(nj)

            cycles_graph.remove_node(ni)
            cycles_graph.remove_node(nj)

        cycle_name = [node for node in cycles_graph.nodes][0]
        return cycles[cycle_name]

    def LARGEARC(self, G):

        max_w = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[0][2]['weight']
        # implement 1.
        for (t1, h1, w1) in self.A:
            for (t2, h2, w2) in self.A:
                if t1 != t2:
                    G[t1][t2]['weight'] = max_w + 1
                    G[h1][h2]['weight'] = max_w + 1
        # implement 2.
        min_matching = nx.min_weight_matching(G)
        min_matching = [(v,u) if v > u else (u,v) for (u,v) in min_matching]
        min_matching = sorted(min_matching, key=lambda x: x[0])
        # implement 3.
        cycles, cycles_graph = self.oriented_disjoint_cycles(min_matching)
        # implement 4.
        cycles_graph = self.add_edges_to_cycles_graph(cycles, cycles_graph, G)
        # implement 5.
        min_spanning_tree = nx.minimum_spanning_tree(cycles_graph)
        # implement 6.
        single_cycle = self.merge_cycles(cycles, cycles_graph, min_spanning_tree)
        return self.constructTour(single_cycle, G)

    def initialize_collapsed_nodes_graph(self, G):
        collapsed_nodes_graph = nx.Graph()
        related_arc = {}

        for i, e in enumerate(self.A):
            (t, h, w) = e
            if self.initial_vertex in (t, h):
                related_arc[self.initial_vertex] = (t, h, w)
                collapsed_nodes_graph.add_node(self.initial_vertex)
            else:
                related_arc[f"n{self.get_arc_id(e)}"] = (t, h, w)
                collapsed_nodes_graph.add_node(f"n{self.get_arc_id(e)}")

        for ni in collapsed_nodes_graph.nodes:
            for nj in collapsed_nodes_graph.nodes:
                if nj == ni: continue
                if collapsed_nodes_graph.has_edge(ni, nj): continue
                internode_distance, min_u, min_v = float('inf'), None, None
                (t_i, h_i, w_i) = related_arc[ni]
                (t_j, h_j, w_j) = related_arc[nj]
                if G[t_i][t_j]['weight'] < internode_distance:
                    internode_distance, min_u, min_v = G[t_i][t_j]['weight'], t_i, t_j
                if G[t_i][h_j]['weight'] < internode_distance:
                    internode_distance, min_u, min_v = G[t_i][h_j]['weight'], t_i, h_j
                if G[h_i][t_j]['weight'] < internode_distance:
                    internode_distance, min_u, min_v = G[h_i][t_j]['weight'], h_i, t_j
                if G[h_i][h_j]['weight'] < internode_distance:
                    internode_distance, min_u, min_v = G[h_i][h_j]['weight'], h_i, h_j
                collapsed_nodes_graph.add_edge(ni, nj, weight=internode_distance, min_edge=(min_u, min_v))
        return collapsed_nodes_graph, related_arc

    def compute_odd_degree_graph(self, min_spanning_tree, collapsed_nodes_graph):
        odd_degree_graph = nx.Graph()
        for node in min_spanning_tree:
            if len(min_spanning_tree[node]) % 2 == 1:
                odd_degree_graph.add_node(node)

        for ni in odd_degree_graph.nodes:
            for nj in odd_degree_graph.nodes:
                if ni == nj: continue
                odd_degree_graph.add_edge(ni, nj, weight=collapsed_nodes_graph[ni][nj]['weight'],
                                          min_edge=collapsed_nodes_graph[ni][nj]['min_edge'])
        return odd_degree_graph

    def get_arcs_spanning_matching_multigraph(self, spanning_edges_renamed, matching_edges_renamed, G):
        out_deg = {u: 0 for u in G.nodes}
        adjacent = {u: [] for u in G.nodes}
        for (u, v) in spanning_edges_renamed + matching_edges_renamed:
            adjacent[u].append(v)
            out_deg[u] += 1
            adjacent[v].append(u)
            out_deg[v] += 1

        even_arcs_cost = 0
        even_arcs = []
        odd_arcs = []
        for e in self.A:
            (t, h, w) = e
            adjacent[t].append(h)
            out_deg[t] += 1
            adjacent[h].append(t)
            out_deg[h] += 1
            if out_deg[t] % 2 == 1 and out_deg[h] % 2 == 1:
                odd_arcs.append({t, h})
            else:
                even_arcs.append({t, h})
                even_arcs_cost += w['weight']
        return adjacent, out_deg, odd_arcs, even_arcs, even_arcs_cost

    def findCircuit(self, adj: dict, out_deg: dict, odd_arcs: list):
        # adj represents the adjacency list of the directed graph
        # edge_count represents the number of edges emerging from a vertex
        if len(adj) == 0:
            return  # empty graph
        # Maintain a stack to keep vertices
        curr_path = []
        # vector to store final circuit
        circuit = []
        # start from initial vertex
        u = self.initial_vertex  # Current vertex
        last_node_in_circuit = None
        curr_path.append(u)
        while len(curr_path):
            # If there is an outgoing edge from the current vertex
            if out_deg[u]:
                # Push the vertex
                curr_path.append(u)
                # Find the next vertex using an edge
                v = adj[u][-1]
                # and remove that edge
                out_deg[u] -= 1
                adj[u].pop()
                # if it is not and odd arc, the edge in the other direction is also removed
                if {u, v} not in odd_arcs:
                    out_deg[v] -= 1
                    adj[v].remove(u)
                # Move to next vertex
                u = v
            # back-track to find remaining circuit
            else:
                if last_node_in_circuit:
                    circuit.append((last_node_in_circuit, u))
                last_node_in_circuit = u
                # Back-tracking
                u = curr_path[-1]
                curr_path.pop()
        return circuit

    def insert_arcs_in_circuit(self, circuit: list, odd_arcs: list, even_arcs: list, even_arcs_cost: list):
        # circuit list of tuples representing directed edges
        # odd_acrs list of sets representing undirected edges but the orientation can be determined
        # even_arcs list of sets representing undirected edges but the orientation can be determined
        # even_arcs_cost float cost of even arcs
        backward_arcs_cost = 0
        for u, v in circuit:
            if {u, v} in even_arcs:
                t, h, w = self.get_arc(u)
                if v == t:  # u == h
                    # arc is traversed backward
                    backward_arcs_cost += w['weight']
        if backward_arcs_cost > even_arcs_cost / 2:
            circuit = list(reversed([(v, u) for (u, v) in circuit]))
        visit = []
        for u, v in circuit:
            if {u, v} in even_arcs:
                t, h, w = self.get_arc(u)
                if u == t:  # v == h
                    # arc is traversed forward
                    visit.append((t, h, w))
                else:
                    # arc is traversed backward
                    assert u == h and v == t
                    visit.append((h, t))
                    visit.append((t, h, w))
                    visit.append((h, t))
            elif {u, v} in odd_arcs:
                t, h, w = self.get_arc(u)
                if u == t:  # v == h
                    # arc is traversed forward
                    visit.append((t, h, w))
                else:
                    # the corresponding backward edge is traversed
                    assert u == h and v == t
                    visit.append((h, t))
            else:
                visit.append((u, v))
        return visit

    def LARGEEDGE(self, G):
        # implement 1.
        collapsed_nodes_graph, related_arc = self.initialize_collapsed_nodes_graph(G)
        # implement 2.
        min_spanning_tree = nx.minimum_spanning_tree(collapsed_nodes_graph)
        # implement 3.
        odd_degree_graph = self.compute_odd_degree_graph(min_spanning_tree, collapsed_nodes_graph)
        min_weight_matching = nx.min_weight_matching(odd_degree_graph)
        # implement 4.
        spanning_edges = nx.get_edge_attributes(min_spanning_tree, 'min_edge')
        spanning_edges_renamed = [(u, v) for (ni, nj), (u, v) in spanning_edges.items()]
        matching_edges = nx.get_edge_attributes(collapsed_nodes_graph.edge_subgraph(min_weight_matching), 'min_edge')
        matching_edges_renamed = [(u, v) for (ni, nj), (u, v) in matching_edges.items()]
        adjacent, out_deg, odd_arcs, even_arcs, even_arcs_cost = self.get_arcs_spanning_matching_multigraph(
            spanning_edges_renamed, matching_edges_renamed, G)
        # implement 5.
        circuit = self.findCircuit(adjacent, out_deg, odd_arcs)
        # implement 6.
        visit = self.insert_arcs_in_circuit(circuit, odd_arcs, even_arcs, even_arcs_cost)
        return self.constructTour(visit, G)

    def CRANE(self):
        self.preprocess()
        tourA, makespanA = self.LARGEARC(self.G.copy())
        tourE, makespanE = self.LARGEEDGE(self.G.copy())
        if makespanA < makespanE:
            return tourA, makespanA
        return tourE, makespanE

    def tour_splitting(self, tour, cost):
        # implement tour splitting
        c_max = max([self.G[u][v]['weight'] for (u, v) in self.G.edges if u == self.initial_vertex])
        tours = {j: [] for j in range(1, self.k + 1)}
        if tour == []: return tours
        tour_from_idx, tour_to_idx = {}, {}
        last_edge_j = tour[0]
        assert last_edge_j[0] == self.initial_vertex
        traversed_cost = self.edge_cost(tour[0])
        tour_from_idx[1] = 1
        i = 1
        for j in range(1, self.k):
            cost_j = j / self.k * (cost - 2 * c_max) + c_max
            i_init = tour_from_idx[j]
            while traversed_cost < cost_j:
                traversed_cost += self.edge_cost(tour[i])
                i += 1
                if i >= len(tour):
                    tours[j] = tour[i_init:]
                    return tours

            previous_to_last_edge_j = tour[i - 2]
            last_edge_j = tour[i - 1]
            r_j = cost_j - traversed_cost
            if len(last_edge_j) == 2:
                # it is an edge and we drop it
                tour_to_idx[j] = i - 2
                tour_from_idx[j + 1] = i
            else:
                assert len(last_edge_j) == 3
                # it is an arc, and we need to place it in R(j) or R(j+1)
                (u, v, arc_weight) = last_edge_j

                if self.G[self.initial_vertex][u]['weight'] + r_j <= arc_weight['weight'] - r_j + \
                        self.G[v][self.initial_vertex]['weight']:
                    # initial_vertex[j+1] = u # the arc is in R(j+1)
                    tour_from_idx[j + 1] = i - 1
                    if len(previous_to_last_edge_j) == 2:
                        # it is an edge, and we drop it
                        tour_to_idx[j] = i - 3
                    else:
                        assert len(previous_to_last_edge_j) == 3
                        # it is an arc, and we keep it
                        tour_to_idx[j] = i - 2
                else:
                    # terminal_vertex[j] = v # the arc is in R(j)
                    tour_to_idx[j] = i - 1
                    next_to_edge_j = tour[i]
                    if len(next_to_edge_j) == 2:
                        # it is an edge, and we drop it
                        tour_from_idx[j + 1] = i + 1
                    else:
                        assert len(next_to_edge_j) == 3
                        # it is an arc, and we keep it
                        tour_from_idx[j + 1] = i
            # implement 4.
            if tour_to_idx[j] + 1 < len(tour) and tour_from_idx[j] < len(tour) and tour_from_idx[j] <= tour_to_idx[j]:
                tours[j] = [(self.initial_vertex, tour[tour_from_idx[j]][0])] \
                           + tour[tour_from_idx[j]:tour_to_idx[j] + 1] \
                           + [(tour[tour_to_idx[j]][1], self.initial_vertex)]

        tour_to_idx[self.k] = len(tour) - 2
        if tour_from_idx[self.k] < len(tour):
            tours[self.k] = [(self.initial_vertex, tour[tour_from_idx[self.k]][0])] \
                            + tour[tour_from_idx[self.k]:]
        return tours

    def check_arcs(self, tours):
        check = {str(a): 0 for a in self.A if a[1] != self.initial_vertex}
        for j, tour in tours.items():
            for i in range(len(tour)):
                if len(tour[i]) == 3:
                    a = tour[i]
                    check[str(a)] += 1
        for a,count in check.items():
            if count != 1:
                raise Exception(f"Arc {a} visited {count} times")
        return True

    def kCRANES(self):
        # implement 1.
        tour, cost = self.CRANE()
        tours = self.tour_splitting(tour, cost)
        self.check_arcs(tours)
        return tours
