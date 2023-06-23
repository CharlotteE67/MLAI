import collections
import random
import numpy as np
# from collections import defaultdict
import heapq

class WDGraph():
    def __init__(self, nodes, edges):
        """
        self.graph is a dictionary to collect nodes with their connected nodes and weights
        e.g. {node:[(cost,neighbour), ...]}
        """
        self.graph = collections.defaultdict(list)
        self.nodes = nodes
        self.edges = edges
        self.add_edge(edges)
        self.sp_record = self.global_record()
    
    def add_edge(self, edges):
        """
        add edges to the graph, input form in edge list is "from_node, to_node and weight"
        """
        for edge in edges:
            (from_node, to_node, weight) = edge
            self.graph[from_node].append((weight, to_node))

    def delete_edge(self, from_node, to_node):
        """
        delete an edge to the graph, input form is "from_node, to_node"
        weight is not needed
        """
        for i in range(len(self.graph[from_node])):
            if self.graph[from_node][i][1] == to_node:
                self.graph[from_node].pop(i)
                break

    def shortest_path(self, start_node, target_node):
        num_nodes = len(self.graph)
        distances = {node: float('inf') for node in self.nodes}
        # distances = {node: float('inf') for node in range(num_nodes)}
        distances[start_node] = 0
        prev_nodes = {node: None for node in self.nodes}
        # prev_nodes = {node: None for node in range(num_nodes)}
        # print(prev_nodes)

        for _ in range(num_nodes - 1):
            for node in self.nodes:
                for weight, neighbor in self.graph[node]:
                    if distances[node] + weight < distances[neighbor]:
                        distances[neighbor] = distances[node] + weight
                        prev_nodes[neighbor] = node

        path = self.reconstruct_path(start_node, target_node, prev_nodes)
        return distances[target_node], path

    def reconstruct_path(self, start_node, target_node, prev_nodes):
        path = []
        current_node = target_node
        while current_node is not None:
            path.append(current_node)
            current_node = prev_nodes[current_node]
        path.reverse()
        return path

    def bellman_ford(self, start_node):
        graph = self.graph
        distances = {node: float('inf') for node in graph.nodes}
        distances[start_node] = 0
        
        for _ in range(len(graph.nodes) - 1):
            for u, v, weight in graph.edges(data='weight'):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
        
        return distances


    def shortest_path_bfs(self, start, end):
        """
        provide start node and end node, the method return the shortest path by BFS
        """
        # cannot search and give error if the graph is not constructed
        if not self.graph:
            raise Exception("Graph is not constructed!")
        # create a priority queue and hash set to store visited nodes
        queue, visited = [(0, start, [])], set()
        heapq.heapify(queue)
        # traverse graph with BFS
        while queue:
            (cost, node, path) = heapq.heappop(queue)
            # visit the node if it was not visited before
            if node not in visited:
                visited.add(node)
                path = path + [node]
                # hit the end
                if node == end:
                    return (cost, path)
                # visit neighbours
                for c, neighbour in self.graph[node]:
                    if neighbour not in visited:
                        heapq.heappush(queue, (cost+c, neighbour, path))
        return float("inf")
    
    def global_record(self):
        record = {}
        for s in self.nodes:
            for e in self.nodes:
                sp = self.shortest_path(s, e)
                pair = s + e
                record[pair] = sp
        return record


class RLShortestPath:
    def __init__(self, graph):
        self.G = graph
        self.R = {}
        self.q_values = self.init_q_values()  # apply q learning
        self.weights = self.init_weights()   # apply value function approximation
        self.node2num = {}
        for i in range(len(graph.nodes)):
            self.node2num[graph.nodes[i]] = i
        # print(self.node2num)

    def init_q_values(self):
        q_values = {}
        for node in self.G.nodes:
            q_values[node] = {}
            for weight, neighbor in self.G.graph[node]:
                q_values[node][neighbor] = 0
                pair = node+neighbor
                self.R[pair] = weight
        return q_values
    
    def init_weights(self):
        num_features = len(self.G.graph) * 2  # Features: node visited (0/1) for each node
        weights = np.zeros(num_features)
        return weights
    
    def get_features(self, node, visited_nodes):
        num_nodes = len(self.G.graph)
        features = np.zeros(num_nodes * 2)
        node_id = self.node2num[node]
        features[node_id] = 1
        features[num_nodes + node_id] = visited_nodes.count(node)
        return features
    
    def value_function(self, state):
        # print(self.weights)
        # print(state)
        return np.dot(self.weights, state)
    
    def update_weights(self, state, target, learning_rate):
        prediction = self.value_function(state)
        error = target - prediction
        # print(error)
        # print(state)
        self.weights += learning_rate * error * state
    
    def train_value_function(self, start_node, target_node, num_episodes=1500, learning_rate=0.1, discount_factor=0.9):
        for episode in range(num_episodes):
            current_node = start_node
            visited_nodes = []
            while current_node != target_node:
                visited_nodes.append(current_node)
                state = self.get_features(current_node, visited_nodes)
                if not self.G.graph[current_node]:
                    break
                next_node = random.choice([neighbor for _, neighbor in self.G.graph[current_node]])
                reward = 1-self.R[current_node+next_node] if next_node == target_node else -self.R[current_node+next_node]
                next_state = self.get_features(next_node, visited_nodes)
                target = reward + discount_factor * self.value_function(next_state)
                self.update_weights(state, target, learning_rate)
                current_node = next_node

    def train_q_learning(self, start_node, target_node, num_episodes=400, learning_rate=0.1, discount_factor=0.9):
        # for episode in range(num_episodes):
        #     rand_start = self.G.nodes[]

        for episode in range(num_episodes):
            current_node = start_node
            while current_node != target_node:
                if not self.q_values[current_node]:
                    break
                action = self.get_epsilon_greedy_action(current_node)
                next_node = action
                reward = 1-self.R[current_node+next_node] if next_node == target_node else -self.R[current_node+next_node]
                self.update_q_value(current_node, action, next_node, reward, learning_rate, discount_factor)
                current_node = next_node

    def get_epsilon_greedy_action(self, node, epsilon=0.1):
        # print(random.choice(list(self.G.graph[node])))
        # print(max(self.q_values[node], key=self.q_values[node].get))
        if random.random() < epsilon:
            (weight, action) = random.choice(list(self.G.graph[node]))
            return action
        else:
            return max(self.q_values[node], key=self.q_values[node].get)

    def update_q_value(self, current_node, action, next_node, reward, learning_rate, discount_factor):
        # print(self.q_values[next_node])
        if not self.q_values[next_node]:
            max_q_value = 0
        else:
            max_q_value = max(self.q_values[next_node].values())
        self.q_values[current_node][action] += learning_rate * (reward + discount_factor * max_q_value - self.q_values[current_node][action])

    def shortest_path_based_on_q_value(self, start_node, target_node):
        print(self.q_values)
        current_node = start_node
        path = [current_node]
        while current_node != target_node:
            if not self.q_values[current_node]:
                return float("inf")
            action = max(self.q_values[current_node], key=self.q_values[current_node].get)
            next_node = action
            path.append(next_node)
            current_node = next_node
        return path
    
    def shortest_path_based_on_VFA(self, start_node, target_node):
        current_node = start_node
        path = [current_node]
        while current_node != target_node:
            visited_nodes = path
            state = self.get_features(current_node, visited_nodes)
            if not self.G.graph[current_node]:
                return float("inf")
            next_node = random.choice([neighbor for _, neighbor in self.G.graph[current_node]])
            path.append(next_node)
            current_node = next_node
        return path


nodes = ["A", "B", "C", "D", "E", "F", "G"]
edges = [
        ("A", "B", 7),
        ("A", "D", 5),
        ("B", "C", 8),
        ("B", "D", 9),
        ("B", "E", 7),
        ("C", "E", 5),
        ("D", "E", 15),
        ("D", "F", 6),
        ("E", "F", 8),
        ("E", "G", 9),
        ("F", "G", 11)
    ]
# edges = [
#     ("A", "B", 10),
#     ("A", "C", 1),
#     ("C", "B", 11)
# ]


# edges = [('7', '10', 1), ('17', '13', 2), ('8', '20', 11), ('13', '19', 29), ('3', '10', 11), ('5', '8', 6), ('5', '11', 17), ('20', '15', 7), ('5', '14', 16), ('18', '6', 27), ('14', '8', 3), ('11', '16', 19), ('13', '6', 14), ('12', '8', 23), ('13', '2', 2), ('4', '1', 4), ('6', '4', 11), ('7', '19', 20), ('1', '2', 14), ('20', '18', 16), ('15', '6', 27), ('19', '3', 11), ('11', '5', 26), ('3', '6', 6), ('2', '5', 29), ('5', '1', 30), ('8', '15', 26)]
# nodes = ['1', '14', '11', '4', '3', '12', '9', '15', '17', '5', '19', '6', '13', '8', '18', '16', '7', '2', '20', '10']
graph = WDGraph(nodes, edges)

print(graph.graph)
print(graph.sp_record)
# print(graph.sp_record['GC'][0] == float("inf"))
# exit(0)
# graph.delete_edge("B", "E")
# s, e = "F", "C"

rl = RLShortestPath(graph)
s, e = "A", "G"
# s, e = "1", "7"
# rl.train_q_learning(s, e)
# path = rl.shortest_path_based_on_q_value(s, e)
# print(path)
rl.train_value_function(s, e)
path = rl.shortest_path_based_on_VFA(s, e)
print(path)
print(graph.shortest_path(s, e))
print(graph.shortest_path_bfs(s, e))

# print ("Find the shortest path with Dijkstra")
# print (edges)
# print ("A -> E:")
# print (graph.shortest_path("A", "E"))
# print(graph.shortest_path('G', 'A'))
# print(graph.shortest_path('A', 'B'))

# # cycle situation
# edges = [('8', '7', 21), ('6', '18', 11), ('13', '10', 3), ('4', '7', 4), ('14', '4', 30), ('8', '1', 17), 
#          ('19', '7', 28), ('13', '5', 15), ('10', '20', 15), ('7', '19', 21), ('16', '8', 20), ('14', '2', 21), 
#          ('4', '13', 18), ('11', '14', 15), ('1', '18', 10), ('9', '6', 18), ('11', '13', 25), ('4', '18', 8), 
#          ('2', '13', 6), ('5', '19', 26), ('5', '14', 25), ('6', '16', 20), ('20', '8', 24), ('18', '20', 26), 
#          ('14', '8', 6), ('17', '18', 16), ('10', '16', 1), ('14', '13', 19), ('12', '6', 9), ('5', '1', 22), 
#          ('11', '17', 19), ('3', '16', 9), ('9', '1', 10), ('19', '11', 30), ('7', '4', 30), ('5', '10', 3), 
#          ('13', '2', 22), ('2', '8', 10), ('16', '14', 12), ('18', '11', 17), ('16', '11', 30), ('20', '2', 10), 
#          ('7', '20', 30), ('19', '5', 4), ('4', '3', 22), ('16', '1', 23), ('2', '17', 9), ('15', '12', 7), 
#          ('5', '16', 21), ('3', '13', 23), ('11', '8', 7), ('12', '2', 8), ('10', '13', 19), ('20', '12', 11), 
#          ('12', '13', 1), ('13', '6', 29), ('14', '18', 2), ('2', '15', 20)]
# nodes = ['4', '2', '3', '1', '12', '16', '11', '7', '17', '14', '15', '13', '8', '6', '5', '19', '18', '10', '9', '20']