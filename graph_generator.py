import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy
from parameter import *
from node import Node
from graph import Graph, a_star

class Graph_generator:
    def __init__(self, map_size, k_size, sensor_range, target_position, plot=False):
        self.k_size = k_size
        self.graph = Graph()
        self.ground_truth_graph = Graph()
        self.node_coords = None
        self.plot = plot
        self.x = []
        self.y = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.route_node = []
        self.nodes_list = []
        self.node_utility = None
        self.indicator = None
        self.direction_vector = None
        self.target_position = target_position
    
    def generate_node_coords(self, robot_location, robot_belief):
        free_area = self.free_area(robot_belief)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]
        node_coords = np.concatenate((robot_location.reshape(1, 2), self.target_position.reshape(1, 2), node_coords))
        return self.unique_coords(node_coords).reshape(-1, 2)

    def edge_clear_all_nodes(self):
        self.graph = Graph()
        self.x = []
        self.y = []
        
    def edge_clear(self, coords):
        node_index = str(self.find_index_from_coords(self.node_coords, coords))
        self.graph.clear_edge(node_index)

    def generate_graph(self, robot_location, ground_truth_belief, robot_belief, frontiers):        
        self.node_coords = self.generate_node_coords(robot_location, robot_belief)
        self.ground_truth_node_coords = self.generate_node_coords(robot_location, ground_truth_belief)
        self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)
        self.find_k_neighbor_all_nodes(self.ground_truth_node_coords, ground_truth_belief, ground_truth=True)
        
        self.node_utility = []
        self.direction_vector = []
        for coords in self.node_coords:
            node = Node(coords, frontiers, robot_belief, self.target_position)
            self.nodes_list.append(node)
            utility = node.utility
            direction_vector = node.direction_vector
            self.direction_vector.append(direction_vector)
            self.node_utility.append(utility)

        self.direction_vector = np.array(self.direction_vector)
        self.node_utility = np.array(self.node_utility)
        self.indicator = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:,0] + self.node_coords[:,1]*1j
        for node in self.route_node:
            index = np.argwhere(x.reshape(-1) == node[0]+node[1]*1j)[0]
            self.indicator[index] = 1
        return self.node_coords, self.graph.edges, self.node_utility, self.indicator, self.direction_vector

    def update_graph(self, robot_position, robot_belief, old_robot_belief, frontiers, old_frontiers):
        # add uniform points in the new free area to the node coords
        new_free_area = self.free_area((robot_belief - old_robot_belief > 0) * 255)
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        new_node_coords = self.uniform_points[candidate_indices]
        old_node_coords = copy.deepcopy(self.node_coords)
        self.node_coords = np.concatenate((self.node_coords, new_node_coords, self.target_position.reshape(1, 2)))
        self.node_coords = self.unique_coords(self.node_coords).reshape(-1, 2)
        self.edge_clear_all_nodes()
        self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)
        # update the observable frontiers through the change of frontiers
        old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
        new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        observed_frontiers_index = np.where(
            np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
        new_frontiers_index = np.where(
            np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
        observed_frontiers = old_frontiers[observed_frontiers_index]
        new_frontiers = frontiers[new_frontiers_index]
        for node in self.nodes_list:
            if np.linalg.norm(node.coords - robot_position) > 2 * self.sensor_range:
                pass
            elif node.zero_utility_node is True:
                pass
            else:
                node.update_observable_frontiers(observed_frontiers, new_frontiers, robot_belief)

        for new_coords in new_node_coords:
            node = Node(new_coords, frontiers, robot_belief, self.target_position)
            self.nodes_list.append(node)
        self.direction_vector = []
        self.node_utility = []
        for i, coords in enumerate(self.node_coords):
            utility = self.nodes_list[i].utility
            node = Node(coords, frontiers, robot_belief, self.target_position)
            self.node_utility.append(utility)
            direction_vector = node.direction_vector
            self.direction_vector.append(direction_vector)
        self.direction_vector = np.array(self.direction_vector)
        self.node_utility = np.array(self.node_utility)
        self.indicator = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:, 0] + self.node_coords[:, 1] * 1j
        for node in self.route_node:
            index = np.argwhere(x.reshape(-1) == node[0] + node[1] * 1j)
            self.indicator[index] = 1
        return self.node_coords, self.graph.edges, self.node_utility, self.indicator, self.direction_vector

    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, 30).round().astype(int)
        y = np.linspace(0, self.map_y - 1, 30).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_belief):
        # free area 255
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_nearest_node_index(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        return index
    
    def find_index_from_coords(self, node_coords, p):
        return np.argmin(np.linalg.norm(node_coords - p, axis=1))
    
    def find_k_neighbor(self, coords, node_coords, robot_belief):
        dist_list = np.linalg.norm((coords - node_coords), axis=-1)
        sorted_index = np.argsort(dist_list)
        k = 0
        neighbor_index_list = []
        while k < self.k_size and k < node_coords.shape[0]:
            neighbor_index = sorted_index[k]
            neighbor_index_list.append(neighbor_index)
            start = coords
            end = node_coords[neighbor_index]
            if not self.check_collision(start, end, robot_belief):
                a = str(self.find_index_from_coords(node_coords, start))
                b = str(neighbor_index)
                dist = np.linalg.norm(start - end)
                self.graph.add_node(a)
                self.graph.add_edge(a, b, dist)
                # test
                self.graph.add_node(b)
                self.graph.add_edge(b, a, dist)

            k += 1
        return neighbor_index_list

    def find_k_neighbor_all_nodes(self, node_coords, robot_belief, ground_truth=False):
        X = node_coords
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            for j, neighbour in enumerate(X[indices[i][:]]):
                start = p
                end = neighbour
                if not self.check_collision(start, end, robot_belief):
                    a = str(self.find_index_from_coords(node_coords, p))
                    b = str(self.find_index_from_coords(node_coords, neighbour))
                    if not ground_truth:
                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[i, j])
                        if self.plot:
                            self.x.append([p[0], neighbour[0]])
                            self.y.append([p[1], neighbour[1]])
                    else:
                        self.ground_truth_graph.add_node(a)
                        self.ground_truth_graph.add_edge(a, b, distances[i, j])

    def find_index_from_coords(self, node_coords, p):
        return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-5)[0][0]

    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False
        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision

    def find_shortest_path(self, current, destination, node_coords, graph):
        start_node = str(self.find_index_from_coords(node_coords, current))
        end_node = str(self.find_index_from_coords(node_coords, destination))
        route, dist = a_star(int(start_node), int(end_node), node_coords, graph)
        if start_node != end_node:
            assert route != []
        route = list(map(str, route))
        return dist, route

