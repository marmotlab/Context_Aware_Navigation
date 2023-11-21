import matplotlib.pyplot as plt
import os
import math
import skimage.io
from skimage.measure import block_reduce
from scipy import *
from sensor import *
from graph_generator import *
from node import *
import time

class Env():
    def __init__(self, map_index, k_size=20, plot=False, test=False):
        self.test = test
        if self.test:
            self.map_dir = f'DungeonMaps/pp/test'
        else:
            self.map_dir = f'DungeonMaps/pp/train'
        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.ground_truth, self.start_position, self.target_position = self.import_ground_truth_pp(
            self.map_dir + '/' + self.map_list[self.map_index])
        self.ground_truth_size = np.shape(self.ground_truth)
        self.robot_belief = np.ones(self.ground_truth_size) * 127  # unexplored
        self.downsampled_belief = None
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.resolution = 4  # downsample belief
        self.sensor_range = 80
        self.explored_rate = 0
        self.frontiers = None
        self.graph_generator = Graph_generator(map_size=self.ground_truth_size, sensor_range=self.sensor_range, k_size=k_size, target_position = self.target_position, plot=plot)
        self.graph_generator.route_node.append(self.start_position)
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector = None, None, None, None, None
        
        self.begin()

        self.plot = plot
        self.frame_files = []
        if self.plot:
            self.xPoints = [self.start_position[0]]
            self.yPoints = [self.start_position[1]]
            self.xTarget = [self.target_position[0]]
            self.yTarget = [self.target_position[1]]

    def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        return index

    def begin(self):
        self.robot_belief = self.update_robot_belief(self.start_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        self.frontiers = self.find_frontier()
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector = self.graph_generator.generate_graph(
            self.start_position, self.ground_truth, self.robot_belief, self.frontiers)

    def step(self, robot_position, next_position, travel_dist):
        dist = np.linalg.norm(robot_position - next_position)
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        astar_dist_cur_to_target, _ = self.graph_generator.find_shortest_path(robot_position, self.target_position, 
                                                                           self.graph_generator.ground_truth_node_coords, self.graph_generator.ground_truth_graph)
        astar_dist_next_to_target, _ = self.graph_generator.find_shortest_path(next_position, self.target_position, 
                                                                            self.graph_generator.ground_truth_node_coords, self.graph_generator.ground_truth_graph)
        travel_dist += dist
        robot_position = next_position
        self.graph_generator.route_node.append(robot_position)
        next_node_index = self.find_index_from_coords(robot_position)
        self.graph_generator.nodes_list[next_node_index].set_visited()
        self.robot_belief = self.update_robot_belief(robot_position, self.sensor_range, self.robot_belief,
                                                     self.ground_truth)
        self.downsampled_belief = block_reduce(self.robot_belief.copy(), block_size=(self.resolution, self.resolution),
                                               func=np.min)
        frontiers = self.find_frontier()
        self.explored_rate = self.evaluate_exploration_rate()
        reward, done = self.calculate_reward(astar_dist_cur_to_target, astar_dist_next_to_target, dist_to_target)
        if self.plot:
            self.xPoints.append(robot_position[0])
            self.yPoints.append(robot_position[1])
        self.node_coords, self.graph, self.node_utility, self.indicator, self.direction_vector = self.graph_generator.update_graph(
            robot_position, self.robot_belief, self.old_robot_belief, frontiers, self.frontiers)
        self.old_robot_belief = copy.deepcopy(self.robot_belief)
        self.frontiers = frontiers

        return reward, done, robot_position, travel_dist
    
    def import_ground_truth_pp(self, map_index):
        ground_truth = (skimage.io.imread(map_index, 1) * 255).astype(int)
        robot_location = np.nonzero(ground_truth == 209)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        target_location = np.nonzero(ground_truth == 68)
        target_location = np.array([np.array(target_location)[1, 127], np.array(target_location)[0, 127]])
        ground_truth = (ground_truth > 150)|((ground_truth<=80)&(ground_truth>=60))
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location, target_location
    
    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_belief

    def calculate_reward(self, astar_dist_cur_to_target, astar_dist_next_to_target, dist_to_target):
        reward = 0
        done = False
        reward -= 0.5
        reward += (astar_dist_cur_to_target - astar_dist_next_to_target) / 64
        if dist_to_target == 0:
            reward += 20
            done = True
        return reward, done

    def evaluate_exploration_rate(self):
        rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)
        return rate

    def find_frontier(self):
        y_len = self.downsampled_belief.shape[0]
        x_len = self.downsampled_belief.shape[1]
        mapping = self.downsampled_belief.copy()
        belief = self.downsampled_belief.copy()
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)
        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        f = points[ind_to]
        f = f.astype(int)
        f = f * self.resolution
        return f

    def plot_env(self, n, path, step, travel_dist):
        plt.switch_backend('agg')
        # plt.ion()
        plt.cla()
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        # for i in range(len(self.graph_generator.x)):
        #    plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'tan', zorder=1)  # plot edges will take long time
        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c=self.node_utility, zorder=5)
        plt.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, zorder=3)
        plt.plot(self.xTarget, self.yTarget, 'o', markersize = 20)
        plt.plot(self.xPoints, self.yPoints, 'b', linewidth=2)
        plt.plot(self.xPoints[-1], self.yPoints[-1], 'mo', markersize=8)
        plt.plot(self.xPoints[0], self.yPoints[0], 'co', markersize=8)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, travel_dist))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=150))
        plt.show()
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)