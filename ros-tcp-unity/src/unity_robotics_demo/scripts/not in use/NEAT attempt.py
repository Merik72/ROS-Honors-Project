#!/usr/bin/env python
# Evolve a control/reward estimation network for the Robot driving itself

# todo - just idk hook this shit up to unity and see how it goes

import multiprocessing
import os
import pickle
import random
import time
from unity_robotics_demo_msgs.msg import UnityColor

import matplotlib.pyplot as plt
import numpy as np

import neat
import visualize

from sklearn.preprocessing import Normalizer
import pandas as pd

import rospy
# import color_publisher
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist

#NUM_CORES = multiprocessing.cpu_cout()
NUM_CORES = 8

lidar_data = None 
received_reward = None
received_termination = None

WHEEL_TOPIC = 'TTTTT'
RESET_TOPIC = 'color'
LIDAR_TOPIC = '/mid/points'
NODE_NAME = 'learning'

def post_reset():
    pub = rospy.Publisher(RESET_TOPIC, UnityColor, queue_size=10)

    color = UnityColor(0, 0, 0, 1)

    #wait_for_connections(pub, TOPIC_NAME)
    pub.publish(color)
    time.sleep(1)
    pub.publish(color)
    time.sleep(1)

def post_wheels(linear=None, angular=None):
    if linear==None:
        pub = rospy.Publisher(WHEEL_TOPIC, Twist, queue_size=10)
        x = random.uniform(-1,1)
        z = random.uniform(-1,1)
        linear = Vector3(x, 0, 0)
        angular = Vector3(0, 0, z)
        #linear = Vector3(1, 0, 0)
        #angular = Vector3(0, 0, 1)
        twist = Twist(linear, angular)


        pub.publish(twist)
        time.sleep(0.1)
    else:
        pub = rospy.Publisher(WHEEL_TOPIC, Twist, queue_size=10)
        pub.publish(Twist(Vector3(linear,0,0), Vector3(0,0,angular)))
        time.sleep(0.1)

def get_lidar():
    hear_lidar()

def hear_lidar():
    rospy.Subscriber(LIDAR_TOPIC, PointCloud2, callback_lidar)
    
def callback_lidar(cloud_msg):
    global lidar_data
    # Helper function to convert PointCloud2 message to numpy array
    dtype_list = []
    for field in cloud_msg.fields:
        if field.datatype == PointField.FLOAT32:
            dtype_list.append((field.name, np.float32))
        elif field.datatype == PointField.UINT16:
            dtype_list.append((field.name, np.uint16))
        # Add other field data types if needed
    # Create numpy array from the point cloud data
    cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.dtype(dtype_list))
    x = cloud_arr['x']
    y = cloud_arr['y']
    z = cloud_arr['z']
    intensity = cloud_arr['intensity']
    ring = cloud_arr['ring']
    time = cloud_arr['time']    

    # Stack columns to form an (n, 6) array
    points_array = np.column_stack((x, y, z, intensity, ring, time))    
    lidar_data = points_array[:500,:3]
    lidar_data=lidar_data.flatten()

def hear_reward():
    rospy.Subscriber('/reward', Float32, callback_reward)

def callback_reward(msg):
    global received_reward
    received_reward = msg.data
    time.sleep(0.1)

def hear_termination():
    rospy.Subscriber('/terminated', Bool, callback_termination)

def callback_termination(msg):
    global received_termination
    received_termination = msg.data
    time.sleep(0.1)

#Takes a numpy array and outputs a pandas dataframe
#def normalize(input):
#    output_dataset = pd.DataFrame(input[:,:3], columns=['x','y','z'])
#    normalizer = Normalizer()
#    normalized_lidar = normalizer.fit_transform(output_dataset)
#    return pd.DataFrame(normalized_lidar)

class dingoGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.discount = None

    def configure_new(self, config):
        super().configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff

    def __str__(self):
        return f"Reward discount: {self.discount}\n{super().__str__()}"

def compute_fitness(genome, net, episodes, min_reward, max_reward):
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_error = []
    for score, data in episodes:
        # Compute normalized discounted reward.
        dr = np.convolve(data[:, -1], discount_function)[m:]
        dr = 2 * (dr - min_reward) / (max_reward - min_reward) - 1.0
        dr = np.clip(dr, -1.0, 1.0)

        for row, dr in zip(data, dr):
            observation = row[:8] 
                # is there a way to make this LIDAR
            action = float(row[8]) 
                # is there a way to make this UNITY
            output = net.activate(observation)  
                # is there a way to make this LIDAR
            reward_error.append(float((output[action] - dr) ** 2))

    return reward_error

class PooledErrorCompute(object):
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.test_episodes = []
        self.generation = 0

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

    def simulate(self, nets):
        global lidar_data
        global received_reward
        global received_termination
        scores = []
        for genome, net in nets:
            
            # observation_init_vals, obervation_init_info = env.reset()
                #send lidar here?
            post_reset()
            
            while lidar_data is None:
                hear_lidar()
                observation_init_vals = lidar_data
            step = 0
            data = []
            while 1:
                action = None
                step += 1
                if step < 200: #and random.random() < 0.2:
                    action = [random.uniform(-1,1), random.uniform(-1,1)]
                        # make into a wheel movement?
                        # UNITY
                else:
                    output = net.activate(observation_init_vals)
                    action = np.argmax(output)
                
                    
                try:
                    post_wheels(action[0], action[1])
                except rospy.ROSInterruptException:
                    pass
                # Note: done has been deprecated.

                try:
                    hear_lidar()
                    observation = lidar_data
                except rospy.ROSInterruptException:
                    pass
                try:
                    hear_reward()
                    reward = received_reward
                except rospy.ROSInterruptException:
                    pass
                try:
                    hear_termination()
                    terminated = received_termination
                except rospy.ROSInterruptException:
                    pass
                    # update the python variables based on the current state of the simulation UNITY
                print(lidar_data)
                print(observation)
                data.append(np.hstack((observation, action, reward)))

                if terminated:
                    break

            data = np.array(data)
            score = np.sum(data[:, -1])
            self.episode_score.append(score)
            scores.append(score)
            self.episode_length.append(step)

            self.test_episodes.append((score, data))

        print("Score range [{:.3f}, {:.3f}]".format(min(scores), max(scores)))

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        print("network creation time {0}".format(time.time() - t0))
        t0 = time.time()

        # Periodically generate a new set of episodes for comparison.
        if 1 == self.generation % 10:
            self.test_episodes = self.test_episodes[-300:]
            self.simulate(nets)
            print("simulation run time {0}".format(time.time() - t0))
            t0 = time.time()

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.
        print("Evaluating {0} test episodes".format(len(self.test_episodes)))
        if self.num_workers < 2:
            for genome, net in nets:
                reward_error = compute_fitness(genome, net, self.test_episodes, self.min_reward, self.max_reward)
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
        else:
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = []
                for genome, net in nets:
                    jobs.append(pool.apply_async(compute_fitness,
                                                 (genome, net, self.test_episodes,
                                                  self.min_reward, self.max_reward)))

                for job, (genome_id, genome) in zip(jobs, genomes):
                    reward_error = job.get(timeout=None)
                    genome.fitness = -np.sum(reward_error) / len(self.test_episodes)

        print("final fitness compute time {0}\n".format(time.time() - t0))


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(dingoGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute(NUM_CORES)

    global lidar_data
    global received_reward
    global received_termination
    while 1:
        try:
            gen_best = pop.run(ec.evaluate_genomes, 5)

            # print(gen_best)

            visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

            plt.plot(ec.episode_score, 'g-', label='score')
            plt.plot(ec.episode_length, 'b-', label='length')
            plt.grid()
            plt.legend(loc='best')
            plt.savefig("scores.svg")
            plt.close()

            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Use the best genomes seen so far as an ensemble-ish control system.
            best_genomes = stats.best_unique_genomes(3)
            best_networks = []
            for g in best_genomes:
                best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))

            solved = True
            best_scores = []
            for k in range(100):
                global lidar_data
                global received_reward
                global received_termination
                #observation_init_vals, observation_init_info = env.reset() # Unity send reset
                post_reset()
                hear_lidar()
                observation_init_vals = lidar_data
                score = 0
                step = 0
                while 1:
                    step += 1
                    # Use the total reward estimates from all five networks to
                    # determine the best action given the current state.
                    votes = np.zeros((4,))
                    for n in best_networks:
                        output = n.activate(observation_init_vals)
                        votes[np.argmax(output)] += 1

                    best_action = np.argmax(votes)
                    # Note: done has been deprecated.
                    post_wheels(best_action[0], best_action[1])
                    # Note: done has been deprecated.
                    
                    observation = None
                    reward = None
                    terminated = None
                    while observation is None: 
                        hear_lidar()
                        observation = lidar_data
                    while reward is None:
                        hear_reward()
                        reward = received_reward
                    while terminated is None: 
                        hear_termination()
                        terminated = received_termination
                            # show best actor going in Unity
                    score += reward
                    # env.render()
                        # show best actor going in Unity
                    if terminated:
                        break

                ec.episode_score.append(score)
                ec.episode_length.append(step)

                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                print(k, score, avg_score)
                if avg_score < 200:
                    solved = False
                    break

            if solved:
                print("Solved.")

                # Save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name + '.pickle', 'wb') as f:
                        pickle.dump(g, f)

                    visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
                    visualize.draw_net(config, g, view=False, filename=name + "-net-pruned.gv", prune_unused=True)

                break
        except KeyboardInterrupt:
            print("User break.")
            break

    #env.close()
    # log off untiy


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True, disable_signals=True)
    run()
