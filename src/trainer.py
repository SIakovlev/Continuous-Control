import gym
import random
import numpy as np
import torch
import os
import sys
from agent import Agent
from unity_env import UnityEnv
from collections import deque
import datetime
import logging
from pprint import pprint
from uo_process import UOProcess

if not os.path.exists('../logs'):
    os.makedirs('../logs')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../logs/run_' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M') + '.log',
                    level=logging.INFO)


class Trainer:
    def __init__(self, params):

        seed = params['general_params']['seed']
        self.__set_seed(seed=seed)

        env_params = params['env_params']
        env_params['seed'] = seed
        self.env = UnityEnv(params=env_params)

        agent_params = params['agent_params']
        agent_params['state_size'] = self.env.observation_space.shape[0]
        agent_params['action_size'] = self.env.action_space_size
        self.agent = Agent(params=agent_params)

        trainer_params = params['trainer_params']
        self.learning_rate_decay = trainer_params['learning_rate_decay']
        self.results_path = trainer_params['results_path']
        self.model_path = trainer_params['model_path']
        self.t_max = trainer_params['t_max']

        self.exploration_noise = UOProcess()

        # data gathering variables
        self.avg_rewards = []
        self.scores = []
        self.score = 0

        self.sigma = 0.5

        print("Configuration:")
        pprint(params)
        logging.info("Configuration: {}".format(params))

    def train(self, num_of_episodes):

        logging.info("Training:")
        reward_window = deque(maxlen=100)
        # reward_matrix = np.zeros((num_of_episodes, 300))

        for episode_i in range(1, num_of_episodes):

            state = self.env.reset()
            self.agent.reset(self.sigma)
            total_reward = 0
            total_loss = 0

            self.sigma *= 0.98
            done = np.zeros((len(state), 1), dtype=np.bool)
            counter = 0
            while not any(done):
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state

                # DEBUG
                # logging.info("epsiode: {}, reward: {}, counter: {}, action: {}".
                #              format(episode_i, reward, counter, action))

                total_loss += self.agent.agent_loss
                total_reward += np.array(reward)
                # reward_matrix[episode_i, counter] = reward
                counter += 1

            reward_window.append(total_reward)
            self.avg_rewards.append(np.mean(total_reward))
            print('\rEpisode {}\tCurrent Score: {:.2f}\tAverage Score: {:.2f} '
                  '\t\tTotal loss: {:.2f}\tLearning rate (actor): {:.4f}\tLearning rate (critic): {:.4f}'.
                  format(episode_i, np.mean(total_reward), np.mean(reward_window),
                         total_loss, self.agent.learning_rate_actor, self.agent.learning_rate_critic), end="")

            logging.info('Episode {}\tCurrent Score (average over 20 robots): {:.2f}\tAverage Score (over episodes): {:.2f} '
                         '\t\tTotal loss: {:.2f}\tLearning rate (actor): {:.4f}\tLearning rate (critic): {:.4f}'.
                         format(episode_i, np.mean(total_reward), np.mean(reward_window),
                                total_loss, self.agent.learning_rate_actor, self.agent.learning_rate_critic))

            self.agent.learning_rate_actor *= self.learning_rate_decay
            self.agent.learning_rate_critic *= self.learning_rate_decay
            self.agent.set_learning_rate(self.agent.learning_rate_actor, self.agent.learning_rate_critic)

            if episode_i % 100 == 0:

                avg_reward = np.mean(np.array(reward_window))
                print("\rEpisode: {}\tAverage total reward: {:.2f}".format(episode_i, avg_reward))

                if avg_reward >= 30.0:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_i - 100,
                                                                                                 avg_reward))
                    if not os.path.exists(self.model_path):
                        os.makedirs(self.model_path)
                    torch.save(self.agent.get_actor().state_dict(), self.model_path + 'checkpoint_actor_{}.pth'.format(
                        datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')))
                    torch.save(self.agent.get_critic().state_dict(), self.model_path + 'checkpoint_critic_{}.pth'.format(
                        datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')))

        t = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        # reward_matrix.dump(self.results_path + 'reward_matrix_new_{}.dat'.format(t))
        np.array(self.avg_rewards).dump(self.results_path + 'average_rewards_new_{}.dat'.format(t))

    def test(self, checkpoint_actor_filename, checkpoint_critic_filename, time_span=10):
        checkpoint_actor_path = self.model_path + checkpoint_actor_filename
        checkpoint_critic_path = self.model_path + checkpoint_critic_filename
        self.agent.get_actor().load_state_dict(torch.load(checkpoint_actor_path))
        self.agent.get_critic().load_state_dict(torch.load(checkpoint_critic_path))
        for t in range(time_span):
            state = self.env.reset(train_mode=False)
            self.score = 0
            #done = False

            while True:
                action = self.agent.choose_action(state, 'test')
                sys.stdout.flush()
                self.env.render()
                state, reward, done, _ = self.env.step(action)
                self.score += np.array(reward)
                if any(done):
                    break

            print('\nFinal score:', self.score)

        self.env.close()

    @staticmethod
    def __set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
