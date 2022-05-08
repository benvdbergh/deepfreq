#region IMPORTS
# Gym stuff
import gym
import gym_anytrading
import random

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.optimizers import Adam

# Processing libraries
from matplotlib import pyplot as plt
from data_manager import importer

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
#endregion

# Initialize data
df = importer.get_df("user_data\\data\\binance\\BTC_USDT-1m.json")

# Make environment
env = gym.make('crypto-v0', df=df, frame_bound=(11,df.shape[0]), window_size=10)
num_timesteps = env.observation_space.shape[0]
state_size= env.observation_space.shape[1]
actions = env.action_space.n
#boopidy boop boop booooop

# Build environment
state = env.reset()

def build_model(num_timesteps, actions):
    model = tf.keras.Sequential()
    #model.add(Flatten(input_shape=(1,num_timesteps,state_size)))
    model.add(LSTM(5, activation='relu',input_shape=(num_timesteps, state_size)))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(num_timesteps, actions)
model.summary()


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=num_timesteps)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=40, target_model_update=1e-2)
    return dqn
  
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, nb_max_episode_steps=2000, verbose=1, log_interval=2000)