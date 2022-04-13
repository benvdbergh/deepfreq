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

# Initialize data
df = importer.get_df("user_data\\data\\binance\\BTC_USDT-1m.json")

# Make environment
env = gym.make('crypto-v0', df=df, frame_bound=(5,863000), window_size=5)
states = env.observation_space.shape[0]
state_size= env.observation_space.shape[1]
actions = env.action_space.n


# Build environment
state = env.reset()
#region Debug plot
# while True: 
#   action = env.action_space.sample()
#   n_state, reward, done, info = env.step(action)
#   if done:
#     print("info", info)
#     break
  
# plt.figure(figsize=(15,6))
# plt.cla()
# env.render_all()
# plt.show()
#endregion

def build_model(states, actions):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(1,states,state_size)))
    # model.add(LSTM(5, activation='relu',input_shape=(1,states)))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
model.summary()


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn
  
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=True, verbose=1, log_interval=10000)