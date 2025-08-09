import streamlit as st
import pygame
import numpy as np
import torch
from dqn_agent import DQNAgent
from game import SnakeGameEnv
import time

st.title('Snake DQN Demo')

speed = st.slider('Vitesse du jeu (frames/sec)', 1, 500, 100)
run_demo = st.button('Lancer la démo')
model_path = 'dqn_snake.pth'

if run_demo:
    env = SnakeGameEnv(speed=speed)
    state_size = len(env.get_state())
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.epsilon = 0.0

    state = env.reset()
    done = False
    score = 0
    frame_placeholder = st.empty()
    score_placeholder = st.empty()
    while not done:
        action = agent.act(state)
        next_state, reward, done, score = env.step(action)
        state = next_state
        # Rendu visuel
        arr = np.transpose(pygame.surfarray.array3d(env.display), (1, 0, 2))
        frame_placeholder.image(arr, channels='RGB')
        score_placeholder.markdown(f"**Score : {score}**")
        time.sleep(1.0/speed)
    st.success(f"Partie terminée ! Score final : {score}")
