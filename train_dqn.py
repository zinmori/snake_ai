import torch

import torch
import numpy as np
import argparse
from dqn_agent import DQNAgent
from game import SnakeGameEnv

EPISODES = 500
BATCH_SIZE = 1000
SCORES = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual', action='store_true',
                        help='Afficher le jeu pendant l\'entrainement')
    parser.add_argument('--model-path', type=str,
                        default='dqn_snake.pth', help='Chemin de sauvegarde du modèle')
    parser.add_argument('--speed', type=int, default=10,
                        help='Vitesse du jeu (frames/sec)')
    args = parser.parse_args()

    env = SnakeGameEnv(speed=args.speed)
    if not args.visual:
        # Monkey patch pour désactiver l'affichage
        def no_update_ui(self):
            pass
        env._update_ui = no_update_ui.__get__(env)

    state_size = len(env.get_state())
    action_size = 3  # [straight, right, left]
    agent = DQNAgent(state_size, action_size)
    max_score = 0
    for e in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, score = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                SCORES.append(score)
                if score > max_score:
                    max_score = score
                print(
                    f"Episode {e+1}/{EPISODES} - Score: {score} - Max: {max_score} - Epsilon: {agent.epsilon:.2f}")
                break
        agent.replay(BATCH_SIZE)
    print("Training finished.")
    print(f"Best score: {max(SCORES)}")
    # Sauvegarde du modèle
    torch.save(agent.model.state_dict(), args.model_path)
    print(f"Modèle sauvegardé dans {args.model_path}")
