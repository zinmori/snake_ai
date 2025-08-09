import torch
import argparse
from dqn_agent import DQNAgent
from game import SnakeGameEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='dqn_snake.pth', help='Chemin du modèle à charger')
    parser.add_argument('--visual', action='store_true',
                        help='Afficher le jeu (sinon rapide sans affichage)')
    parser.add_argument('--speed', type=int, default=10,
                        help='Vitesse du jeu (frames/sec)')
    args = parser.parse_args()

    env = SnakeGameEnv(speed=args.speed)
    if not args.visual:
        def no_update_ui(self):
            pass
        env._update_ui = no_update_ui.__get__(env)

    state_size = len(env.get_state())
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(
        args.model_path, map_location='cpu'))
    agent.epsilon = 0.0  # exploitation only

    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, score = env.step(action)
        state = next_state
    print(f"Score de l'agent: {score}")
