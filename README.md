# Snake DQN

Ce projet implémente le jeu Snake avec un agent Deep Q-Learning (DQN) en Python, utilisant Pygame et PyTorch. Vous pouvez entraîner l'agent, le faire jouer automatiquement, et visualiser une démo dans une application Streamlit.

## Prérequis

-   Python 3.8+
-   `pip install -r requirements.txt`

## Entraîner l'agent DQN

```bash
python train_dqn.py --speed 200 --visual
```

-   `--speed` : vitesse du jeu (plus grand = plus rapide)
-   `--visual` : affiche le jeu pendant l'entraînement (optionnel)

## Faire jouer l'agent entraîné

```bash
python play_dqn.py --speed 100 --visual
```

-   `--speed` : vitesse du jeu
-   `--visual` : affiche le jeu (optionnel)

## Démo Streamlit

```bash
streamlit run streamlit_app.py
```

Vous pouvez choisir la vitesse et le modèle à charger dans l'interface web.

## Fichiers principaux

-   `game.py` : environnement Snake compatible DQN
-   `dqn_agent.py` : agent DQN (PyTorch)
-   `train_dqn.py` : script d'entraînement
-   `play_dqn.py` : script de test/jeu
-   `streamlit_app.py` : démo web

## Astuces

-   Pour accélérer l'entraînement, utilisez un grand `--speed` et désactivez `--visual`.
-   Le modèle est sauvegardé automatiquement à la fin de l'entraînement.

## Licence

MIT
