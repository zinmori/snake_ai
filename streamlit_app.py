import streamlit as st
import pygame
import numpy as np
import torch
from dqn_agent import DQNAgent
from game import SnakeGameEnv
import time
import os
import pandas as pd
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Snake DQN Demo",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header avec style
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="color: #2E7D3E; font-size: 3em;">🐍 Snake DQN Demo</h1>
    <p style="font-size: 1.2em; color: #666;">Intelligence Artificielle jouant au Snake avec Deep Q-Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar pour les contrôles
st.sidebar.header("🎮 Contrôles de jeu")
speed = st.sidebar.slider('Vitesse du jeu (frames/sec)', 1, 500, 100)
model_path = 'dqn_snake.pth'

# Initialisation de l'historique des scores
if 'score_history' not in st.session_state:
    st.session_state.score_history = []
    st.session_state.game_number = 0

if os.path.exists(model_path):
    model_exists = True

# Colonnes pour l'interface
col1, col2 = st.columns([2, 1])

with col1:
    if model_exists:
        run_demo = st.button('🚀 Lancer la démo',
                             type="primary", use_container_width=True)
    else:
        st.button('🚀 Lancer la démo', disabled=True, use_container_width=True)

with col2:
    st.subheader("📊 Historique des scores")
    score_placeholder = st.metric("Score actuel", 0)
    chart_placeholder = st.empty()

    # Afficher le graphique initial s'il y a déjà des données
    if st.session_state.score_history:
        df = pd.DataFrame({
            'Partie': range(1, len(st.session_state.score_history) + 1),
            'Score': st.session_state.score_history
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Partie'],
            y=df['Score'],
            mode='lines+markers',
            name='Score',
            line=dict(color='#2E7D3E', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Évolution des scores",
            xaxis_title="Numéro de partie",
            yaxis_title="Score",
            height=300,
            showlegend=False
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

if model_exists and run_demo:
    with st.spinner("🎮 Initialisation du jeu..."):
        env = SnakeGameEnv(speed=speed)
        state_size = len(env.get_state())
        action_size = 3
        agent = DQNAgent(state_size, action_size)
        agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        agent.epsilon = 0.0

    with col1:
        frame_placeholder = st.empty()

    state = env.reset()
    done = False
    score = 0
    moves = 0
    status_text = st.empty()

    while not done:
        action = agent.act(state)
        next_state, reward, done, score = env.step(action)
        state = next_state
        moves += 1

        # Rendu visuel
        arr = np.transpose(pygame.surfarray.array3d(env.display), (1, 0, 2))
        frame_placeholder.image(arr, channels='RGB', width=400)

        # Mise à jour des métriques
        with col2:
            score_placeholder.metric(
                "Score actuel", score, delta=reward if reward > 0 else None)

        time.sleep(1.0/speed)

    # Fin de partie - ajouter le score à l'historique
    st.session_state.game_number += 1
    st.session_state.score_history.append(score)

    # Mettre à jour le graphique final
    with col2:
        df_final = pd.DataFrame({
            'Partie': range(1, len(st.session_state.score_history) + 1),
            'Score': st.session_state.score_history
        })
        fig_final = go.Figure()
        fig_final.add_trace(go.Scatter(
            x=df_final['Partie'],
            y=df_final['Score'],
            mode='lines+markers',
            name='Score',
            line=dict(color='#2E7D3E', width=3),
            marker=dict(size=8)
        ))
        # Mettre en évidence le dernier score
        fig_final.add_trace(go.Scatter(
            x=[st.session_state.game_number],
            y=[score],
            mode='markers',
            name='Dernier score',
            marker=dict(size=15, color='#FF6B6B')
        ))
        fig_final.update_layout(
            title=f"Historique des scores ({len(st.session_state.score_history)} parties)",
            xaxis_title="Numéro de partie",
            yaxis_title="Score",
            height=300,
            showlegend=False
        )
        chart_placeholder.plotly_chart(fig_final, use_container_width=True)

    # Fin de partie
    st.balloons()
    st.success(
        f"🎉 Partie terminée ! Score final : **{score}**")

    # Afficher quelques statistiques
    if len(st.session_state.score_history) > 1:
        avg_score = np.mean(st.session_state.score_history)
        best_score = max(st.session_state.score_history)
        st.info(
            f"📈 Score moyen: {avg_score:.1f} | 🏆 Meilleur score: {best_score}")

# Bouton pour réinitialiser l'historique
if st.sidebar.button("🗑️ Réinitialiser l'historique"):
    st.session_state.score_history = []
    st.session_state.game_number = 0
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    🐍 Snake DQN Demo - Projet d'apprentissage par renforcement avec Deep Q-Learning
</div>
""", unsafe_allow_html=True)
