# Projet_NLP_M1

> Un chatbot intelligent capable de détecter les émotions dans un message utilisateur et de générer des réponses adaptées émotionnellement.

Objectif

Développer un système de génération de texte basé sur GPT, capable :
- de détecter les émotions dans un message utilisateur,
- et de répondre avec un ton émotionnel approprié (joie, tristesse, colère, etc.).

Fonctionnalités

- Détection automatique des émotions** dans un texte utilisateur (basée sur BERT/DistilBERT).
- Génération de texte conditionnée par l’émotion (via GPT-2).
- Interface simple en Streamlit pour tester le système.

Structure du projet

bash
.
├── data/               # Données d'entraînement (ex: GoEmotions)
├── models/             # Modèles entraînés
├── notebooks/          # Notebooks d'entraînement/test
├── scripts/            # Scripts Python principaux
├── app/                # Interface utilisateur (Streamlit)
├── README.md
└── requirements.txt
