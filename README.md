# SmartCardAI

/!\ Tis document is deprecated !

## Description

SmartCardAI est un projet de module python permettant d'améliorer la création et la gestion d'agents pour les environnements de la bibliothèque RLCard.

Ce module se concentre sur les agents de types Rules, DQN, NFSP et DMC.

Le but est de pouvoir expérimenter de manière simple et sans faire l'impasse sur les différentes méthodes d'apprentissages existantes en les implémentant dans le module.

## Fonctionnalités

* Entraînement d'agents RL (DQN, NFSP)
* Évaluation et comparaison des performances des agents
* Possibilité de jouer contre les agents entraînés
* Support pour l'entraînement sur GPU (CUDA)
* Reprise de l'entraînement à partir de modèles sauvegardés

## Installation

1. Clonez ce dépôt :

   ```shell
   git clone [URL_DU_REPO]
   cd [NOM_DU_DOSSIER]
   ```
2. Créez un environnement virtuel et activez-le (facultatif mais recomandé)  :

   ```shell
   python -m venv .venv
   source .venv/bin/activate  # Sur Windows, utilisez `.venv\Scripts\activate`
   ```
3. Installez les dépendances :

   ```shell
   pip install -r requirements.txt
   ```

## Structure cible du projet

```
SmartCardAI/
│
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py
│   ├── nfsp_agent.py
│   └── dmc_agent.py
│
├── environments/
│   ├── __init__.py
│   └── rlcard_env.py
│
├── experiments/
│   ├── __init__.py
│   └── experiment_manager.py
│
├── utils/
│   ├── __init__.py
│   └── logger.py
│
└── main.py

```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à proposer une pull request.

## Licence

[Apache License Version 2.0](./LICENSE)
