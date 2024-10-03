# Agent d'apprentissage par renforcement pour le jeu UNO

Ce projet implémente et compare différents agents d'intelligence artificielle pour jouer au jeu de cartes UNO en utilisant des techniques d'apprentissage par renforcement.

**Remarque :** *Ce projet fonctionne aussi pour les autre environnement de rlcard dans __la plus part des cas__.*

## Fonctionnalités

* Entraînement d'agents RL (DQN, NFSP) pour jouer à UNO
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

## Utilisation

### Entraînement d'un agent

Pour entraîner un agent, utilisez la commande suivante :

```shell
python -m agent_manager --env_type uno --algorithm dqn --num_episodes 5000
```

Options disponibles :

* `--env_type` : Type d'environnement (par défaut : 'uno')
* `--algorithm` : Algorithme à utiliser ('dqn' ou 'nfsp')
* `--num_episodes` : Nombre d'épisodes d'entraînement
* `--cuda` : Spécifiez le GPU à utiliser (laissez vide pour CPU)
* `--resume_training` : Reprendre l'entraînement à partir d'un modèle existant
* `--train_against_self` : Entraîner l'agent contre une copie de lui-même

### Évaluation des agents

Pour évaluer les performances des agents, utilisez le script dans `test.ipynb`.

## Structure du projet

* `agent_manager/` : Le module python contenant les implémentations des agents et les fonctions d'entraînement
* `experiments/` : Dossier de sortie par déffaut pour les logs et les modèles entraînés
* `test.ipynb` : Notebook Jupyter pour tester et comparer les agents

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à proposer une pull request.

## Licence

[Apache License Version 2.0](./LICENSE)
