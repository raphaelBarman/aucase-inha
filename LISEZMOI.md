# Description générale

AuCaSe est un projet réalisé par Raphaël Barman à l'Institut national d'histoire de l'art (INHA) lors d'un stage dans le cadre de sa maîtrise en Humanités Digitales à l'École polytechnique fédérale de Lausanne (EPFL).

L'objectif du projet était d'utiliser la numérisation des catalogues de ventes aux enchères appartenant à l'INHA afin de créer une base de données des ventes d'objets avec leurs métadonnées, ceci à travers un processus de segmentation des images par apprentissage profond.

Le projet a réussi à extraire de 1911 catalogues de vente aux enchères publiés par la maison Drouot entre le 1er janvier 1939 et le 31 décembre 1945 plus de 300'000 objets réalisés par plus de 5'000 artistes mise en vente dans des enchères supervisées par plus de 200 commissaires-priseurs et experts.

Ces données pourront être explorées et téléchargées gratuitement à l'avenir.

# Structure du dépôt

Le point d'entrée principal de ce dépôt est le fichier "Notebook annoté" qui contient toutes les étapes de la pipeline depuis le téléchargement des numérisations jusqu'à l'importation des données dans une base sql.

Techniquement, en modifiant le fichier config.toml et en exécutant le fichier main.py, tout la pipeline devrait fonctionner. Cependant, il reste encore quelques étapes à passer avant de l'exécuter, comme l'entraînement ou l'obtention de différents modèles. En général, c'est une bonne idée de commenter la plupart des étapes du fichier main.py et de les exécuter une par une. L'ensemble de la pipeline peut prendre des jours, voire une semaine ou deux (sans GPU).

# Description du schéma SQL
La description du schéma SQL est [disponible ici](https://github.com/sriak/aucase-inha/blob/master/description_schema_sql.md)
