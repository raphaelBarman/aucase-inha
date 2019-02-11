Ceci est une description du schéma sql utilisé pour la base de donnée.

## Table "actor"
Ce tableau contient tous les différents acteurs qui apparaissent dans les métadonnées de la bibliothèque numérique de l'INHA.

Il comporte les champs suivants :
- actor_id : id de l'acteur
- prénom : prénom de l'acteur
- nom_famille : nom de famille de l'acteur
- rôle : rôle de l'acteur, un parmi "Commisaire-priseur", "Expert", "Commisaire-priseur, Expert", "Éditeur".
## Table "sale"
Cette table contient toutes les métadonnées des catalogues de vente aux enchères.

Elle comporte les champs suivants :
- sale_id : id de la vente
- date : date de la vente
- cote_inha : numéro de classement à l'INHA
- url_inha : url de la numérisation du catalogue

## Table "actor_sale"
Ce tableau fait le lien entre les acteurs et les différentes ventes auxquelles ils ont participé.

Elle comporte les champs suivants :
- actor_id : id de l'acteur
- sale_id : id de la vente
## Table "section"
Cette table contient toutes les sections qui ont été détectées par la segmentation.

Les sections caractérisent les objet et autres sections qui les suivent, par exemple en spécifiant
le type de l'objet ("peinture") ou l'auteur de l'objet ("Picasso").

Une section peut être placée sous une autre section, dans une relation dite parent-enfant, par exemple les sections d'auteur sont
la plupart du temps placé dans une section de catégorie (par exemple "Picasso" est placé dans la catégorie "Peinture").

Ceci est représenté dans la table par une référence interne (à la table) à l'autre section qui est le parent de la section en cours.

Une section n'est pas identifiée directement par un identifiant, mais plutôt par un triplet, qui est composé de la vente dans laquelle elle apparaît, de la page et du numéro de l'entité dans la page. Par exemple, la section avec l'identificateur (1787, 5, 4) est la cinquième entité de la page six (il y a un décalage car on commence à compter à zéro) du numéro de vente 1787.

Elle comporte les champs suivants :
- sale_id : id de la vente dans laquelle la section apparaît
- page : page dans laquelle la section apparaît
- entité : numéro d'entité de la section dans la page
- parent_section_sale : id de la vente de la section parent
- parent_section_page : page de la section parent
- parent_section_entity : numéro d'entité de la section parent
- classe : classe de la section, une parmi "auteur", "catégorie", "ecole", "classe".
- texte : texte de la section
- bbox : (xmin, ymin, ymin, xmax, ymax) de la boîte de délimitation de la section
- inha_url : url de la page de la numérisation
- iiif_url : url iiif de base de la page, peut être utilisé avec le champ bbox pour obtenir un recadrage de la section actuelle
## Table "object"
Cette table contient tous les objets qui ont été détectés par la segmentation.

Elle utilise le même principe que dans la table de sections pour l'identifiant et les sections parent.

Elle comporte les champs suivants :
- sale_id : id de la vente dans laquelle l'objet apparaît
- page : page dans laquelle l'objet apparaît
- entity : numéro d'entité de l'objet dans la page
- parent_section_sale : id de la vente de la section parent
- parent_section_page : page de la section parent
- parent_section_entity : numéro d'entité de la section parent
- num_ref : numéro de référence corrigé de l'objet, pas 100% précis
- texte : texte de l'objet
- bbox : (xmin, ymin, ymin, xmax, ymax) de la boîte de délimitation de l'objet
- inha_url : url de la page de la numérisation
- iiif_url : url iiif de base de la page, peut être utilisé avec le champ bbox pour obtenir un recadrage de l'objet réel
