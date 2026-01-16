# SpiroSim

**Langues :** [English](../../README.md) | Français | [Toutes les langues](../../LANGUAGES.md)

Un simulateur/banc d’essai pour des dessins inspirés du Spirographe. Plusieurs couches d’engrenages, plusieurs tracés par couche, pistes personnalisées façon « Super Spirograph ». Tailles des engrenages, décalages de tracé et couleurs configurables. Export en JSON, PNG et SVG.

## Localisation

SpiroSim intègre un système de localisation pour les textes de l’interface. Utilisez
**Options → Langue** pour changer de langue et consultez
[`localisation.md`](../../localisation.md) pour le format des fichiers et les instructions
d’ajout d’une nouvelle langue au programme et au dépôt.

## Installation

1. Assurez-vous d’avoir Python 3.10+ installé.
2. (Optionnel) Créez et activez un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   Ou avec Conda/Miniconda :
   ```bash
   conda create -n spirosim python=3.10
   conda activate spirosim
   ```
3. Installez la dépendance graphique :
   ```bash
   python -m pip install PySide6
   ```

## Utilisation

Lancez l’application principale :

```bash
python SpiroSim.py
```

## Vue d’ensemble de l’interface

La fenêtre principale affiche le dessin et propose les menus et dialogues suivants.

### Menu Fichier

- **Charger paramètres (JSON)** : importer les couches, tracés, couleurs et paramètres de piste.
- **Sauvegarder paramètres (JSON)** : exporter la configuration actuelle.
- **Exporter en SVG** : sauvegarder un export vectoriel des couches visibles.
- **Exporter en PNG haute résolution** : sauvegarder un export raster à une résolution spécifiée.
- **Quitter** : fermer l’application.

### Menu Couches

- **Gérer les couches et les tracés** : ouvrir le gestionnaire de couches/tracés pour composer le dessin.

### Menu Options

- **Couleur de fond** : définir la couleur d’arrière-plan (nom CSS4, hex ou tuple HSL).
- **Taille du canevas et précision** : régler largeur/hauteur et points par tracé.
- **Langue** : basculer l’interface en français ou en anglais.

### Menu Régénérer

- **Animation** : activer/désactiver les contrôles d’animation sous le canevas.
- **Afficher la piste** : activer/désactiver l’affichage des lignes centrales de piste.
- **Régénérer le dessin** : recalculer et rafraîchir le rendu.

### Menu Aide

- **Manuel** : ouvrir le README localisé correspondant à la langue active.
- **À propos** : afficher la version et la licence.

### Contrôles d’animation

Quand l’animation est activée, des contrôles sous le canevas permettent de démarrer,
mettre en pause, réinitialiser l’aperçu et régler la vitesse (points par seconde,
avec un mode instantané).

## Couches et tracés

Une **couche** représente une configuration d’engrenages et peut contenir plusieurs
**tracés**. Chaque tracé est dessiné à partir du même mouvement d’engrenages, avec
son propre trou de stylo, déphasage, couleur et épaisseur.

### Paramètres d’une couche

Dans l’éditeur de couche, vous pouvez régler :

- **Nom** : libellé utilisé dans les exports et la liste des couches.
- **Visible** : afficher/masquer la couche.
- **Zoom de couche** : met à l’échelle tous les tracés de la couche.
- **Translation/rotation de couche** : déplacer et faire pivoter toute la couche.
- **Nombre d’engrenages (2 ou 3)** : choisir un système à 2 ou 3 engrenages.

### Paramètres des engrenages

Chaque couche comporte 2 ou 3 engrenages. L’engrenage 1 est stationnaire (anneau
ou piste modulaire), et l’engrenage 2 (et 3 éventuel) est mobile. Pour chaque
engrenage, vous pouvez configurer :

- **Nom** : libellé affiché dans le gestionnaire.
- **Type** :
  - `anneau`, `roue`, `dsl`
  - `modulaire` (piste modulaire de base, uniquement pour l’engrenage 1)
- **Taille (roue / int. anneau)** : taille de la roue ou de l’anneau intérieur.
- **Taille ext. (anneau)** : taille de l’anneau extérieur.
- **Relation** :
  - `stationnaire` : uniquement pour l’engrenage 1 (fixe).
  - `dedans` : la roue roule à l’intérieur de l’anneau (hypotrochoïde).
  - `dehors` : la roue roule à l’extérieur de l’anneau (épitrochoïde).
- **Piste modulaire (notation)** : visible uniquement pour l’engrenage 1 en type
  `modulaire`. Utilise la notation décrite ci-dessous.
  Utilisez le bouton **…** pour ouvrir l’éditeur de piste modulaire.

### Paramètres d’un tracé

Chaque tracé définit la position du stylo sur la roue mobile :

- **Nom** : libellé affiché dans le gestionnaire et dans les exports.
- **Décalage du trou** : décalage radial du trou sur l’engrenage (distance au centre).
- **Décalage de phase (unités de piste)** : déphasage appliqué à la position du stylo.
- **Couleur** : nom CSS4, hex `#RRGGBB` ou tuple HSL `(H, S, L)`.
- **Largeur de trait** : épaisseur dans l’aperçu et les exports.
- **Zoom du tracé** : échelle appliquée uniquement à ce tracé (multipliée par le zoom de couche).
- **Translation/rotation du tracé** : déplacer et faire pivoter le tracé par rapport à la couche.

### Test de piste

Si l’engrenage 1 est une piste modulaire avec une notation valide, le gestionnaire
active **Test du tracé** pour prévisualiser la piste et le mouvement de la roue.

### Actions du gestionnaire

Le gestionnaire permet d’ajouter, modifier, réordonner, activer/désactiver et supprimer
des couches ou des tracés, ainsi que d’activer/désactiver tous les tracés d’une couche.

## Notation des pistes modulaires

Les pistes sont définies par une notation algébrique compacte composée de blocs
`lettre + nombre` séparés par les opérateurs `+`, `-` ou `*`. Les espaces sont
ignorés et la notation est insensible à la casse.

### Opérateurs

- `+` / `-` : fixe le sens de rotation (gauche/droite) de la pièce suivante.
- `*` : passe à la branche ouverte suivante créée par `y` ou `b`.

Un `+` ou `-` initial définit le sens par défaut de la première pièce. Un `*`
initial saute directement à la première branche ouverte.

### Pièces

- `aNN` : arc de `NN` degrés. Le signe (`+`/`-`) indique le sens.
- `dNN` : segment droit de `NN` unités.
- `b` : bout arrondi (demi-cercle) reliant les deux côtés de la piste.
- `y` : jonction triangulaire composée de trois arcs de 120° espacés de la
  largeur de la piste.
- `nNN` : décalage d’origine en unités, dans le sens du signe courant.
- `oNN` : décalage angulaire d’origine en degrés, avec la même convention de signe.

`NN` peut être entier ou décimal.

### Exemple

```
+a90-d40+b*a90
```

Construit un arc de 90° à gauche, une droite de 40 unités, un bout arrondi, puis
poursuit sur la branche suivante avec un autre arc de 90°.
