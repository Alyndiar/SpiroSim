# Comparaison d'interface : ancien module modulaire vs `modular_tracks_2.py`

## Principales différences côté API (vue depuis `SpiroSim.py`)
- **Structure de retour** : la nouvelle version ajoute `segments` dans `TrackBuildResult` et recentre/retourne la piste, tout en conservant `.points`, `.total_length`, `.total_teeth` et `.offset_teeth`. 【F:modular_tracks_2.py†L57-L71】
- **Parsing plus permissif mais aussi plus limité** : la notation est nettoyée (`replace(" ", "")`) et les caractères inconnus sont ignorés au lieu de lever `ValueError`; les pièces spéciales `Y`, `Z` et `*` sont pour l'instant ignorées silencieusement lors de la construction (aucune exception n'est levée). 【F:modular_tracks_2.py†L151-L205】【F:modular_tracks_2.py†L233-L279】
- **Construction de piste** : `_build_segments_from_parsed` produit des segments analytiques et applique l'offset uniquement sur la première pièce en rotation/translation, mais l'appel direct `build_track_from_notation` ne renvoie plus de polyline détaillée contrôlée par `steps_per_tooth` (paramètre conservé mais ignoré). 【F:modular_tracks_2.py†L233-L279】【F:modular_tracks_2.py†L551-L569】
- **Outils d'interpolation** : l'ancien module exposait `_precompute_length_and_tangent` et `_interpolate_on_track` utilisés dans `SpiroSim.py` (dessin des dents, suivi de tangente). Ils n'existent plus; la nouvelle version propose seulement `_interpolate_on_segments` qui travaille sur les segments géométriques. 【F:modular_tracks_2.py†L571-L650】
- **Génération de courbe** : `generate_track_base_points` accepte `output_mode` (`stylo`/`contact`/`centre`) ainsi que `wheel_phase_teeth` pour décaler la roue; la trajectoire retournée dépend du mode demandé. 【F:modular_tracks_2.py†L1119-L1171】

## Fonctions/comportements manquants dans `modular_tracks_2.py`
- **Compatibilité avec les usages internes de `SpiroSim.py`** : le rendu consomme désormais les segments via `_interpolate_on_segments` et `drawing.build_track_teeth_markers_from_segments`; toute évolution doit conserver cette API segmentée pour les dents et marqueurs. 【F:SpiroSim.py†L2596-L2655】【F:drawing.py†L154-L193】【F:modular_tracks_2.py†L571-L650】
- **Modes de sortie et phase roue** : la nouvelle version expose `output_mode` et `wheel_phase_teeth` directement dans `generate_track_base_points` et les applique dans le calcul de la trajectoire et des marqueurs. Aucun écart connu sur ce point.
- **Pièces spéciales et branches** : les éléments `Y`, `Z` et `*` sont ignorés pendant la construction (aucune erreur levée) en attendant une implémentation géométrique future. 【F:modular_tracks_2.py†L233-L279】

## Pistes d'évolution restantes
1. **Restaurer les helpers d'interpolation** :
   - Implémenter `_precompute_length_and_tangent` et `_interpolate_on_track` comme wrappers autour de la polyline `TrackBuildResult.points` (ou en échantillonnant `segments`) afin que les appels existants continuent de fonctionner sans modifier tout le code UI.
2. **Assouplir/implémenter les pièces spéciales** :
   - Traiter `Y`, `Z` et `branch (*)` en ignorant leur géométrie (comme avant) ou en fournissant une approximation, mais sans lever d'exception pour les notations déjà utilisées par l'application.

## Ajustements nécessaires dans `SpiroSim.py` pour utiliser la nouvelle version
- **Dessin des dents de piste** : le code s'appuie désormais sur `_interpolate_on_segments` via `drawing.build_track_teeth_markers_from_segments`; les helpers doivent rester cohérents avec le modèle segmenté. 【F:SpiroSim.py†L2596-L2655】【F:drawing.py†L154-L193】【F:modular_tracks_2.py†L571-L650】
- **Appels de génération de courbe** : les appels existants peuvent passer `output_mode` et `wheel_phase_teeth` directement à `modular_tracks_2.generate_track_base_points`, qui gère ces paramètres. Aucun contournement n'est requis côté appelant. 【F:modular_tracks_2.py†L1119-L1171】
- **Gestion des notations avec `*`/`Y`/`Z`** : si l'éditeur autorise ces pièces, `SpiroSim.py` devra soit filtrer ces éléments avant d'appeler la nouvelle version, soit attendre que leur prise en charge soit restaurée pour éviter des `NotImplementedError`. 【F:modular_tracks_2.py†L233-L279】【F:SpiroSim.py†L579-L651】
