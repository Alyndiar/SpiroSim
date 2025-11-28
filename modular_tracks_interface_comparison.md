# Comparaison d'interface : `modular_tracks.py` vs `modular_tracks_2.py`

## Principales différences côté API (vue depuis `SpiroSim.py`)
- **Structure de retour** : la nouvelle version ajoute `segments` dans `TrackBuildResult` et recentre/retourne la piste, tout en conservant `.points`, `.total_length`, `.total_teeth` et `.offset_teeth`. 【F:modular_tracks_2.py†L57-L71】
- **Parsing plus permissif mais aussi plus limité** : la notation est nettoyée (`replace(" ", "")`) et les caractères inconnus sont ignorés au lieu de lever `ValueError`; les pièces spéciales `Y`, `Z` et `*` lèvent désormais `NotImplementedError` lorsqu'elles arrivent à la construction, alors qu'elles étaient simplement ignorées ou non implémentées silencieusement avant. 【F:modular_tracks_2.py†L151-L205】【F:modular_tracks_2.py†L233-L279】【F:modular_tracks.py†L400-L434】
- **Construction de piste** : `_build_segments_from_parsed` produit des segments analytiques et applique l'offset uniquement sur la première pièce en rotation/translation, mais l'appel direct `build_track_from_notation` ne renvoie plus de polyline détaillée contrôlée par `steps_per_tooth` (paramètre conservé mais ignoré). 【F:modular_tracks_2.py†L233-L279】【F:modular_tracks_2.py†L551-L569】【F:modular_tracks.py†L500-L535】
- **Outils d'interpolation** : l'ancienne polyline exposait `_precompute_length_and_tangent` et `_interpolate_on_track` utilisés dans `SpiroSim.py` (dessin des dents, suivi de tangente). Ils n'existent plus; la nouvelle version propose seulement `_interpolate_on_segments` qui travaille sur les segments géométriques. 【F:modular_tracks.py†L538-L606】【F:modular_tracks_2.py†L571-L650】
- **Génération de courbe** : `generate_track_base_points` ne gère plus `output_mode` (`stylo`/`contact`/`centre`) ni `wheel_phase_teeth`; l'appel calcule uniquement la trajectoire stylo. 【F:modular_tracks.py†L612-L706】【F:modular_tracks_2.py†L653-L759】

## Fonctions/comportements manquants dans `modular_tracks_2.py`
- **Compatibilité avec les usages internes de `SpiroSim.py`** : le code de rendu des pistes appelle `_precompute_length_and_tangent` et `_interpolate_on_track` pour dessiner les dents et la normale finale. Ces helpers doivent être réintroduits (en s'appuyant sur `segments` ou en générant une polyline) ou `SpiroSim.py` doit être adapté pour consommer `_interpolate_on_segments`. 【F:SpiroSim.py†L925-L950】【F:modular_tracks_2.py†L571-L650】
- **Modes de sortie et phase roue** : l'ancienne génération permettait `output_mode="contact"/"centre"` et `wheel_phase_teeth` pour décaler la roue; ces options manquent. Ajouter des paramètres optionnels (avec valeurs par défaut pour compatibilité) et utiliser `mode` pour renvoyer le point demandé rétablirait ces capacités. 【F:modular_tracks.py†L612-L706】
- **Pièces spéciales et branches** : les éléments `Y`, `Z` et `*` lèvent désormais des erreurs. Pour retrouver la tolérance précédente, il faut soit implémenter leur géométrie, soit les ignorer comme avant (au minimum ne pas lever pendant la construction). 【F:modular_tracks_2.py†L233-L279】【F:modular_tracks.py†L400-L434】

## Comment ajouter les parties manquantes tout en restant compatible
1. **Restaurer les helpers d'interpolation** :
   - Implémenter `_precompute_length_and_tangent` et `_interpolate_on_track` comme wrappers autour de la polyline `TrackBuildResult.points` (ou en échantillonnant `segments`) afin que les appels existants continuent de fonctionner sans modifier tout le code UI.
2. **Réintroduire `output_mode` et `wheel_phase_teeth`** :
   - Étendre `generate_track_base_points` avec ces paramètres facultatifs; appliquer `wheel_phase_teeth` dans le calcul de `phi` et renvoyer soit le point stylo, soit le point de contact, soit le centre, en suivant la logique de l'ancienne version.
3. **Assouplir/implémenter les pièces spéciales** :
   - Traiter `Y`, `Z` et `branch (*)` en ignorant leur géométrie (comme avant) ou en fournissant une approximation, mais sans lever d'exception pour les notations déjà utilisées par l'application.

## Ajustements nécessaires dans `SpiroSim.py` pour utiliser la nouvelle version
- **Dessin des dents de piste** : remplacer l'usage de `_precompute_length_and_tangent` / `_interpolate_on_track` par `_interpolate_on_segments` et les segments renvoyés dans `TrackBuildResult`, ou s'appuyer sur les helpers réintroduits. 【F:SpiroSim.py†L925-L950】【F:modular_tracks_2.py†L571-L650】
- **Appels de génération de courbe** : les appels existants passent `output_mode` et `wheel_phase_teeth`; il faut soit remettre ces paramètres dans `modular_tracks_2.generate_track_base_points`, soit retirer/adapter ces arguments côté `SpiroSim.py` en préservant le même comportement (e.g. calculer la phase et filtrer le mode au niveau appelant). 【F:modular_tracks.py†L612-L706】【F:SpiroSim.py†L560-L640】
- **Gestion des notations avec `*`/`Y`/`Z`** : si l'éditeur autorise ces pièces, `SpiroSim.py` devra soit filtrer ces éléments avant d'appeler la nouvelle version, soit attendre que leur prise en charge soit restaurée pour éviter des `NotImplementedError`. 【F:modular_tracks_2.py†L233-L279】【F:SpiroSim.py†L560-L640】
