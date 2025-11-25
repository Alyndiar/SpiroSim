# État actuel

- La modélisation de piste utilise des arcs et segments droits circulaires simplifiés avec un pas constant pour convertir longueur d'arc en "dents équivalentes" (`teeth_equiv`). La géométrie hypo/épitrochoïdale des pièces concaves/convexes n'est pas générée explicitement.
- La pièce Y est traitée comme un simple arc concave de 120° sans placement spécifique des trois branches ni prise en compte des sauts de branche (`*`).
- La progression d'engrènement de la roue est déduite d'un profil de dents interpolé à partir de la polyligne, pas d'une loi d'engrènement exacte par pièce. L'offset initial ne s'applique qu'à la première pièce.

# Changements restants à implémenter

1. Générer la piste modulaire à partir des lois de roulement hypo/épitrochoïdales/racks pour les sections concaves, convexes et droites afin de respecter les nombres de dents fractionnaires et maintenir la phase d'engrènement exacte aux raccords.
2. Implémenter la géométrie complète de la pièce Y (trois arcs concaves à 120°, orientation par branche, gestion de `*`) pour qu'elle s'emboîte avec arcs et barres conformément à la spécification.
3. Calculer la rotation de roue et le tracé (centre, contact, stylo) à partir de la phase d'engrènement exacte par section plutôt que de l'approximation longueur/pas, y compris l'offset de phase initial et les décalages de trous fractionnaires.
4. Gérer la continuité C¹/C² et la fermeture stricte des pistes (ex. trèfle "+C-C-C+C-C-C+C-C-C+C-C-C") en alignant positions, tangentes et phase au bouclage.
5. Étendre l'échantillonnage pour qu'il suive la variation de courbure/phase plutôt qu'un `steps_per_tooth` constant afin de capturer précisément les lobes trochoïdaux.
