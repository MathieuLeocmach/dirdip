function liste_cotes = trouve_cotes(liste_vertex)

% Détermination des côtés normaux : un côté est une paire de vertex. En parcourant
% la liste des vertex, à chaque paire de bulles trouvée, on associe un côté

% On construit les paires de bulles en contact
paires = [liste_vertex(:,1), liste_vertex(:,2), liste_vertex(:,3);...
    liste_vertex(:,1), liste_vertex(:,2), liste_vertex(:,4);...
    liste_vertex(:,1), liste_vertex(:,2), liste_vertex(:,5);...
    liste_vertex(:,1), liste_vertex(:,3), liste_vertex(:,4);...
    liste_vertex(:,1), liste_vertex(:,3), liste_vertex(:,5);...
    liste_vertex(:,1), liste_vertex(:,4) liste_vertex(:,5)];
% On élimine les paires où il y a un zéro
paires = paires(paires(:,2).*paires(:,3)~=0,:);
paires_sans_coord = paires(:,2:3);
% On identifie les paires uniques
[singletons,m,n] = unique(paires(:,2:3),'rows');
indices_doublons = setdiff(1:size(paires,1),m);
% Un côté est l'ensemble de deux paires
cotes = paires(ismember(paires_sans_coord,paires_sans_coord(indices_doublons,:),'rows'),:);
liste_cotes = sortrows(cotes,[2 3]);