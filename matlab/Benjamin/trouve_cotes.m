function liste_cotes = trouve_cotes(liste_vertex)

% D�termination des c�t�s normaux : un c�t� est une paire de vertex. En parcourant
% la liste des vertex, � chaque paire de bulles trouv�e, on associe un c�t�

% On construit les paires de bulles en contact
paires = [liste_vertex(:,1), liste_vertex(:,2), liste_vertex(:,3);...
    liste_vertex(:,1), liste_vertex(:,2), liste_vertex(:,4);...
    liste_vertex(:,1), liste_vertex(:,2), liste_vertex(:,5);...
    liste_vertex(:,1), liste_vertex(:,3), liste_vertex(:,4);...
    liste_vertex(:,1), liste_vertex(:,3), liste_vertex(:,5);...
    liste_vertex(:,1), liste_vertex(:,4) liste_vertex(:,5)];
% On �limine les paires o� il y a un z�ro
paires = paires(paires(:,2).*paires(:,3)~=0,:);
paires_sans_coord = paires(:,2:3);
% On identifie les paires uniques
[singletons,m,n] = unique(paires(:,2:3),'rows');
indices_doublons = setdiff(1:size(paires,1),m);
% Un c�t� est l'ensemble de deux paires
cotes = paires(ismember(paires_sans_coord,paires_sans_coord(indices_doublons,:),'rows'),:);
liste_cotes = sortrows(cotes,[2 3]);