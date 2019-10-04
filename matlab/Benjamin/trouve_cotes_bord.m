function liste_cotes_bord = trouve_cotes_bord(liste_vertex)

% Identification des côtés de bord comme reliant les vertex de bord à deux bulles
% voisines
vertex_de_bord = liste_vertex(find(liste_vertex(:,4)==0),1:3);
bulles_de_bord = [vertex_de_bord(:,1:2); vertex_de_bord(:,[1 3])];
bulles_de_bord_tri = sortrows(bulles_de_bord(:,[2 1]));
bulles_de_bord_tri = bulles_de_bord_tri(:,[2 1]);
numeros_bulles_de_bord = bulles_de_bord_tri(:,2);
[singletons_bord,m_bord,n_bord] = unique(numeros_bulles_de_bord);
indices_doublons_bord = setdiff(1:length(numeros_bulles_de_bord),m_bord);
numeros_bulles_de_bord_renv = numeros_bulles_de_bord(length(numeros_bulles_de_bord):-1:1);
[singletons_bord_renv,m_bord_renv,n_bord_renv] = unique(numeros_bulles_de_bord_renv);
m_bord_renv = length(numeros_bulles_de_bord) + 1 - m_bord_renv;
indices_singletons_bord = setdiff(m_bord_renv,indices_doublons_bord);
liste_cotes_bord = bulles_de_bord_tri(setdiff(1:length(numeros_bulles_de_bord),indices_singletons_bord),:);