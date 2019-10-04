function liste_vertex = trouve_vertex(image)

% Explication : 12/03/2013

hauteur_image = size(image,1);
largeur_image = size(image,2);
i_noir = find(image(2:hauteur_image-1,2:largeur_image-1) == 0);
% relation entre i_noir et l'indice linéaire "réel"
k = i_noir + hauteur_image + 2*floor((i_noir-1)/(hauteur_image-2)) + 1;
nord = k-1;
sud = k+1;
ouest = k - hauteur_image;
est = k + hauteur_image;
nord_ouest = k - hauteur_image - 1;
nord_est = k + hauteur_image - 1;
sud_ouest = k - hauteur_image + 1;
sud_est = k + hauteur_image + 1;
voisins8 = [image(nord) image(nord_est) image(est) image(sud_est)...
    image(sud) image(sud_ouest) image(ouest) image(nord_ouest)];

zeros_voisin = sign(voisins8);
c = zeros_voisin(:,1:8) + zeros_voisin(:,[2:8, 1]) + zeros_voisin(:,[3:8, 1 2]) + zeros_voisin(:,[4:8, 1:3]);
est_pixel_bord = (min(c,[],2)==0);

nb_voisines = sum(diff(sort([zeros(length(k),1) voisins8]')) ~= 0) + est_pixel_bord';
a_garder = (nb_voisines >= 3);
liste_vertex = k(a_garder);