% load 'C:\Users\bdollet\Desktop\CNRS\Encadrement\Stages\2015 M1 Bocher\20150504_SDS_100mLmin_2mm_pompe_1\bulles_toutes_images.mat';

images_analysees = 5:2:1100;
i_boite = 32; taille_boite_i = 32;
j_boite = 32; taille_boite_j = 32;

% Facteurs de conversion en unités réelles (mm et s)
mm_par_pix = 100/434;
fps = 50;

fichier_vitesse = [];
vitesse_moyenne_par_image = zeros(length(images_analysees),2);
vitesse_moyenne_par_image(:,1) = images_analysees';

for num_image = images_analysees
    bulles_une_image = bulles_toutes_images{num_image}(:);
    nombre_bulles = size(bulles_une_image,1);
    vitesse_une_image = zeros(nombre_bulles,8);
    for j = 1:nombre_bulles
        if length(bulles_une_image(j).Cotes > 0)
            contraintes = [sum(bulles_une_image(j).Cotes(:,1).^2./sqrt(bulles_une_image(j).Cotes(:,1).^2 + bulles_une_image(j).Cotes(:,2).^2))...
                sum(bulles_une_image(j).Cotes(:,1).*bulles_une_image(j).Cotes(:,2)./sqrt(bulles_une_image(j).Cotes(:,1).^2 + bulles_une_image(j).Cotes(:,2).^2))...
                sum(bulles_une_image(j).Cotes(:,2).^2./sqrt(bulles_une_image(j).Cotes(:,1).^2 + bulles_une_image(j).Cotes(:,2).^2))];
        else contraintes = [0 0 0];
        end;
        vitesse_une_image(j,:) = [bulles_une_image(j).Centroid(1)...
            bulles_une_image(j).Centroid(2)...
            bulles_une_image(j).Deplacement(1)...
            bulles_une_image(j).Deplacement(2)...
            bulles_une_image(j).Area...
            contraintes];
    end;
    fichier_vitesse = [fichier_vitesse; vitesse_une_image];
    vitesse_moyenne_par_image(find(vitesse_moyenne_par_image(:,1) == num_image),2) = mm_par_pix * fps * mean(vitesse_une_image(:,4));
end;

for i = 1:i_boite
    for j = 1:j_boite
        dans_boite_ij = find(ceil(fichier_vitesse(:,1)/taille_boite_i) == i...
            & ceil(fichier_vitesse(:,2)/taille_boite_j) == j);
        v_i(i,j) = mean(fichier_vitesse(dans_boite_ij,3));
        v_j(i,j) = mean(fichier_vitesse(dans_boite_ij,4));
        aire(i,j) = mean(fichier_vitesse(dans_boite_ij,5));
        sigma_ii(i,j) = mean(fichier_vitesse(dans_boite_ij,6));
        sigma_ij(i,j) = mean(fichier_vitesse(dans_boite_ij,7));
        sigma_jj(i,j) = mean(fichier_vitesse(dans_boite_ij,8));
%         contrainte_xy_moyenne(j) = .02189 * mean(fichier_vitesse(dans_boite_ij,7));
%         contrainte_xx_yy_moyenne(j) = .02189 *
%         mean(fichier_vitesse(dans_boite_ij,8) - fichier_vitesse(dans_boite_ij,6));
    end;
end;

v_j(1,:) = NaN;
v_i(i_boite,:) = NaN;
% Tracé du champ de vitesse
figure();
% subplot(2,1,2);
% quiver(repmat([1:32]',1,15),repmat(1:15,32,1),v_i,v_j);
% % set(gca,'Units','centimeters');
% % pos = get(gca,'Position');
% % set(gca,'Position',[pos(1:2) 1024/60 375/60]);
% axis equal off;
% hold on;
% fill([238 238 860 860]/taille_boite_i + ones(1,4),0.5*ones(1,4) + [69 186 315 190]/taille_boite_j,'k');
% fill([1 1 1024 1024]/taille_boite_i + ones(1,4),0.5*ones(1,4) + [1 7 7 1]/taille_boite_j,'k');
% fill([1 1 1024 1024]/taille_boite_i + ones(1,4),0.5*ones(1,4) + [371 375 375 371]/taille_boite_j,'k');
quiver(taille_boite_i * repmat([1:i_boite]',1,j_boite),taille_boite_j * repmat(1:j_boite,i_boite,1),v_i,v_j,10);
% set(gca,'Units','centimeters');
% pos = get(gca,'Position');
% set(gca,'Position',[pos(1:2) 1024/60 375/60]);
axis equal off;
hold on;
fill([1 1 38+1 911+1] + 0.5*ones(1,4)*taille_boite_i, [1 468+1 468+1 1] + 0.5*ones(1,4)*taille_boite_j,'k');
fill([1 50+1 1013+1 1] + 0.5*ones(1,4)*taille_boite_i, [564+1 564+1 1023+1 1024] + 0.5*ones(1,4)*taille_boite_j,'k');

% Tracé du champ de contraintes
figure();
% subplot(2,1,1);
angle = 0:pi/30:2*pi;
for i = 2:i_boite-1
    for j = 1:j_boite
        line(taille_boite_i*(i-0.5) + (sigma_ii(i,j)*cos(angle) + sigma_ij(i,j)*sin(angle))/5,...
            taille_boite_j*(j-0.5) + (sigma_ij(i,j)*cos(angle) + sigma_jj(i,j)*sin(angle))/5);
    end;
end;
axis equal off;
hold on;
fill([1 1 38+1 911+1] + 0.5*ones(1,4)*taille_boite_i, [1 468+1 468+1 1] + 0.5*ones(1,4)*taille_boite_j,'k');
fill([1 50+1 1013+1 1] + 0.5*ones(1,4)*taille_boite_i, [564+1 564+1 1023+1 1024] + 0.5*ones(1,4)*taille_boite_j,'k');

% figure;
% plot(vitesse_moyenne_par_image(:,1),vitesse_moyenne_par_image(:,2));
% xlabel('Numéro de l''image'); ylabel('u (mm/s)');
% 
% figure;
% plot(fichier_vitesse_2(:,2),fichier_vitesse_2(:,3),'o');
% xlabel('Y (mm)'); ylabel('u (mm/s)');
% hold on; line([-53.28 -53.28],[0 20]); hold on; line([53.28 53.28],[0 20]);
% 
% figure;
% plot(fichier_vitesse_2(:,2),fichier_vitesse_2(:,5),'o');
% xlabel('Y (mm)'); ylabel('Aire (mm^2)');
% hold on; line([-53.28 -53.28],[0 100]); hold on; line([53.28 53.28],[0 100]);
% 
% % voir 25/04/2013 pour le facteur numérique des contraintes
% figure;
% plot(fichier_vitesse_2(:,2),21.89/1000*fichier_vitesse_2(:,7),'or',...
%     fichier_vitesse_2(:,2),21.89/1000*(fichier_vitesse_2(:,8) - fichier_vitesse_2(:,6)),'og');
% xlabel('Y (mm)'); ylabel('Contraintes (Pa m)'); legend('\sigma_{xy}','\sigma_{xx} - \sigma_{yy}');
% hold on; line([-53.28 -53.28],[-20 20]); hold on; line([53.28 53.28],[-20 20]);
% 
% figure;
% cdfplot(fichier_vitesse_2(:,2));
% xlabel('Y (mm)'); ylabel('Proportion de bulles en y < Y');
% hold on; line([-53.28 -53.28],[0 1]); hold on; line([53.28 53.28],[0 1]);
% 
% params_fit_Janiaud = fit_Janiaud(position_boite,vitesse_moyenne);
% 
% figure;
% errorbar(position_boite,vitesse_moyenne,std_vitesse);
% xlabel('Y (mm)'); ylabel('u (mm/s)');
% hold on; line([-53.28 -53.28],[0 20]); hold on; line([53.28 53.28],[0 20]);
% hold on; plot(position_boite, params_fit_Janiaud(1).*(1 + params_fit_Janiaud(2).*cosh(position_boite./params_fit_Janiaud(3))));
% 
% figure;
% % errorbar([position_boite position_boite],[contrainte_xy_moyenne contrainte_xx_yy_moyenne],...
% %     [std_contrainte_xy std_contrainte_xx_yy]);
% errorbar(position_boite(2:nombre_boites-1),contrainte_xy_moyenne(2:nombre_boites-1),std_contrainte_xy(2:nombre_boites-1));
% hold on;
% errorbar(position_boite(2:nombre_boites-1),contrainte_xx_yy_moyenne(2:nombre_boites-1),std_contrainte_xx_yy(2:nombre_boites-1));
% xlabel('Y (mm)'); ylabel('COntraintes (Pa m)');
% hold on; line([-53.28 -53.28],[-.5 .5]); hold on; line([53.28 53.28],[-.5 .5]);