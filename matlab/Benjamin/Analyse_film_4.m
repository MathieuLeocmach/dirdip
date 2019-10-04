tic

nombre_image = 100;

vitesse_derive = [0 0];

bulles_toutes_images = {};
disparus = [];
apparus = [];

I_avant = imread(['E:\Claire\20150515_LAc_500mLmin_2mm_pompe\Essai_analyse_images\20150515_LAc_500mLmin_2mm_pompe_0001Frame.bmp'])/255;

% % spécifique Sian_tilted!!!
% I_avant = 1 - I_avant;

% I_avant = imread(['H:\Sian_320080\001Frame.bmp'])/255;
% I_avant = imread(['C:\Users\bdollet\Desktop\CNRS\Manips\2011 canal droit Simon\Analyse_Manip_4_20130425\Analyse_T1_442\T1_442_001.tif'])/255;
[I_avant, nombre_bulles_avant] = bwlabel(I_avant,4);

hauteur_image = size(I_avant,1);
largeur_image = size(I_avant,2);

% L = NaN*ones(hauteur_image,largeur_image,nombre_image);
% 
for num_image=1:nombre_image-1

    % Affichage du numéro courant
    disp(['Image n°' num2str(num_image)])
    
    % Pour la première image, on analyse à la fois l'image 1 et 2 dans la
    % première occurrence de la boucle. En effet, on a besoin de ces deux
    % images pour déterminer les déplacements
    if num_image == 1
        
        imshow(I_avant);

        % La fonction trouve_vertex2 sort la liste des indices linéaires des vertex, ainsi que le
        % numéro des huit pixels voisins
        v = trouve_vertex2(I_avant);

        v_temp = zeros(size(v,1),5);
        v_temp(:,1) = v(:,1);
        for j = 1:size(v,1)
            vv = unique(nonzeros(v(j,2:9)));
            for k = 1:length(vv)
                v_temp(j,k+1) = vv(k);
            end;
        end;
        v = v_temp;
        clear v_temp;

        bulles_avant = regionprops(I_avant','Area','Centroid');
        bulles_avant(1).Vertex = [];
        bulles_avant(1).Cotes = [];
        bulles_avant(1).Deplacement = [];
        bulles_avant(1).Bord = [];
        bulles_avant(1).Suivante = [];

        for j = 1:size(v,1)
            vv = nonzeros(v(j,2:5));
            for k = 1:length(vv)
                bulles_avant(vv(k)).Vertex = [bulles_avant(vv(k)).Vertex v(j,1)];
            end;
        end;

        bulles_de_bord_appariees = trouve_cotes_bord(v);
        % Note du 21/03/2013 : il y a des cas rares où des triplets apparaissent.
        % On identifie d'abord les triplets
        patho_bord = find(diff(diff(bulles_de_bord_appariees(:,2)))==0);
        % et on les élimine
        if length(patho_bord)>0
            bulles_de_bord_appariees = bulles_de_bord_appariees...
                (setdiff(1:size(bulles_de_bord_appariees,1),patho_bord+2),:);
        end;

        for j = 1:2:size(bulles_de_bord_appariees,1)
            [i_depart j_depart] = ind2sub(size(I_avant),bulles_de_bord_appariees(j,1));
            [i_arrivee j_arrivee] = ind2sub(size(I_avant),bulles_de_bord_appariees(j+1,1));
            depart = [i_depart j_depart];
            arrivee = [i_arrivee j_arrivee];
            bulles_avant(bulles_de_bord_appariees(j,2)).Cotes = [bulles_avant(bulles_de_bord_appariees(j,2)).Cotes; depart - arrivee];
        end;

        cotes_avant = trouve_cotes(v);
        % Note du 21/03/2013 : il y a des cas rares où des triplets apparaissent.
        % On identifie d'abord les triplets
        patho = find(ismember(diff(diff(cotes_avant(:,2:3))), [0 0], 'rows'));
        % et on les élimine
        if length(patho)>0
            cotes_avant = cotes_avant(setdiff(1:size(cotes_avant,1),patho+2),:);
        end;

        for j = 1:2:size(cotes_avant,1)
            [i_depart j_depart] = ind2sub(size(I_avant),cotes_avant(j,1));
            [i_arrivee j_arrivee] = ind2sub(size(I_avant),cotes_avant(j+1,1));
            depart = [i_depart j_depart];
            arrivee = [i_arrivee j_arrivee];
            bulles_avant(cotes_avant(j,2)).Cotes = [bulles_avant(cotes_avant(j,2)).Cotes; depart - arrivee];
            bulles_avant(cotes_avant(j,3)).Cotes = [bulles_avant(cotes_avant(j,3)).Cotes; depart - arrivee];
        end;
        
        % Détection des bulles touchant le bord de l'image
        bulles_bord_image_avant = unique(nonzeros([I_avant(1,:) I_avant(size(I_avant,1),:) I_avant(:,1)' I_avant(:,size(I_avant,2))']));
        for j = 1:nombre_bulles_avant
            if ismember(j,bulles_bord_image_avant)
                bulles_avant(j).Bord = 1;
            else bulles_avant(j).Bord = 0;
            end;
        end;
        
        I_apres = imread(['E:\Claire\20150515_LAc_500mLmin_2mm_pompe\Essai_analyse_images\20150515_LAc_500mLmin_2mm_pompe_0002Frame.bmp'])/255;
%         I_apres = imread(['H:\Sian_320080\002Frame.bmp'])/255;
%         I_apres = imread(['C:\Users\bdollet\Desktop\CNRS\Manips\2011 canal droit Simon\Analyse_Manip_4_20130425\Analyse_T1_442\T1_442_002.tif'])/255;

        % spécifique Sian_tilted!!!
        I_apres = 1 - I_apres;
        
        % On labellise l'image
        [I_apres, nombre_bulles_apres] = bwlabel(I_apres,4);
        imshow(I_apres);

        % La fonction trouve_vertex2 sort la liste des indices linéaires des vertex, ainsi que le
        % numéro des huit pixels voisins
        v = trouve_vertex2(I_apres);

        v_temp = zeros(size(v,1),5);
        v_temp(:,1) = v(:,1);
        for j = 1:size(v,1)
            vv = unique(nonzeros(v(j,2:9)));
            for k = 1:length(vv)
                v_temp(j,k+1) = vv(k);
            end;
        end;
        v = v_temp;
        clear v_temp;

        bulles_apres = regionprops(I_apres','Area','Centroid');
        bulles_apres(1).Vertex = [];
        bulles_apres(1).Cotes = [];
        bulles_apres(1).Deplacement = [];
        bulles_apres(1).Bord = [];

        for j = 1:size(v,1)
            vv = nonzeros(v(j,2:5));
            for k = 1:length(vv)
                bulles_apres(vv(k)).Vertex = [bulles_apres(vv(k)).Vertex v(j,1)];
            end;
        end;

        bulles_de_bord_appariees = trouve_cotes_bord(v);
        % Note du 21/03/2013 : il y a des cas rares où des triplets apparaissent.
        % On identifie d'abord les triplets
        patho_bord = find(diff(diff(bulles_de_bord_appariees(:,2)))==0);
        % et on les élimine
        if length(patho_bord)>0
            bulles_de_bord_appariees = bulles_de_bord_appariees...
                (setdiff(1:size(bulles_de_bord_appariees,1),patho_bord+2),:);
        end;

        for j = 1:2:size(bulles_de_bord_appariees,1)
            [i_depart j_depart] = ind2sub(size(I_apres),bulles_de_bord_appariees(j,1));
            [i_arrivee j_arrivee] = ind2sub(size(I_apres),bulles_de_bord_appariees(j+1,1));
            depart = [i_depart j_depart];
            arrivee = [i_arrivee j_arrivee];
            bulles_apres(bulles_de_bord_appariees(j,2)).Cotes = [bulles_apres(bulles_de_bord_appariees(j,2)).Cotes; depart - arrivee];
        end;

        cotes_apres = trouve_cotes(v);
        % Note du 21/03/2013 : il y a des cas rares où des triplets apparaissent.
        % On identifie d'abord les triplets
        patho = find(ismember(diff(diff(cotes_apres(:,2:3))), [0 0], 'rows'));
        % et on les élimine
        if length(patho)>0
            cotes_apres = cotes_apres(setdiff(1:size(cotes_apres,1),patho+2),:);
        end;

        for j = 1:2:size(cotes_apres,1)
            [i_depart j_depart] = ind2sub(size(I_apres),cotes_apres(j,1));
            [i_arrivee j_arrivee] = ind2sub(size(I_apres),cotes_apres(j+1,1));
            depart = [i_depart j_depart];
            arrivee = [i_arrivee j_arrivee];
            bulles_apres(cotes_apres(j,2)).Cotes = [bulles_apres(cotes_apres(j,2)).Cotes; depart - arrivee];
            bulles_apres(cotes_apres(j,3)).Cotes = [bulles_apres(cotes_apres(j,3)).Cotes; depart - arrivee];
        end;
        
        % Détection des bulles touchant le bord de l'image
        bulles_bord_image_apres = unique(nonzeros([I_apres(1,:) I_apres(size(I_apres,1),:) I_apres(:,1)' I_apres(:,size(I_apres,2))']));
        for j = 1:nombre_bulles_apres
            if ismember(j,bulles_bord_image_apres)
                bulles_apres(j).Bord = 1;
            else bulles_apres(j).Bord = 0;
            end;
        end;
        
        % Calcul du déplacement
        source = [bulles_avant(:).Centroid];
        source = reshape(source,2,length(source)/2)';
        target = [bulles_apres(:).Centroid];
        target = reshape(target,2,length(target)/2)';
        target_indices = nearestneighborlinker_benji(source,target,vitesse_derive);
        source_indices = nearestneighborlinker_benji(target,source,-vitesse_derive);
        deplacement = target(target_indices,:) - source;
        provenance = source(source_indices,:) - target;
        condition_de_bon_suivi = source_indices(target_indices) == [1:length(source_indices(target_indices))]';
        bon_suivi = find(condition_de_bon_suivi);
        condition_terug = target_indices(source_indices) == [1:length(target_indices(source_indices))]';
        bon_terug = find(condition_terug);
        for j = 1:length(target_indices)
            bulles_avant(j).Deplacement = deplacement(j,:);
            if ismember(j,condition_de_bon_suivi)
                bulles_avant(j).Suivante = target_indices(j);
            end;
        end;
        
        % Détection des T1
        % Sur l'image avant, on ne garde (en simple exemplaire) que les
        % côtés des bulles bien suivies ET ne touchant pas les bords de
        % l'image
        cotes_attendus = cotes_avant(1:2:size(cotes_avant,1),2:3);
%         cotes_attendus = cotes_attendus(find(condition_de_bon_suivi(cotes_attendus(:,1)).*...
%             condition_de_bon_suivi(cotes_attendus(:,2))),:);
        cotes_attendus = cotes_attendus(find(condition_de_bon_suivi(cotes_attendus(:,1)).*...
            condition_de_bon_suivi(cotes_attendus(:,2)) &...
            ~ismember(cotes_attendus(:,1),bulles_bord_image_avant) & ~ismember(cotes_attendus(:,2),bulles_bord_image_avant)),:);
        % On applique sur cette liste de côtés des bulles bien suivies la
        % transformation menant aux indices de ces mêmes bulles à l'image d'après
        cotes_attendus_2 = target_indices(cotes_attendus);
        cotes_attendus_2 = sortrows(sort(cotes_attendus_2,2));
        % cotes_disparus est la liste des côtés présents sur l'image avant
        % et absents sur l'image après
        cotes_disparus = find(~ismember(cotes_attendus_2,cotes_apres(1:2:size(cotes_apres,1),2:3),'rows'));
        % On fait le même travail à l'envers, pour détecter les côtés
        % apparus
        cotes_terug = cotes_apres(1:2:size(cotes_apres,1),2:3);
%         cotes_terug = cotes_terug(find(condition_terug(cotes_terug(:,1)).*...
%             condition_terug(cotes_terug(:,2))),:);
        cotes_terug = cotes_terug(find(condition_terug(cotes_terug(:,1)).*...
            condition_terug(cotes_terug(:,2)) &...
            ~ismember(cotes_terug(:,1),bulles_bord_image_apres) & ~ismember(cotes_terug(:,2),bulles_bord_image_apres)),:);
        cotes_terug_2 = source_indices(cotes_terug);
        cotes_terug_2 = sortrows(sort(cotes_terug_2,2));
        cotes_apparus = find(~ismember(cotes_terug_2,cotes_avant(1:2:size(cotes_avant,1),2:3),'rows'));

        bulles_toutes_images{num_image} = bulles_avant;
        imshow(I_apres);
        hold on;
        imshow(I_avant);

%         I_combi = I_apres;
%         I_combi(find(I_avant & I_apres)) = 1;
%         I_combi(find(~I_apres)) = 0;
%         I_combi(find(~I_avant & I_apres)) = 0.5;
%         imshow(I_combi);
% 
%         hold on;
%         quiver(source(:,2),source(:,1),deplacement(:,2),deplacement(:,1));
%         hold on;
%         quiver(target(:,2),target(:,1),provenance(:,2),provenance(:,1));
%         hold on;
%         quiver(source(bon_suivi,2),source(bon_suivi,1),deplacement(bon_suivi,2),deplacement(bon_suivi,1),1);

%         hold on;
        clear depart_d arrivee_d;
        depart_d = reshape([bulles_avant(source_indices(cotes_attendus_2(cotes_disparus,1))).Centroid],2,length(cotes_disparus));
        arrivee_d = reshape([bulles_avant(source_indices(cotes_attendus_2(cotes_disparus,2))).Centroid],2,length(cotes_disparus));
%         line([depart_d(2,:); arrivee_d(2,:)],[depart_d(1,:); arrivee_d(1,:)],'Color','r');
%         hold on;
        clear depart_a arrivee_a;
        depart_a = reshape([bulles_apres(target_indices(cotes_terug_2(cotes_apparus,1))).Centroid],2,length(cotes_apparus));
        arrivee_a = reshape([bulles_apres(target_indices(cotes_terug_2(cotes_apparus,2))).Centroid],2,length(cotes_apparus));
%         line([depart_a(2,:); arrivee_a(2,:)],[depart_a(1,:); arrivee_a(1,:)],'Color','g');
%         saveas(gcf,['C:\Users\bdollet\Desktop\CNRS\Encadrement\Stages\2015 M1 Bocher\20150504_SDS_100mLmin_2mm_pompe_1\Analyse_T1\', num2str(num_image+1,'%04.0f'), 'Frame.fig']);
        
        disparus_pour_sauv = zeros(length(cotes_disparus),7);
        disparus_pour_sauv(:,1) = num_image;
        disparus_pour_sauv(:,2:3) = depart_d';
        disparus_pour_sauv(:,4) = source_indices(cotes_attendus_2(cotes_disparus,1));
        disparus_pour_sauv(:,5:6) = arrivee_d';
        disparus_pour_sauv(:,7) = source_indices(cotes_attendus_2(cotes_disparus,2));
        disparus = [disparus; disparus_pour_sauv];
        apparus_pour_sauv = zeros(length(cotes_apparus),7);
        apparus_pour_sauv(:,1) = num_image;
        apparus_pour_sauv(:,2:3) = depart_a';
        apparus_pour_sauv(:,4) = target_indices(cotes_terug_2(cotes_apparus,1));
        apparus_pour_sauv(:,5:6) = arrivee_a';
        apparus_pour_sauv(:,7) = target_indices(cotes_terug_2(cotes_apparus,2));
        apparus = [apparus; apparus_pour_sauv];
       




        
    % Pour les images autres que la première, les données pour l'image
    % num_image ont déjà été obtenues à l'occurrence précédente. On analyse
    % donc l'image num_image+1
    else
        
        I_avant = I_apres;
        bulles_avant = bulles_apres;
        cotes_avant = cotes_apres;
        bulles_bord_image_avant = bulles_bord_image_apres;
        numbre_bulles_avant = nombre_bulles_apres;
        I_apres = imread(['E:\Claire\20150515_LAc_500mLmin_2mm_pompe\Essai_analyse_images\20150515_LAc_500mLmin_2mm_pompe_', num2str(num_image+1,'%04.0f'), 'Frame.bmp'])/255;
%         I_apres = imread(['H:\Sian_320080\', num2str(num_image+1,'%03.0f'), 'Frame.bmp'])/255;
%         I_apres = imread(['C:\Users\bdollet\Desktop\CNRS\Manips\2011 canal droit Simon\Analyse_Manip_4_20130425\Analyse_T1_442\T1_442_', num2str(num_image+1,'%03.0f'), '.tif'])/255;
        
        % spécifique Sian_tilted!!!
        I_apres = 1 - I_apres;

        % On labellise l'image
        [I_apres, nombre_bulles_apres] = bwlabel(I_apres,4);

        % La fonction trouve_vertex2 sort la liste des indices linéaires des vertex, ainsi que le
        % numéro des huit pixels voisins
        v = trouve_vertex2(I_apres);

        v_temp = zeros(size(v,1),5);
        v_temp(:,1) = v(:,1);
        for j = 1:size(v,1)
            vv = unique(nonzeros(v(j,2:9)));
            for k = 1:length(vv)
                v_temp(j,k+1) = vv(k);
            end;
        end;
        v = v_temp;
        clear v_temp;

        bulles_apres = regionprops(I_apres','Area','Centroid');
        bulles_apres(1).Vertex = [];
        bulles_apres(1).Cotes = [];
        bulles_apres(1).Deplacement = [];
        bulles_apres(1).Bord = [];

        for j = 1:size(v,1)
            vv = nonzeros(v(j,2:5));
            for k = 1:length(vv)
                bulles_apres(vv(k)).Vertex = [bulles_apres(vv(k)).Vertex v(j,1)];
            end;
        end;

        bulles_de_bord_appariees = trouve_cotes_bord(v);
        % Note du 21/03/2013 : il y a des cas rares où des triplets apparaissent.
        % On identifie d'abord les triplets
        patho_bord = find(diff(diff(bulles_de_bord_appariees(:,2)))==0);
        % et on les élimine
        if length(patho_bord)>0
            bulles_de_bord_appariees = bulles_de_bord_appariees...
                (setdiff(1:size(bulles_de_bord_appariees,1),patho_bord+2),:);
        end;

        for j = 1:2:size(bulles_de_bord_appariees,1)
            [i_depart j_depart] = ind2sub(size(I_apres),bulles_de_bord_appariees(j,1));
            [i_arrivee j_arrivee] = ind2sub(size(I_apres),bulles_de_bord_appariees(j+1,1));
            depart = [i_depart j_depart];
            arrivee = [i_arrivee j_arrivee];
            bulles_apres(bulles_de_bord_appariees(j,2)).Cotes = [bulles_apres(bulles_de_bord_appariees(j,2)).Cotes; depart - arrivee];
        end;

        cotes_apres = trouve_cotes(v);
        % Note du 21/03/2013 : il y a des cas rares où des triplets apparaissent.
        % On identifie d'abord les triplets
        patho = find(ismember(diff(diff(cotes_apres(:,2:3))), [0 0], 'rows'));
        % et on les élimine
        if length(patho)>0
            cotes_apres = cotes_apres(setdiff(1:size(cotes_apres,1),patho+2),:);
        end;

        for j = 1:2:size(cotes_apres,1)
            [i_depart j_depart] = ind2sub(size(I_apres),cotes_apres(j,1));
            [i_arrivee j_arrivee] = ind2sub(size(I_apres),cotes_apres(j+1,1));
            depart = [i_depart j_depart];
            arrivee = [i_arrivee j_arrivee];
            bulles_apres(cotes_apres(j,2)).Cotes = [bulles_apres(cotes_apres(j,2)).Cotes; depart - arrivee];
            bulles_apres(cotes_apres(j,3)).Cotes = [bulles_apres(cotes_apres(j,3)).Cotes; depart - arrivee];
        end;
        
        % Détection des bulles touchant le bord de l'image
        bulles_bord_image_apres = unique(nonzeros([I_apres(1,:) I_apres(size(I_apres,1),:) I_apres(:,1)' I_apres(:,size(I_apres,2))']));
        for j = 1:nombre_bulles_apres
            if ismember(j,bulles_bord_image_apres)
                bulles_apres(j).Bord = 1;
            else bulles_apres(j).Bord = 0;
            end;
        end;
        
        % Calcul du déplacement
        source = [bulles_avant(:).Centroid];
        source = reshape(source,2,length(source)/2)';
        target = [bulles_apres(:).Centroid];
        target = reshape(target,2,length(target)/2)';
        target_indices = nearestneighborlinker_benji(source,target,vitesse_derive);
        source_indices = nearestneighborlinker_benji(target,source,-vitesse_derive);
        deplacement = target(target_indices,:) - source;
        provenance = source(source_indices,:) - target;
        condition_de_bon_suivi = source_indices(target_indices) == [1:length(source_indices(target_indices))]';
        bon_suivi = find(condition_de_bon_suivi);
        condition_terug = target_indices(source_indices) == [1:length(target_indices(source_indices))]';
        bon_terug = find(condition_terug);
        for j = 1:length(target_indices)
            bulles_avant(j).Deplacement = deplacement(j,:);
        end;
        for j = 1:length(target_indices)
            bulles_avant(j).Deplacement = deplacement(j,:);
            if ismember(j,bon_suivi)
                bulles_avant(j).Suivante = target_indices(j);
            end;
        end;
        
        % Détection des T1
        % Sur l'image avant, on ne garde (en simple exemplaire) que les
        % côtés des bulles bien suivies
        cotes_attendus = cotes_avant(1:2:size(cotes_avant,1),2:3);
%         cotes_attendus = cotes_attendus(find(condition_de_bon_suivi(cotes_attendus(:,1)).*...
%             condition_de_bon_suivi(cotes_attendus(:,2))),:);
        cotes_attendus = cotes_attendus(find(condition_de_bon_suivi(cotes_attendus(:,1)).*...
            condition_de_bon_suivi(cotes_attendus(:,2)) &...
            ~ismember(cotes_attendus(:,1),bulles_bord_image_avant) & ~ismember(cotes_attendus(:,2),bulles_bord_image_avant)),:);
        % On applique sur cette liste de côtés des bulles bien suivies la
        % transformation menant aux indices de ces mêmes bulles à l'image d'après
        cotes_attendus_2 = target_indices(cotes_attendus);
        cotes_attendus_2 = sortrows(sort(cotes_attendus_2,2));
        % cotes_disparus est la liste des côtés présents sur l'image avant
        % et absents sur l'image après
        cotes_disparus = find(~ismember(cotes_attendus_2,cotes_apres(1:2:size(cotes_apres,1),2:3),'rows'));
        % On fait le même travail à l'envers, pour détecter les côtés
        % apparus
        cotes_terug = cotes_apres(1:2:size(cotes_apres,1),2:3);
%         cotes_terug = cotes_terug(find(condition_terug(cotes_terug(:,1)).*...
%             condition_terug(cotes_terug(:,2))),:);
        cotes_terug = cotes_terug(find(condition_terug(cotes_terug(:,1)).*...
            condition_terug(cotes_terug(:,2)) &...
            ~ismember(cotes_terug(:,1),bulles_bord_image_apres) & ~ismember(cotes_terug(:,2),bulles_bord_image_apres)),:);
        cotes_terug_2 = source_indices(cotes_terug);
        cotes_terug_2 = sortrows(sort(cotes_terug_2,2));
        cotes_apparus = find(~ismember(cotes_terug_2,cotes_avant(1:2:size(cotes_avant,1),2:3),'rows'));

        bulles_toutes_images{num_image} = bulles_avant;
        imshow(I_apres);
        hold on;
        imshow(I_avant);

        clear I_combi;
        I_combi = I_apres;
        I_combi(find(I_avant & I_apres)) = 1;
        I_combi(find(~I_apres)) = 0;
        I_combi(find(~I_avant & I_apres)) = 0.5;
        imshow(I_combi);
%  
%         hold on;
%         quiver(source(:,2),source(:,1),deplacement(:,2),deplacement(:,1));
%         hold on;
%         quiver(target(:,2),target(:,1),provenance(:,2),provenance(:,1));
%         hold on;
%         quiver(source(bon_suivi,2),source(bon_suivi,1),deplacement(bon_suivi,2),deplacement(bon_suivi,1),1);

        hold on;
        clear depart_d arrivee_d;
        depart_d = reshape([bulles_avant(source_indices(cotes_attendus_2(cotes_disparus,1))).Centroid],2,length(cotes_disparus));
        arrivee_d = reshape([bulles_avant(source_indices(cotes_attendus_2(cotes_disparus,2))).Centroid],2,length(cotes_disparus));
        line([depart_d(2,:); arrivee_d(2,:)],[depart_d(1,:); arrivee_d(1,:)],'Color','r');
        hold on;
        clear depart_a arrivee_a;
        depart_a = reshape([bulles_apres(target_indices(cotes_terug_2(cotes_apparus,1))).Centroid],2,length(cotes_apparus));
        arrivee_a = reshape([bulles_apres(target_indices(cotes_terug_2(cotes_apparus,2))).Centroid],2,length(cotes_apparus));
        line([depart_a(2,:); arrivee_a(2,:)],[depart_a(1,:); arrivee_a(1,:)],'Color','g');
        saveas(gcf,['C:\Users\bdollet\Desktop\CNRS\Encadrement\Stages\2015 M1 Bocher\20150515_LAc_500mLmin_mm_pompe\Analyse_T1\', num2str(num_image+1,'%04.0f'), 'Frame.fig']);
        
        disparus_pour_sauv = zeros(length(cotes_disparus),7);
        disparus_pour_sauv(:,1) = num_image;
        disparus_pour_sauv(:,2:3) = depart_d';
        disparus_pour_sauv(:,4) = source_indices(cotes_attendus_2(cotes_disparus,1));
        disparus_pour_sauv(:,5:6) = arrivee_d';
        disparus_pour_sauv(:,7) = source_indices(cotes_attendus_2(cotes_disparus,2));
        disparus = [disparus; disparus_pour_sauv];
        apparus_pour_sauv = zeros(length(cotes_apparus),7);
        apparus_pour_sauv(:,1) = num_image;
        apparus_pour_sauv(:,2:3) = depart_a';
        apparus_pour_sauv(:,4) = target_indices(cotes_terug_2(cotes_apparus,1));
        apparus_pour_sauv(:,5:6) = arrivee_a';
        apparus_pour_sauv(:,7) = target_indices(cotes_terug_2(cotes_apparus,2));
        apparus = [apparus; apparus_pour_sauv];
        
        
        
        
        
        
    end;
    
%     I = imread(['H:\ManipsIsaben\Poiseuille\20130110\Manip_3\Analyse\', numero_image_texte, 'Frame.bmp'])/255;
%     I = imread(['Obstacle_artificiel.tif'])/255;
%     I = imread(['001Frame-extrait.bmp'])/255;
    
end;

toc