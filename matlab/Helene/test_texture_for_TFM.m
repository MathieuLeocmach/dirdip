clear;
%close all;
clc;

dossier='/media/thoman/My Passport/Melina/190212_MDCK_WT_hyppodrome/Fluo/registered4/';
nom='Device_fluo_bb_top_1_step_a.tiff';
nomph='Device_phase_bb_1_step_a.tiff';
param.minDist=5;
param.maxFeature=40000;
for tt=1:511
    tt
    im=im2double(imread([dossier,nom],tt));


    imbg=imopen(im, strel('disk',100));
 
    [Z, ~] = dop2DApprox( imbg, 3,3 );
%     figure
%     imagesc(Z)
   im1=im-Z;

% bk=imgaussfilt(im,100);
% im1 = im-bk;

    [pts(:,1),pts(:,2)]=localMaximum_h(im1,param.minDist,1,param.maxFeature);
%     J=imadjust(im1,[0 0.01],[]);
%     figure
% hold on
% imagesc(J)
% colormap(gray)
% 
% plot(pts(:,2),pts(:,1),'Marker','.','MarkerSize',5,'LineStyle','none')
nb=200;
 [mat2,matdist] = kNearestNeighbors(pts,nb);
%  figure
%  histogram(mean(matdist,2))

tabx=repmat(pts(:,2),1,size(mat2,2))-reshape(pts(mat2,2),size(mat2));
taby=repmat(pts(:,1),1,size(mat2,2))-reshape(pts(mat2,1),size(mat2));
%%
ind=matdist>40;
tabx(ind)=NaN;
taby(ind)=NaN;
esp(:,1)=nanmean(tabx.^2,2);
esp(:,2)=nanmean(taby.^2,2);
esp(:,3)=nanmean(tabx.*taby,2);
% pas=50;
% [X,Y]=meshgrid(1:pas:size(im1,2),1:pas:size(im,1));
% X2=(X(1:end-1,1:end-1)+X(2:end,2:end))/2;
% Y2=(Y(1:end-1,1:end-1)+Y(2:end,2:end))/2;
% imph=imread([dossier,nomph],tt);

%% get ref M0
% figure
% imagesc(imph)
% colormap(gray)
%  h=impoly;
%     BW = createMask(h);
%     id=sub2ind(size(BW),round(pts(:,1)),round(pts(:,2)));
%     indp=BW(id)==1;
%     A(1,1)=nanmean(esp(indp,1));
%     A(2,2)=nanmean(esp(indp,2));
%     A(1,2)=nanmean(esp(indp,3));
%     A(2,1)=A(1,2);
%     if sum(isnan(A(:)))==0
%            [V,D]=eig(A);
%     end
%     D0=trace(D)/2
%% Visualisation part
% close all
% figure
% hold on
% imagesc(imph)
% colormap(gray)
% axis equal
% set(gca,'Ydir','reverse')
% clear ind
% for ii=1:size(X,1)-1
%     for jj=1:size(X,2)-1
% ind=pts(:,2)>X(ii,jj)&pts(:,2)<X(ii+1,jj+1)&pts(:,1)>Y(ii,jj)&pts(:,1)<Y(ii+1,jj+1);
%     A(1,1)=nanmean(esp(ind,1));
%     A(2,2)=nanmean(esp(ind,2));
%     A(1,2)=nanmean(esp(ind,3));
%     A(2,1)=A(1,2);
%     if sum(isnan(A(:)))==0
%            [V,D]=eig(A);
%    m=max(D(1,1),D(2,2));
% %    if D{num}(1,1)<10000&&D{num}(2,2)<10000
% %    ellipse2(D{num}(1,1)/25,D{num}(2,2)/25,atan2(V{num}(2,1),V{num}(1,1)),X(ii),Y(jj),'r',10);
%    scale=0.01;
%    ellipse2_ax(abs((log(D(1,1))-log(D0)))/scale,abs((log(D(2,2))-log(D0))/scale),atan2(V(2,1),V(1,1)),X2(ii,jj),Y2(ii,jj),'r',30);         
%     
%     
%     end
%     end
% end
res(tt).pts=pts;
res(tt).esp=esp;
save([dossier,'restext.mat'],'res')

end

