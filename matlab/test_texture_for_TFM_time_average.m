clear;
close all;
clc;

dossier='/media/thoman/My Passport/Melina/190212_MDCK_WT_hyppodrome/Fluo/registered4/';
nom='Device_fluo_bb_top_1_step_a.tiff';
nomph='Device_phase_bb_1_step_a.tiff';
load([dossier,'restext.mat'])
for tt=511:511
    nav=0;
     pas=50;
    im=imread([dossier,nom],tt);
    A=zeros(ceil(size(im,1)/pas),ceil(size(im,2)/pas),2,2);
%     for uu=tt-nav/2+1:tt+nav/2
       for uu=tt
        [X,Y]=meshgrid(1:pas:size(im,2),1:pas:size(im,1));
        X2=(X(1:end-1,1:end-1)+X(2:end,2:end))/2;
        Y2=(Y(1:end-1,1:end-1)+Y(2:end,2:end))/2;
        
        
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
        %%
 
        for ii=1:size(X,1)-1
            for jj=1:size(X,2)-1
                
                ind=res(uu).pts(:,2)>X(ii,jj)&res(uu).pts(:,2)<X(ii+1,jj+1)&res(uu).pts(:,1)>Y(ii,jj)&res(uu).pts(:,1)<Y(ii+1,jj+1);
                A(ii,jj,1,1)=nanmean(res(uu).esp(ind,1))+A(ii,jj,1,1);
                A(ii,jj,2,2)=nanmean(res(uu).esp(ind,2))+A(ii,jj,2,2);
                A(ii,jj,1,2)=nanmean(res(uu).esp(ind,3))+A(ii,jj,1,2);
                A(ii,jj,2,1)=A(ii,jj,1,2);
                
                
            end
        end
        
    end
    %%
    A=A/nav;
    D0=400;
   %% 
    figure
    hold on
    imagesc(im)
    colormap(gray)
    set(gca,'Ydir','reverse')
    axis equal
    for ii=1:size(X,1)-1
        for jj=1:size(X,2)-1
            if sum(sum(isnan(A(ii,jj,:,:))))==0&sum(sum(isinf(A(ii,jj,:,:))))==0
                
                
                [V,D]=eig(squeeze(A(ii,jj,:,:)));
                
                m=max(D(1,1),D(2,2))
              
                %    if D{num}(1,1)<10000&&D{num}(2,2)<10000
                %    ellipse2(D{num}(1,1)/25,D{num}(2,2)/25,atan2(V{num}(2,1),V{num}(1,1)),X(ii),Y(jj),'r',10);
                scale=0.006;
               ellipse2_ax(abs((log(D(1,1))-log(D0)))/scale,abs((log(D(2,2))-log(D0))/scale),atan2(V(2,1),V(1,1)),X2(ii,jj),Y2(ii,jj),'r',30);
               % ellipse2_ax(abs(((D(1,1))))/scale,abs(((D(2,2)))/scale),atan2(V(2,1),V(1,1)),X2(ii,jj),Y2(ii,jj),'r',30);

            end
        end
    end
    
    
    %%
    
end

