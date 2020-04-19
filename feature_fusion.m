function [ F ] = feature_fusion(I,dsifts,mode)



[H,W,~,N]=size(I);
% 
imgs=double(I)/255;

imgs_gray=zeros(H,W,N);
for i=1:N
    imgs_gray(:,:,i)=rgb2gray(imgs(:,:,:,i));
end
% 

%local contrast
contrast_map=zeros(H,W,N);
for i=1:N
    contrast_map(:,:,i)=sum(dsifts(:,:,:,i),3);
end

%exposure quality
exposure_map=ones(H,W,N);
exposure_map((imgs_gray>=0.8)|(imgs_gray<=0.2))=0;

%spatial consistency (only for dynamic fusion)
if mode~=1
    for i=1:N
        [dsifts(:,:,:,i)] = FNormalization(dsifts(:,:,:,i));
    end
    distance_map=zeros(H,W,N);
    sigma_map=0.03.*ones(H,W);
%     ker=ones(19,19)./(19*19);
    for i=1:N
        for j=1:N
            if j~=i
%                 distance=imfilter(sum((dsifts(:,:,:,i)-dsifts(:,:,:,j)).^2, 3),ker,'replicate');      
                 distance=sum((dsifts(:,:,:,i)-dsifts(:,:,:,j)).^2, 3);    
                distance_map(:,:,i)=distance_map(:,:,i)+exp(-(0.5.*distance)./(sigma_map.^2));                   
            end
        end
    end
    T_map=exposure_map.*distance_map;
else
    T_map=exposure_map;
end

T_map = T_map + 10^-25; %avoids division by zero
T_map = T_map./repmat(sum(T_map,3),[1 1 N]);

weight_map=contrast_map.*T_map;

%weight map refinement
for i=1:N
    weight_map(:,:,i) = RF(weight_map(:,:,i), 50, 4, 3, imgs(:,:,:,i));
end

weight_map = weight_map + 10^-25; %avoids division by zero
weight_map = weight_map./repmat(sum(weight_map,3),[1 1 N]);

%fusion
F=zeros(H,W,3);
for i=1:N
    w = repmat(weight_map(:,:,i),[1 1 3]);
    F=F+imgs(:,:,:,i).*w;
end

end
