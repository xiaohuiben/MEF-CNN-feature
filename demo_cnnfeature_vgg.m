
% ========================================================================
% Multi-Exposure Fusion with CNN Features, ICIP,2018
% algorithm Version 1.0
% Copyright(c) 2018, Hui Li and Lei Zhang
% All Rights Reserved.
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% This is an implementation of paper "Multi-Exposure Fusion with CNN Features"
% Please refer to the following paper:
% H. Li et al., "Multi-Exposure Fusion with CNN Features, ICIP,2018" In press
% Please kindly report any suggestions or corrections to xiaohui102788@126.com

clear ;
close all;

addpath(genpath(pwd));

net = load('imagenet-vgg-f.mat') ;  %% please download the pretrained model form the matconvnet website

% net = load('pascal-fcn32s-dag.mat') ;
% net1 = load('imagenet-resnet-50-dag.mat') ;
net = vl_simplenn_tidy(net) ;

    I = load_images3('ArchSequence'); % [0,1]
    imgSeqColor2 = uint8(load_images(Dir,1)); % use im2double
%     figure(1),imshow(uint8(mean(I,4)))
    [h,w,c,n]=size(I);
    %       output=zeros(h,w,size(Y,3),n); %% cannot be defined below
    tic
    for j=1:n
        imgs_gray=I(:,:,:,j);
        %         imgs_gray=rgb2gray(imgs_gray);
        
        useGPU=1;
        if useGPU
            input = gpuArray(single(imgs_gray));
        end
        if useGPU
            net = vl_simplenn_move(net, 'gpu') ;
        end
        
        %           res    = vl_simplenn(net,imgs_gray,[],[],'conserveMemory',true,'mode','test');
     
          res= vl_simplenn(net,input,[],[],'conserveMemory',false,'mode','test');
          Y=res(1).x;

        
        if useGPU
            t = gather(Y);
%              t1 =imresize(t,(w/size(t,2)));
              output(:,:,:,j)=t;
            %         input  = gather(input);
        end
    end
    
    %%
%     tic
%   mode=2; % for dynamic scene
mode=1; % for static scene
        C_out=feature_fusion(I,output,mode);
    toc
    figure,imshow(C_out)
    clear output;
    

