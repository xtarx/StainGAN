% Clear all previous data
clc, clear all, close all;


%% Display results of each method
verbose = 0;

% imread('/Users/xtarx/Documents/TUM/5th/Thesis/dataset/camelyon16/Processed/patches_Tumor_110.tif/(100, 111).png')

% dirName = 'Images/mitosis_exp5/source_images/';              %# folder path
dirName = '/Users/xtarx/Documents/TUM/5th/Thesis/dataset/camelyon16/Processed/patches_Tumor_110.tif/';              %# folder path

files = dir( fullfile(dirName,'*.png') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
% TargetImage = imread('Images/mitosis_exp5/target_images/0.png');
TargetImage = imread('/Users/xtarx/Documents/TUM/5th/Thesis/dataset/camelyon16/Processed/center_3_target_Tumor_016.tif/target.png');

ss_cycle=0;
ss_khan=0;
ss_macenko=0;

psnr_cycle=0;
psnr_khan=0;
psnr_macenko=0;

mse_cycle=0;
mse_khan=0;
mse_macenko=0;
% 
% i = 1;
% K = imabsdiff(imread(strcat('Images/mitosis_exp4/source_images/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));
% figure
% imshow(K,[])
%     
for i=1:numel(files)
    fname = fullfile(dirName,files{i});     %# full path to file
    disp(fname)
     SourceImage = imread(fname);
%     ss_source_target= ssim(imread(strcat('Images/mitosis_exp4/40samples@top-left/A/',files{i})),imread(strcat('Images/mitosis_exp4/40samples@top-left/H/',files{i})));
%     disp(ss_source_target)
%     ss_cycle= ss_cycle +ssim(imread(strcat('Images/mitosis_exp4/cycleGAN/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));
%     ss_khan= ss_khan +ssim(imread(strcat('Images/mitosis_exp4/khan/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));
%     ss_macenko= ss_macenko +ssim(imread(strcat('Images/mitosis_exp4/macenko/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));

   % PSNR
%     psnr_cycle= psnr_cycle +psnr(imread(strcat('Images/mitosis_exp4/cycleGAN/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));
%     psnr_khan= psnr_khan +psnr(imread(strcat('Images/mitosis_exp4/khan/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));
%     psnr_macenko= psnr_macenko +psnr(imread(strcat('Images/mitosis_exp4/macenko/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));

    %MSE
%     
%     mse_cycle= mse_cycle +immse(imread(strcat('Images/mitosis_exp4/cycleGAN/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));
%     mse_khan= mse_khan +immse(imread(strcat('Images/mitosis_exp4/khan/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));
%     mse_macenko= mse_macenko +immse(imread(strcat('Images/mitosis_exp4/macenko/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));

% 
%     K = imabsdiff(imread(strcat('Images/mitosis_exp4/macenko/',files{i})),imread(strcat('Images/mitosis_exp4/target_images/',files{i})));
%     figure
%     imshow(K,[])

    %% Stain Normalisation using Reinhard Method
    
%      disp('Stain Normalisation using Reinhard''s Method');   
%    
   [ NormRH ] = Norm( SourceImage, TargetImage, 'Reinhard', verbose );
%    filename = strcat('Images/mitosis_exp5/reinhard/',files{i});
   
   filename = strcat('/Users/xtarx/Documents/TUM/5th/Thesis/dataset/camelyon16/Processed/reinhard_stained_Tumor_110/',files{i});

   disp(filename)
   imwrite(NormRH,filename)

% 
% 
% %% Stain Separation using Macenko'
% 

%     [ NormMM ] = Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1, verbose);
%     filename = strcat('/Users/xtarx/Documents/TUM/5th/Thesis/dataset/camelyon16/Processed/macenko_stained_Tumor_110/',files{i});
%     disp(filename)
%     imwrite(NormMM,filename);
    
%    %% Stain Normalisation using the Non-Linear Spline Mapping Method khan
% 
% % 
%     [ NormSM ] = Norm(SourceImage, TargetImage, 'SCD', [], verbose);
%     filename = strcat('Images/mitosis_exp5/khan/',files{i});
%     disp(filename)
%     imwrite(NormSM,filename);
%  
    
end

% len=39;
% disp(strcat('SSIM of Macenko : ' ,num2str(ss_macenko/len)))
% disp(strcat('SSIM of Khan : ',num2str(ss_khan/len)))
% disp(strcat('SSIM of CYCLE: ',num2str(ss_cycle/len)))

% disp(strcat('PSNR of Macenko : ' ,num2str(psnr_macenko/len)))
% disp(strcat('PSNR of Khan : ',num2str(psnr_khan/len)))
% disp(strcat('PSNR of CYCLE: ',num2str(psnr_cycle/len)))


% disp(strcat('MSE of Macenko : ' ,num2str(mse_macenko/len)))
% disp(strcat('MSE of Khan : ',num2str(mse_khan/len)))
% disp(strcat('MSE of CYCLE: ',num2str(mse_cycle/len)))


%% Load Source & Target images
% SourceImage = imread('Images/mitosis_exp4/A03_00A.png');
% TargetImage = imread('Images/H03_00A.png');
% 




%% Stain Normalisation using Reinhard Method

% disp('Stain Normalisation using Reinhard''s Method');
% 
% [ NormRH ] = Norm( SourceImage, TargetImage, 'Reinhard', verbose );
% imwrite(NormRH,'NNNNNNN.png')
% ss = ssim(imread('NNNNNNN.png'),TargetImage);

% 
% 
% % ,num2str(ssimval))
% % ssimval = num2str(ssim(imread(strcat('Images/H',img,'_sn1.png')),TargetImage));
% % 
%  figure,
%  subplot(231); imshow(SourceImage);   title('SourceImage');
%  subplot(232); imshow(TargetImage);   title('TargetImage');
% %  subplot(233); imshow("232");  title(ssimval);
%  ss = ssim(imread('A03_00A_sn1.png'),TargetImage);
%  subplot(234); imshow('A03_00A_sn1.png');    title(strcat('SN1 GAN  : ',num2str(ss)));
%  ss = ssim(imread('A03_00A_sn2.png'),TargetImage);
%  subplot(235); imshow('A03_00A_sn2.png');     title(strcat('SN2 MM : ',num2str(ss)));
%   ss = ssim(imread('A03_00A_sn3.png'),TargetImage);
% 
%  subplot(236); imshow('A03_00A_sn3.png');    title(strcat('SN3 KH  : ',num2str(ss)));
%  set(gcf,'units','normalized','outerposition',[0 0 1 1]);
% 
%  
% SourceImage = imread('Images/A03_01C.png');
% TargetImage = imread('Images/H03_01C.png');
% 
%   figure,
%  subplot(231); imshow(SourceImage);   title('SourceImage');
%  subplot(232); imshow(TargetImage);   title('TargetImage');
% %  subplot(233); imshow("232");  title(ssimval);
%  ss = ssim(imread('A03_01C_sn1.png'),TargetImage);
%  subplot(234); imshow('A03_01C_sn1.png');    title(strcat('SN1 GAN  : ',num2str(ss)));
%  ss = ssim(imread('A03_01C_sn2.png'),TargetImage);
%  subplot(235); imshow('A03_01C_sn2.png');     title(strcat('SN2 MM : ',num2str(ss)));
%   ss = ssim(imread('A03_01C_sn3.png'),TargetImage);
% 
%  subplot(236); imshow('A03_01C_sn3.png');    title(strcat('SN3 KH  : ',num2str(ss)));
%  set(gcf,'units','normalized','outerposition',[0 0 1 1]);
% 
% 
%  
%   
% SourceImage = imread('Images/A03_01D.png');
% TargetImage = imread('Images/H03_01D.png');
% 
%   figure,
%  subplot(231); imshow(SourceImage);   title('SourceImage');
%  subplot(232); imshow(TargetImage);   title('TargetImage');
% %  subplot(233); imshow("232");  title(ssimval);
%  ss = ssim(imread('A03_01D_sn1.png'),TargetImage);
%  subplot(234); imshow('A03_01D_sn1.png');    title(strcat('SN1 GAN  : ',num2str(ss)));
%  ss = ssim(imread('A03_01D_sn2.png'),TargetImage);
%  subplot(235); imshow('A03_01D_sn2.png');     title(strcat('SN2 MM : ',num2str(ss)));
%   ss = ssim(imread('A03_01D_sn3.png'),TargetImage);
% 
%  subplot(236); imshow('A03_01D_sn3.png');    title(strcat('SN3 KH  : ',num2str(ss)));
%  set(gcf,'units','normalized','outerposition',[0 0 1 1]);
% 
% 
%  
%  

%% Stain Normalisation using Macenko's Method

% disp('Stain Normalisation using Macenko''s Method');
% 
% [ NormMM ] = Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1, verbose);

% imwrite(NormMM,'A03_01D_sn2.png')
%% End of Demo
