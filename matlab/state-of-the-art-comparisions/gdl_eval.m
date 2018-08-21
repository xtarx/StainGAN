clc, clear all, close all;

 media_url='/Users/xtarx/Documents/TUM/5th/Thesis/evaluation/results/eval_256_registered(2-2-18)epoch27/';
 file_name='0_0.png';
 
 A = rgb2gray( imread( strcat(media_url,'A/',file_name)));
 H = rgb2gray(imread( strcat(media_url,'H/',file_name)));
[Gmag, Gdir] = imgradient(A,'prewitt');
[Gmag2, Gdir2] = imgradient(H,'prewitt');
diff = abs(Gmag-Gmag2);
% figure
% imshowpair(Gmag, Gmag2, 'montage');
% title('Gradient Magnitude, Gmag (left), and Gradient Direction, Gdir (right), using Prewitt method')
% 

% figure
% imshowpair(Gmag, Gmag2, 'montage')
% title('Directional Gradients, Gx and Gy, using Sobel method')


figure(2);
subplot(1,3,1);
imagesc(Gmag);
title('Scanner A GD')
subplot(1,3,2);
imagesc(Gmag2);
title('Scanner H GD')
subplot(1,3,3);
imagesc(diff);
title('Difference image')
