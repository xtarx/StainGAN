function [mqssim, qssim_map] = qssim(strRefImgPath, strDstImgPath,ch,sqrt_3, K, window, L)

%========================================================================
%QSSIM Index, Version 1.2
%Copyright(c) 2011 Amir Kolaman
%All Rights Reserved.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%This code was modified from the original ssim_index.m downloaded from 
%http://www.ece.uwaterloo.ca/~z70wang/research/ssim/
%which is the implementation of 
%Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality
%assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.
%__________________________________________________________________________
%
%This is an implementation of the algorithm for calculating the
%Quaternion Structural SIMilarity (QSSIM) index between two images. Please refer
%to the following paper:
%
%A. Kolaman, Orly Yadid-Pecht "Quaternion Structural Similarity a New Quality Index for Color Images"
%IEEE Transactios on Image Processing, vol. ??, no. ??, ??. 2011.
%
%Kindly report any suggestions or corrections to kolaman@bgu.ac.il
%
%To use this index you need to first download Quaternion Toolbox for Matlab at
%http://qtfm.sourceforge.net/
%and add it to Matlab's path
%----------------------------------------------------------------------
%% IMG1(N,N) while N an odd number SHOULD BE CHANGED 
%Input : (1) img1: the first image being compared 
%        (2) img2: the second image being compared
%%
%         (3) ch: the percent of upsizing chrominance sautration as to detect saturation changes better  
%         (4) sqrt_3: whether to use normilize by square root of 3
%        (5) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (6) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (7) L: dynamic range of the images. default: L = 255
%
%Output: (1) mqssim: the mean QSSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mqssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mqssim = 1.
%        (2) qqssim_map: the QSSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Color image Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mqssim qqssim_map] = qssim_index(img1, img2);
%
%Default gray scale image Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255, and
%   are gray scale (m,n,1) images 
%if sqrt_3=1 than use:
% img1_RGB(:,:,3)=img1./sqrt(3);
% img2_RGB(:,:,3)=img2./sqrt(3);
%else
% img1_RGB(:,:,3)=img1;
% img2_RGB(:,:,3)=img2;
% img1_RGB(:,:,2)=img1_RGB(:,:,3);
% img1_RGB(:,:,1)=img1_RGB(:,:,3);
% img2_RGB(:,:,2)=img2_RGB(:,:,3);
% img2_RGB(:,:,1)=img2_RGB(:,:,3);
%
%   [mqssim qqssim_map] = qssim_index(img1_RGB, img2_RGB);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mqssim qqssim_map] = qssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mqssim                        %Gives the mqssim value
%   imshow(max(0, qqssim_map).^4)  %Shows the QSSIM index map
%
%========================================================================
%% read image
img1 = imread(strRefImgPath);
img2 = imread(strDstImgPath);

%% making sure the input is in double
img1=double(img1)./256;

img2=double(img2)./256;

%% *******************checking  function input settings****************
if (nargin < 2 || nargin > 7)
   mqssim = -Inf;
   qssim_map = -Inf;
   display('parameters are not adaquate');
   return;
end

if (nargin == 2)
    ch=1;
    sqrt_3=1;
       window = fspecial('gaussian', 11, 1.5);	% creating 11x11 gaussian window 
   K(1) = 0.01;								      % default settings
   K(2) = 0.03;								      %
   L = 1;                                  %
end
    
if (size(img1) ~= size(img2))
   mqssim = -Inf;
   qssim_map = -Inf;
      display('images are different in size');
   return;
end

[M N] = size(img1(:,:,1));

if (nargin == 4)
   if ((M < 11) || (N < 11))
	   mqssim = -Inf;
	   qssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);	% creating 11x11 gaussian window 
   K(1) = 0.01;								      % default settings
   K(2) = 0.03;								      %
   L = 1;                                  %
  end

if (nargin == 5)
   if ((M < 11) || (N < 11))
	   mqssim = -Inf;
	   qssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L =1;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   mqssim = -Inf;
   		qssim_map = -Inf;
	   	return;
      end
   else
	   mqssim = -Inf;
   	qssim_map = -Inf;
	   return;
   end
end

if (nargin == 6)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   mqssim = -Inf;
	   qssim_map = -Inf;
      return
   end
   L = 1;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   mqssim = -Inf;
   		qssim_map = -Inf;
	   	return;
      end
   else
	   mqssim = -Inf;
   	qssim_map = -Inf;
	   return;
   end
end

if (nargin == 7)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   mqssim = -Inf;
	   qssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   mqssim = -Inf;
   		qssim_map = -Inf;
	   	return;
      end
   else
	   mqssim = -Inf;
   	qssim_map = -Inf;
	   return;
   end
end
%% ***********************beginning of the code***********************
   %$$$$$$$$$$$$$$$$$$$$  Dilating and chrominance channel of both $$$$$$$
   %%$$$$$$$$$$$$$$$$$$$$$$$$$$$       images by ch       $$$$$$$$$$$$$$$$$$$$$$$$$$$$   
 img1_L(:,:,3)=img1(:,:,1)/3+img1(:,:,2)/3+img1(:,:,3)/3;
img1_L(:,:,2)=img1_L(:,:,3);
img1_L(:,:,1)=img1_L(:,:,3);
img1_ch=img1-img1_L;
img1=img1_ch.*ch+img1_L;
% img1=img1_ch+img1_L./ch;

 img2_L(:,:,3)=img2(:,:,1)/3+img2(:,:,2)/3+img2(:,:,3)/3;
img2_L(:,:,2)=img2_L(:,:,3);
img2_L(:,:,1)=img2_L(:,:,3);
img2_ch=img2-img2_L;
img2=img2_ch.*ch+img2_L;
% img2=img2_ch+img2_L./ch;

%$$$$$$$$$$$$$$$$$$$$  Dilating and chrominance channel of both $$$$$$$
%%$$$$$$$$$$$$$$$$$$$$$$$$$$$       images by ch      $$$$$$$$$$$$$$$$$$$$$$$$$$$$   

% automatic downsampling
f = max(1,round(min(M,N)/256));
%downsampling by f
%use a simple low-pass filter 
if(f>1)
    lpf = ones(f,f);
    lpf = (1./(f*f))*lpf;
    img1 = imfilter(img1,lpf,'symmetric','same');
    img2 = imfilter(img2,lpf,'symmetric','same');

    img1 = img1(1:f:end,1:f:end,:);
    img2 = img2(1:f:end,1:f:end,:);
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
C1=quaternion(C1,0,0,0);
C2=quaternion(C2,0,0,0);
window = window/sum(sum(window)); %normalize to sum 1
img1_Q=img_to_Q(img1,sqrt_3);
img2_Q=img_to_Q(img2,sqrt_3);

mu1 = filter2_RGB(img1,window);%gaussian average of all the colors in image 1
mu2 = filter2_RGB(img2,window);%gaussian average of all the colors in image 2

mu1_Q = img_to_Q(mu1,sqrt_3);%convert F image to quaternion
mu2_Q = img_to_Q(mu2,sqrt_3);%convert G image to quaternion

mu1_sq_Q=mu1_Q.*conj(mu1_Q);%compute mu1_squared in quaternions
mu2_sq_Q=mu2_Q.*conj(mu2_Q);%compute mu2_squared in quaternions
mu1_mu2_Q=mu1_Q.*conj(mu2_Q);%compute the correlation between mu1 and mu2 in quaternions


img1_hue_sq_Q=img1_Q.*conj(img1_Q);%compute img1_hue_squared in quaternions
img2_hue_sq_Q=img2_Q.*conj(img2_Q);%compute img2_hue_squared in quaternions
img1_img2_hue_Q=img1_Q.*conj(img2_Q);%compute the correlation between img1_hue and img2_hue.


 sigma1_sq_Q = filter2_Q(img1_hue_sq_Q,window)-mu1_sq_Q;%average the hue squared with a gaussian
 sigma2_sq_Q = filter2_Q(img2_hue_sq_Q,window)-mu2_sq_Q ;%average the hue squared with a gaussian
 sigma12_Q = filter2_Q(img1_img2_hue_Q,window)-mu1_mu2_Q ;%average the hue squared with a gaussian
 
if (abs(C1) > 0 && abs(C2) > 0)
   qssim_map_Q = ((2*mu1_mu2_Q + C1).*(2*conj(sigma12_Q) + C2))./((mu1_sq_Q + mu2_sq_Q + C1).*(conj(sigma1_sq_Q + sigma2_sq_Q) + C2));
   qssim_map = abs(qssim_map_Q);
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
	denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   qssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   qssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   qssim_map(index) = numerator1(index)./denominator1(index);
end
mqssim = mean2(qssim_map);
end
function f_img=filter2_RGB(img,window)
%this function performs a gaussian average on an RGB image
f_img(:,:,1)   = filter2(window, img(:,:,1), 'valid');%gaussian average of red color in image 1
f_img(:,:,2)   = filter2(window, img(:,:,2), 'valid');%gaussian average of green color in image 1
f_img(:,:,3)   = filter2(window, img(:,:,3), 'valid');%gaussian average of blue color in image 1
end

function result=filter2_Q(f_Q,window)
% this function does a regular filter2 on every part of the quaternion
% matrix seperatly
temp(:,:,1)   = filter2(window, s(f_Q), 'valid');%gaussian average of red color in image 1
temp(:,:,2)   = filter2(window, x(f_Q), 'valid');%gaussian average of green color in image 1
temp(:,:,3)   = filter2(window, y(f_Q), 'valid');%gaussian average of blue color in image 1
temp(:,:,4)   = filter2(window, z(f_Q), 'valid');%gaussian average of blue color in image 1
result = convert(quaternion(temp(:,:,1), ...
                       temp(:,:,2), ...
                       temp(:,:,3),...
                       temp(:,:,4)), 'double'); 
                   
end

function img_Q=img_to_Q(img,sqrt_3)
% this function convert RGB image to quaternion space
%it converts the image to quaternion and normalizes it to 1
if sqrt_3==1
img_Q = convert(quaternion(img(:,:,1), ...
                       img(:,:,2), ...
                       img(:,:,3)), 'double') ./sqrt(3); 
else
img_Q = convert(quaternion(img(:,:,1), ...
                       img(:,:,2), ...
                       img(:,:,3)), 'double') ;     
end
end