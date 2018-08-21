clc, clear all, close all;
%
media_url='/Users/xtarx/Documents/TUM/5th/Thesis/dataset/mitosis@20x/png_resized/';
files = dir(strcat(media_url,'/A/*.png'));
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.009;
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 300;
n=0;

% sort(files,'descend');
for file = files'
    file_name=file.name;
    fprintf('file_name is %s.\n',file_name);

    n = n +1;

    fprintf('N is %0.4f.\n',n);

    %     fprintf('filename is %s.\n',file_name);
    %     fprintf('fixed is %s.\n',strcat(media_url,'A/',file_name));
    fixed = (imread(strcat(media_url,'A/',file_name)));
    moving_name=strcat('H',file_name(2:end));
    moving = (imread(strcat(media_url,'H/',moving_name)));

    tform = imregtform(rgb2gray(moving), rgb2gray(fixed), 'rigid',optimizer, metric);
    moving = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));
    imwrite(moving,strcat(media_url,'H_registered/',moving_name));

end
% file_name='A05_05C.png';
% moving_name=strcat('H',file_name(2:end));
% 
% fixed = (imread(strcat(media_url,'A/',file_name)));
% moving = (imread(strcat(media_url,'H/',moving_name)));
% 
% diff = abs(moving-fixed);
% figure(1);
% subplot(1,3,1);
% imshow(fixed);
% title('Reference image')
% subplot(1,3,2);
% imshow(moving);
% title('moving image')
% subplot(1,3,3);
% imshow(diff);
% title('Difference image')
% 
% 
% moving = (imread(strcat(media_url,'H_registered/',moving_name)));
% diff = abs(moving-fixed);
% figure(2);
% subplot(1,3,1);
% imshow(fixed);
% title('Reference image')
% subplot(1,3,2);
% imshow(moving);
% title('moving image')
% subplot(1,3,3);
% imshow(diff);
% title('Difference image')

%
% %
% [optimizer, metric] = imregconfig('multimodal');
% optimizer.InitialRadius = 0.009;
% optimizer.Epsilon = 1.5e-4;
% optimizer.GrowthFactor = 1.01;
% optimizer.MaximumIterations = 300;
% tform = imregtform(rgb2gray(moving), rgb2gray(fixed), 'rigid',optimizer, metric);
% moving = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));
%
% imwrite(moving,'/Users/xtarx/Documents/TUM/5th/Thesis/dataset/mitosis@20x/png_resized/REGISTERED.png')

%
% diff = abs(moving-fixed);
% figure(2);
% subplot(1,3,1);
% imshow(fixed);
% title('Reference image')
% subplot(1,3,2);
% imshow(moving);
% title('moving image')
% subplot(1,3,3);
% imshow(diff);
% title('Difference image')