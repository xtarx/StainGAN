
% Clear all previous data
clc, clear all, close all;


%% Load Source & Target images
fixed = imread('/Users/xtarx/Documents/TUM/5th/Thesis/dataset/mitosis@20x/png_resized/A/A03_00A.png');
moving = imread('/Users/xtarx/Documents/TUM/5th/Thesis/dataset/mitosis@20x/png_resized/H/H03_00A.png');

% imshowpair(fixed, moving,'Scaling','joint')
% 
% [optimizer, metric] = imregconfig('multimodal');
% optimizer.InitialRadius = 0.009;
% optimizer.Epsilon = 1.5e-4;
% optimizer.GrowthFactor = 1.01;
% optimizer.MaximumIterations = 300;
% tform = imregtform(moving, fixed, 'affine', optimizer, metric);
% movingRegistered = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));
% 
% figure
% imshowpair(fixed, movingRegistered,'Scaling','joint')

alignimage(fixed,moving)
function [ Rreg ] = alignimage( pic, refpic )
%to correct translational shifts, use imregister to compare shifted picture to a
%fixed reference image
close all;
pic=rgb2gray(pic);
refpic=rgb2gray(refpic);
% imshow(pic);
% figure;
% imshow(refpic);
% figure;

[optimizer, metric] = imregconfig('monomodal');

[ picRegistered, Rreg] = imregister(pic, refpic, 'rigid',optimizer, metric);
% imshowpair(refpic, picRegistered,'Scaling','joint');

figure;
imshow(ind2rgb(picRegistered, colormap));
x=1;
end
