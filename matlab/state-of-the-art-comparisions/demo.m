%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A demonstration of the included Stain Normalisation methods.
%
%
% Adnan Khan and Nicholas Trahearn
% Department of Computer Science, 
% University of Warwick, UK.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Clear all previous data
clc, clear all, close all;


%% Display results of each method
verbose = 1;


%% Load Source & Target images
SourceImage = imread('Images/Source_small.png');
TargetImage = imread('Images/Ref.png');


%% Stain Normalisation using RGB Histogram Specification Method

disp('Stain Normalisation using RGB Histogram Specification Method');

[ NormHS ] = Norm( SourceImage, TargetImage, 'RGBHist', verbose );


%% Stain Normalisation using Reinhard Method

disp('Stain Normalisation using Reinhard''s Method');

[ NormRH ] = Norm( SourceImage, TargetImage, 'Reinhard', verbose );


%% Color Deconvolution using online available code with Standard Stain Matrix
% Original credit for C code to G.Landini

disp('Stain Separation using Lindini''s Color Deconvolution C code');

[s1, s2, s3] = colour_deconvolution(SourceImage, 'H&E');

% Get pseudo-coloured deconvolved channels
stainsc = log(255./(double(cat(3, s1, s2, s3))+0.0001));
[ Hc, Ec, Bgc ] = PseudoColourStains( stainsc, [] );


%% Color Deconvolution using Our Implementation with Standard Stain Matrix

disp(['Stain Separation using Our Implementation with Standard Stain'...
    ' Matrix ']);

% Get pseudo-coloured deconvolved channels
stains = Deconvolve( SourceImage, [], 0 );
[H, E, Bg] = PseudoColourStains( stains, [] );


%% Display comparative results of the two deconvolution implementations

figure,
subplot(231); imshow(Hc);   title('H (C code)');
subplot(232); imshow(Ec);   title('E (C code)');
subplot(233); imshow(Bgc);  title('Bg (C code)');
subplot(234); imshow(H);    title('H (Our Implementation)');
subplot(235); imshow(E);    title('E (Our Implementation)');
subplot(236); imshow(Bg);   title('Bg (Our Implementation)');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);


%% Stain Separation using Macenko's Image specific Stain Matrix for H&E 

disp(['Stain Separation using an Image specific Stain matrix '...
    'estimated using Macenko''s method']);

MacenkoMatrix = EstUsingMacenko( SourceImage );

Deconvolve( SourceImage, MacenkoMatrix, verbose );


%% Stain Normalisation using Macenko's Method

disp('Stain Normalisation using Macenko''s Method');

[ NormMM ] = Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1, verbose);



%% Stain Separation using Image specific Stain Matrix for H&E 

disp(['Stain Separation using an Image specific Stain matrix estimated '...
    'using the Stain Colour Descriptor Method']);

SCDMatrix = EstUsingSCD( SourceImage );

Deconvolve( SourceImage, SCDMatrix, verbose );


%% Stain Normalisation using the Non-Linear Spline Mapping Method

disp('Stain Normalisation using the Non-Linear Spline Mapping Method');

[ NormSM ] = Norm(SourceImage, TargetImage, 'SCD', [], verbose);


%%  Comparitive Results

disp(' Now Displaying all Results for comparison');

figure,
subplot(231); imshow(TargetImage);          title('Reference');
subplot(234); imshow(SourceImage);      title('Source');
subplot(232); imshow(NormHS);       title('Histogram Specification');
subplot(235); imshow(NormRH);       title('Reinhard');
subplot(233); imshow(NormMM);       title('Macenko');
subplot(236); imshow(NormSM);       title('SCD');
set(gcf,'units','normalized','outerposition',[0 0 1 1]);


%% End of Demo
disp('End of Demo');
