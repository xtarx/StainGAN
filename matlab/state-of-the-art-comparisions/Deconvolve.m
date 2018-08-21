function [ DCh, M ] = Deconvolve( I, M, verbose )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deconvolve: Deconvolution of an RGB image into its constituent stain
% channels
% 
%
% Input:
% I         - RGB input image.
% M         - (optional) Stain matrix. 
%                        (default Ruifrok & Johnston H&E matrix)
% verbose   - (optional) Display results. (default 0)
%
%
% Note: M must be an 2x3 or 3x3 matrix, where rows corrrespond to the stain
%       vectors. If only two rows are given the third is estimated as a
%       cross product of the first two.
%
%
% Output:
% DCh       - Deconvolved Channels concatatenated to form a stack. 
%             Each channel is a double in Optical Density space.
% M         - Stain matrix.
%
%
% References:
% [1] AC Ruifrok, DA Johnston. "Quantification of histochemical staining by
%     color deconvolution". Analytical & Quantitative Cytology & Histology,
%     vol.23, no.4, pp.291-299, 2001.
%
%
% Acknowledgements:
% This function is inspired by Mitko Veta's Stain Unmixing and Normalisation 
% code, which is available for download at Amida's Website:
%     http://amida13.isi.uu.nl/?q=node/69
%
%
% Example:
%           I = imread('hestain.png');
%           [ DCh, H, E, Bg, M ] = Deconvolve( I, [], 1);
%
%
% Copyright (c) 2013, Adnan Khan
% Department of Computer Science,
% University of Warwick, UK.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run in DEMO Mode
if nargin<1
    I = imread('hestain.png');
end

% Convert to double
I = double(I);

%% Sanity check

[h, w, c] = size(I);

% Image must be RGB
if c<3
    error('Image must be RGB'); 
elseif c>3 
    I = I(:,:,1:3);
end


%% Display results or not?
if ~exist('verbose', 'var') || isempty(verbose)
   verbose = 0; 
end


%% Default Color Deconvolution Matrix proposed in Ruifork and Johnston
if ~exist('M', 'var') || isempty(M)
   M = [   0.644211 0.716556 0.266844; 
           0.092789 0.954111 0.283111; 
       ];
end

%% Add third Stain vector, if only two stain vectors are provided. 
% This stain vector is obtained as the cross product of first two
% stain vectors 
if size (M,1) < 3
    M = [M; cross(M(1, :), M(2, :))];
end

% Normalise the input so that each stain vector has a Euclidean norm of 1
M = (M./repmat(sqrt(sum(M.^2, 2)), [1 3]));


%% MAIN IMPLEMENTATION OF METHOD

% the intensity of light entering the specimen (see section 2a of [1])
Io = 255;

% Vectorize
J = reshape(I, [], 3);

% calculate optical density
OD = -log((J+1)/Io);
Y = reshape(OD, [], 3);

% determine concentrations of the individual stains
% M is 3 x 3,  Y is N x 3, C is N x 3
C = Y / M;
%C = Y * pinv(M);

% Stack back deconvolved channels
DCh = reshape(C, h, w, 3);

%% VISUALISATION
% Display pseudo coloured version of results if verbose mode is true
if verbose,
    [ H, E, Bg ] = PseudoColourStains(DCh, M);
    
    figure,
    subplot(141); imshow(uint8(I)); title('Source');
    subplot(144); imshow(Bg); title('Background');
    subplot(142); imshow(H); title('Haematoxylin');
    subplot(143); imshow(E); title('Eosin');
end

