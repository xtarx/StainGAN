function [ H, E, Bg ] = PseudoColourStains( Stains, M )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PseudoColourStains: Convert a grayscale stain channel into a colour image
% 
%
% Input:
% Stains    - The Deconvolved Stain channels.
% M         - (optional) Stain matrix.
%                        (default Ruifrok & Johnston H&E matrix)
% verbose   - (optional) Display results. (default 0)
%
%
% Output:
% H         - RGB image for First Stain (Usually the Hematoxylin Channel).
% E         - RGB image for Second Stain (Usually the Eosin Channel).
% Bg        - RGB image for Third Stain (Usually the Background Channel).
%
%
% Note: Input Stain channels must be in Optical Density (OD) space. If your
%       channels are not already in OD space (such as the channels returned
%       by the Colour Deconvolution C code) then please apply the following
%       conversion before using this function:
%
%                    Stains_OD = log(Io/double(Stains))
%
%       Where Io is the transmitted light intensity (typically 255 for a 
%       uint8 image).
%
%       M must be an 2x3 or 3x3 matrix, where rows corrrespond to the stain
%       vectors. If only two rows are given the third is estimated as a
%       cross product of the first two.
%
%
% Acknowledgements:
% This function is inspired by Mitko Veta's Stain Unmixing and Normalisation 
% code, which is available for download at Amida's Website:
%     http://amida13.isi.uu.nl/?q=node/69
%
%
% Copyright (c) 2013, Adnan Khan
% Department of Computer Science,
% University of Warwick, UK.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


[h, w, c] = size(Stains);

%% Sanity check
if c < 3
  errordlg('Must provide 3 stain channels');
  return;
end

%% Default Color Deconvolution Matrix proposed in Ruifork and Johnston [1]
if ~exist('M', 'var') || isempty(M)
   M = [   0.644211 0.716556 0.266844; 
           0.092789 0.954111 0.283111; 
       ]; 
end


%% Add third Stain vector, if only two stain vectors are provided. 
% This stain vector is obtained as the cross product of first two
% stain vectors. 
if size (M,1) < 3
    M = [M; cross(M(1, :), M(2, :))];
end

% Normalise the input so that each stain vector has a Euclidean norm of 1
M = (M./repmat(sqrt(sum(M.^2, 2)), [1 3]));

% the intensity of light entering the specimen
Io = 255;


%% Make stain concentration matrix by stacking the 3 channels
C = reshape(Stains, [], 3);

%% Generate pseudo-colour stain images
% We use the input stain matrix to determine the colour for each stain.
H = Io*exp(C(:, 1) * -M(1, :));
H = reshape(H, h, w, 3);
H = uint8(H);

E = Io*exp(C(:, 2) * -M(2, :));
E = reshape(E, h, w, 3);
E = uint8(E);

Bg = Io*exp(C(:, 3) * -M(3, :));
Bg = reshape(Bg, h, w, 3);
Bg = uint8(Bg);

end

