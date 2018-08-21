
% Clear all previous data
clc, clear all, close all;


%% Display results of each method
verbose = 1;
%  Hd1 = mfilt.firdecim;
%     Hd2 = convert(Hd1,'firtdecim');

%% Load Source & Target images
% SourceImage = imread('Images/24jan/Source_small.png');
% TargetImage = imread('Images/Ref.png');


ssim_target=0;
ssim_reinhard=0;
ssim_vahadane=0;
ssim_macenko=0;
ssim_gan=0;

qssim_target=0;
qssim_reinhard=0;
qssim_vahadane=0;
qssim_macenko=0;
corr2_gan=0;

fssim_target=0;
fssim_reinhard=0;
fssim_vahadane=0;
fssim_macenko=0;
fssim_gan=0;

psnr_target=0;
psnr_reinhard=0;
psnr_vahadane=0;
psnr_macenko=0;
psnr_gan=0;
n=0;

A = zeros(10, 1, 'double');
base_dir='/Users/xtarx/Documents/TUM/5th/Thesis/exps/various-size/fixed-time/';
files = dir(strcat(base_dir,'H/*.png'));
which_size='6400';
for file = files'
    n = n +1;
    fprintf('N is %0.4f.\n',n);
    %     display(file.name)
    ref_url=strcat(base_dir,'H/',file.name);
    
    
    
    %     gan
    %     src_url=strcat(base_dir,'varying-size-exps/',which_size,'/',file.name);
    src_url=strcat(base_dir,which_size,'/',file.name);
    
    ref_img=imread(ref_url);
    src_img=imread(src_url);
    %         [ssimval, ssimmap] = ssim(src_img,ref_img);
    %     ssim_gan= ssim_gan + ssimval;
    %     fprintf('SSIM is %0.4f.\n',ssimval);
    %
    %         corrasdas= corr2(src_img,ref_img); % returns a scalar
    
    %corr2
%     R = corr2(rgb2gray(src_img),rgb2gray(ref_img));
%     A(n,1)=R;
%     fprintf('corr2 is %0.4f.\n',R);
%     corr2_gan = corr2_gan+ R;
    
    %
        [FSIM, FSIMc]=   FeatureSIM(src_url, ref_url);
    
        fssim_gan = fssim_gan+ FSIMc;
        A(n,1)=FSIMc;
        fprintf('FSIM is %0.4f.\n',FSIMc);
    
    %         [peaksnr, snr] = psnr(src_img, ref_img);
    %         psnr_gan = psnr_gan+ peaksnr;
    %         A(n,1)=peaksnr;
    %         fprintf('PSNR is %0.4f.\n',aapeaksnr);
    
    
    
    
end
csvwrite(strcat(base_dir,which_size,'/','varying_size_fssim_fixed_time',which_size,'.dat'),A)

fprintf('#of samples is %0.4f.\n',n);
% fprintf('Expirment Size is %0.4f.\n',which_size);
disp(which_size)
fprintf('The FSIM of gan  is %0.4f.\n',(fssim_gan/n));

% fprintf('The coor2 of %s  is %0.4f.\n',(corr2_gan/n));





