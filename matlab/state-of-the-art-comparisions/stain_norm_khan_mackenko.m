clc, clear all, close all;

media_url='/Users/xtarx/Documents/TUM/5th/Thesis/dataset/camelyon16/babak-exp/';
files = dir(strcat(media_url,'unstained/normal/*.png'));
TargetImage = (imread(strcat(media_url,'tumor18.png')));
output_url=media_url;

%% Display results of each method
verbose = 0;

n=0;
tic;
for file = files'
    
    %     if(n==3)
    %         break;
    %     end
    %
    
    file_name=file.name;
    n = n +1;
    
    
    fprintf('staining  %s.\n',file_name);
    %     src_url=strcat(media_url,'/unstained/tumor/',file_name)
    SourceImage = (imread(strcat(media_url,'/unstained/normal/',file_name)));
    
    disp('Stain Normalisation using Reinhard''s Method');
    
    [ NormRH ] = Norm( SourceImage, TargetImage, 'Reinhard', verbose );
    
    imwrite(NormRH,strcat(output_url,'Reinhard/normal/',file_name));
    
    
    %     imshow(SourceImage)
    %             try
    [ NormSM ] = Norm(SourceImage, TargetImage, 'SCD', [], 0);
    imwrite(NormSM,strcat(output_url,'Khan/normal/',file_name));
    
    %% Stain Normalisation using Macenko's Method
    %
    [ NormMM ] = Norm(SourceImage, TargetImage, 'Macenko', 255, 0.15, 1, verbose);
    %
    imwrite(NormMM,strcat(output_url,'Macenko/normal/',file_name));
    %
    %         catch
    %             warning('Problem using function.  Assigning a value of 0.');
    %         end
end


toc;