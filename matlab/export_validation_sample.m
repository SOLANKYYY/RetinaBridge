function export_validation_sample
% Select the ORIGINAL active MATLAB model and an authorised fundus image.
% Produces data for migration comparison; does not retrain or alter the model.
[mf,mp]=uigetfile('*.mat','Select the original dr_model.mat');
if isequal(mf,0),return;end
S=load(fullfile(mp,mf),'net');
[imf,imp]=uigetfile({'*.png;*.jpg;*.jpeg','Fundus image'},'Select an authorised test image');
if isequal(imf,0),return;end
I=imread(fullfile(imp,imf));
if size(I,3)==1,I=repmat(I,1,1,3);end
I=im2uint8(I(:,:,1:3));
[J,q]=preprocess_fundus(I);
[label,scores]=classify(S.net,J);
out=uigetdir(pwd,'Choose a folder for validation samples');
if isequal(out,0),return;end
out=tempname(out);mkdir(out);
imwrite(I,fullfile(out,'original.png'));
imwrite(J,fullfile(out,'enhanced.png'));
r=struct('label',char(string(label)),'scores',double(scores),...
    'classes',{cellstr(string(S.net.Layers(end).Classes))},'quality',q);
fid=fopen(fullfile(out,'reference.json'),'w');
cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(r));
fprintf('Validation sample saved at:\n%s\n',out);
end
