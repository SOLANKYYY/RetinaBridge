function runDir = train_dr_improved(dataRoot,epochs)
% Separate fine-tuning experiment for the original RetinaBridge ResNet-18.
% Place next to preprocess_fundus.m. Never overwrites models/dr_model.mat.
% Uses trainNetwork for compatibility with the original classify/Grad-CAM bridge.
% MATLAB runtime not available during authoring; send any full error text.
% Reference: https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
if nargin<2, epochs=12; end
validateattributes(epochs,{'numeric'},{'scalar','integer','positive'});
root=fileparts(fileparts(mfilename('fullpath')));
baseFile=fullfile(root,'models','dr_model.mat');
assert(isfile(baseFile),'Missing baseline models/dr_model.mat.');
S=load(baseFile);
assert(all(isfield(S,{'net','trainFiles','valFiles','testFiles','testLabels'})), ...
    'Baseline must contain net and original split metadata.');
assert(~isfield(S,'loraInfo'),'Active model is LoRA. Restore the original baseline first.');
assert(~isfield(S,'fineTuneInfo'),'Active model is already fine-tuned. Restore original baseline first.');
T=readtable(fullfile(dataRoot,'train.csv'),'TextType','string');
assert(all(ismember(["id_code","diagnosis"],string(T.Properties.VariableNames))), ...
    'CSV needs id_code and diagnosis columns.');
assert(numel(unique(T.id_code))==height(T),'Duplicate IDs in CSV.');
assert(all(ismember(T.diagnosis,0:4)),'Invalid grades.');
[tr,trainFiles,trainIDs]=makeData(S.trainFiles,T,dataRoot);
[va,valFiles,valIDs]=makeData(S.valFiles,T,dataRoot);
testIDs=getIDs(S.testFiles);
assert(isempty(intersect(trainIDs,valIDs)) && isempty(intersect(trainIDs,testIDs)) ...
    && isempty(intersect(valIDs,testIDs)),'Saved partitions overlap.');
counts=countcats(tr.Labels);assert(all(counts>0),'Missing training class.');
weights=sum(counts)./(5*counts);
assert(isequal(string(S.net.Layers(end).Classes(:)),string((0:4)')), ...
    'Expected model output order 0..4.');
rng(42);lg=layerGraph(S.net);layers=lg.Layers;trainableNames=strings(0,1);
for i=1:numel(layers)
    layer=layers(i);
    % Standard MATLAB ResNet-18 final residual block = res5b.
    active=startsWith(string(layer.Name),'res5b_') || strcmp(layer.Name,'dr_fc');
    changed=false;
    for prop={'WeightLearnRateFactor','BiasLearnRateFactor','ScaleLearnRateFactor','OffsetLearnRateFactor'}
        if isprop(layer,prop{1})
            layer.(prop{1})=double(active);changed=true;
        end
    end
    if changed
        lg=replaceLayer(lg,layer.Name,layer);
        if active,trainableNames(end+1,1)=string(layer.Name);end %#ok<AGROW>
    end
end
assert(any(startsWith(trainableNames,'res5b_')) && any(trainableNames=="dr_fc"), ...
    'Expected res5b residual block and dr_fc. Do not bypass: network names differ.');
% BN affine parameters are frozen. Running mean/variance still adapt in trainNetwork.
% This is freezing early learnable parameters, NOT freezing every network state.
lg=replaceLayer(lg,S.net.Layers(end).Name,classificationLayer( ...
    'Name',S.net.Layers(end).Name,'Classes',S.net.Layers(end).Classes, ...
    'ClassWeights',weights'));
runDir=tempname(fullfile(root,'models'));mkdir(runDir);
copyfile(baseFile,fullfile(runDir,'baseline.mat'));
checkpointDir=fullfile(runDir,'checkpoints');mkdir(checkpointDir);
fprintf('Run folder: %s\nTrainable layers:\n',runDir);disp(trainableNames);
fprintf('BN running statistics adapt; BN learned scale/offset remain frozen.\n');
fprintf('Measuring baseline validation performance...\n');
baseMetrics=measure(S.net,va,weights);reset(va);
aug=imageDataAugmenter('RandXReflection',true,'RandRotation',[-15 15]);
training=augmentedImageDatastore([224 224],tr,'DataAugmentation',aug);
opts=trainingOptions('adam','InitialLearnRate',1e-5, ...
    'MiniBatchSize',16,'MaxEpochs',epochs,'Shuffle','every-epoch', ...
    'ValidationData',va,'ValidationFrequency',max(1,floor(numel(trainFiles)/16)), ...
    'ValidationPatience',4,'OutputNetwork','best-validation', ...
    'ResetInputNormalization',false,'BatchNormalizationStatistics','moving', ...
    'CheckpointPath',checkpointDir,'ExecutionEnvironment','cpu', ...
    'Verbose',true,'Plots','training-progress');
[net,info]=trainNetwork(training,lg,opts);
reset(va);candidateMetrics=measure(net,va,weights);
% Verify early learnable parameters were not updated (running stats excluded).
old=S.net.Layers;new=net.Layers;
for i=1:numel(old)
    if any(string(old(i).Name)==trainableNames),continue;end
    j=find(strcmp({new.Name},old(i).Name));
    for prop={'Weights','Bias','Scale','Offset'}
        if isprop(old(i),prop{1})
            assert(isequaln(old(i).(prop{1}),new(j).(prop{1})), ...
                'Frozen parameter changed: %s / %s',old(i).Name,prop{1});
        end
    end
end
S.net=net;S.trainFiles=trainFiles;S.valFiles=valFiles;
S.fineTuneInfo=struct('learningRate',1e-5,'optimizer','Adam', ...
    'trainableLayers',trainableNames,'baselineValidation',baseMetrics, ...
    'candidateValidation',candidateMetrics,'selection','best validation loss', ...
    'batchNormRunningStats','moving; not frozen');
save(fullfile(runDir,'dr_model_candidate.mat'),'-struct','S','-v7.3');
save(fullfile(runDir,'training_info.mat'),'info','opts','-v7.3');
comparison=table(["Baseline";"Candidate"], ...
    [baseMetrics.accuracy;candidateMetrics.accuracy]*100, ...
    [baseMetrics.sensitivity;candidateMetrics.sensitivity]*100, ...
    [baseMetrics.specificity;candidateMetrics.specificity]*100, ...
    [baseMetrics.loss;candidateMetrics.loss], ...
    [baseMetrics.FN;candidateMetrics.FN], ...
    'VariableNames',{'Model','AccuracyPercent','SensitivityPercent', ...
    'SpecificityPercent','WeightedLoss','MissedReferable'});
disp(comparison);writetable(comparison,fullfile(runDir,'validation_comparison.csv'));
save(fullfile(runDir,'validation_details.mat'),'baseMetrics','candidateMetrics');
fprintf('Candidate saved separately. Active model is unchanged. No test images were read.\n');
fprintf('Send the comparison table before considering activation. Improvement is not guaranteed.\n');
end
function [ds,files,ids]=makeData(saved,T,dataRoot)
ids=getIDs(saved);assert(numel(unique(ids))==numel(ids),'Duplicate IDs in saved split.');
[ok,idx]=ismember(ids,T.id_code);assert(all(ok),'Saved IDs missing in CSV.');
files=cellstr(fullfile(dataRoot,'train_images',ids+'.png'));
assert(all(isfile(files)),'Images missing from train_images.');
ds=imageDatastore(files);ds.Labels=categorical(T.diagnosis(idx),0:4,{'0','1','2','3','4'});
ds.ReadFcn=@readImage;
end
function ids=getIDs(files)
files=cellstr(string(files));ids=strings(numel(files),1);
for i=1:numel(files),[~,name]=fileparts(files{i});ids(i)=string(name);end
end
function J=readImage(f),[J,~]=preprocess_fundus(imread(f));end
function m=measure(net,ds,weights)
reset(ds);[p,s]=classify(net,ds,'MiniBatchSize',16,'ExecutionEnvironment','cpu');
y=str2double(string(ds.Labels(:)));p=str2double(string(p(:)));
a=y>=2;b=p>=2;assert(any(a)&&any(~a),'Both binary classes required.');
m.TP=sum(a&b);m.TN=sum(~a&~b);m.FP=sum(~a&b);m.FN=sum(a&~b);
m.accuracy=mean(y==p);m.sensitivity=m.TP/(m.TP+m.FN);m.specificity=m.TN/(m.TN+m.FP);
n=numel(y);chosen=double(s(sub2ind(size(s),(1:n)',y+1)));
m.loss=-mean(weights(y+1).*log(max(chosen,realmin('double'))));
m.confusion=confusionmat(y,p,'Order',0:4);m.truth=y;m.predictions=p;
end
