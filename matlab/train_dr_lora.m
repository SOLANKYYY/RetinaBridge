function runDir=train_dr_lora(dataRoot,epochs,rankValue)
% Head-only LoRA on the user's saved DR ResNet-18. Frozen backbone and bias.
% No GAN, no optimizer change. Adam updates A and B only.
if nargin<2,epochs=30;end
if nargin<3,rankValue=2;end
validateattributes(epochs,{'numeric'},{'scalar','integer','positive'});
validateattributes(rankValue,{'numeric'},{'scalar','integer','>=',1,'<=',4});
root=fileparts(fileparts(mfilename('fullpath')));
baseFile=fullfile(root,'models','dr_model.mat');
assert(isfile(baseFile),'Train the baseline first: models/dr_model.mat is missing.');
S=load(baseFile); required={'net','trainFiles','valFiles','testFiles','testLabels'};
assert(all(isfield(S,required)),'Baseline must contain saved train/validation/test split.');
L=S.net.Layers; idx=find(strcmp({L.Name},'dr_fc'));
assert(numel(idx)==1,'Expected the original dr_fc grading layer.');
fc=L(idx); classes=string(L(end).Classes(:));
assert(isequal(classes,string((0:4)')),'Expected output class order 0,1,2,3,4.');
C=S.net.Connections; pre=string(C.Source(string(C.Destination)=="dr_fc"));
assert(numel(pre)==1,'Expected one feature input to dr_fc.');
W0=double(fc.Weights); b0=double(fc.Bias); b0=b0(:);
assert(size(W0,1)==5,'Expected five output classes.');
T=readtable(fullfile(dataRoot,'train.csv'),'TextType','string');
assert(all(ismember(["id_code","diagnosis"],string(T.Properties.VariableNames))),'CSV needs id_code and diagnosis.');
assert(numel(unique(T.id_code))==height(T),'Duplicate CSV ids.');
assert(all(ismember(T.diagnosis,0:4)),'Invalid grade labels.');
[trainFiles,yTrain]=resolveFiles(S.trainFiles,T,dataRoot);
[valFiles,yVal]=resolveFiles(S.valFiles,T,dataRoot);
% Test paths/labels are retained, but no test images are opened during training.
assert(isempty(intersect(trainFiles,valFiles)),'Train/validation overlap.');
assert(isempty(intersect(baseNames(trainFiles),baseNames(S.testFiles))),'Train/test overlap.');
assert(isempty(intersect(baseNames(valFiles),baseNames(S.testFiles))),'Validation/test overlap.');
counts=accumarray(yTrain,1,[5 1]); assert(all(counts>0),'Missing training class.');
cw=sum(counts)./(5*counts);
runDir=tempname(fullfile(root,'models')); mkdir(runDir);
copyfile(baseFile,fullfile(runDir,'baseline.mat'));
fprintf('Run saved to: %s\nExtracting frozen training features (CPU)...\n',runDir);
X=features(S.net,trainFiles,pre); fprintf('Extracting validation features...\n');
V=features(S.net,valFiles,pre);
assert(size(X,1)==size(W0,2),'Feature width does not match classifier.');
% Verify that cached features reproduce the existing model before training.
D=imageDatastore(valFiles(1:min(8,numel(valFiles))));D.ReadFcn=@readImage;
[~,ref]=classify(S.net,D,'ExecutionEnvironment','cpu');
P=prob(W0*V(:,1:size(ref,1))+b0);
assert(max(abs(P'-double(ref)),[],'all')<1e-4,'Feature/head equivalence check failed.');
rng(42); alpha=rankValue; scale=alpha/rankValue;
A=randn(rankValue,size(W0,2))*0.01; B=zeros(5,rankValue);
ma=zeros(size(A));va=ma;mb=zeros(size(B));vb=mb;iteration=0;
lr=1e-4;batchSize=64;patience=5;bad=0;
[bestLoss,baseAcc]=metrics(W0,b0,V,yVal,cw);
bestA=A;bestB=B;bestEpoch=0;
history=[0 bestLoss baseAcc];
config=struct('method','head-only LoRA','rank',rankValue,'alpha',alpha,'learningRate',lr,'batchSize',batchSize,'patience',patience,'seed',42,'optimizer','Adam','featureLayer',pre,'trainableParameters',numel(A)+numel(B));
save(fullfile(runDir,'best_adapter.mat'),'bestA','bestB','bestLoss','bestEpoch','W0','b0','config');
fprintf('Baseline validation: weighted loss %.5f, accuracy %.2f%%\n',bestLoss,baseAcc*100);
for epoch=1:epochs
 order=randperm(numel(yTrain));
 for start=1:batchSize:numel(order)
  ids=order(start:min(start+batchSize-1,numel(order)));x=X(:,ids);y=yTrain(ids);n=numel(ids);
  P=prob((W0+scale*B*A)*x+b0);
  target=zeros(5,n);target(sub2ind([5 n],y(:)',1:n))=1;
  weights=reshape(cw(y),1,[]);G=(P-target).*weights/n;
  gB=scale*G*(A*x)';gA=scale*B'*G*x';iteration=iteration+1;
  [A,ma,va]=step(A,gA,ma,va,iteration,lr);
  [B,mb,vb]=step(B,gB,mb,vb,iteration,lr);
 end
 [loss,acc]=metrics(W0+scale*B*A,b0,V,yVal,cw);
 assert(isfinite(loss),'Nonfinite loss; stop and inspect data.');
 history(end+1,:)=[epoch loss acc]; %#ok<AGROW>
 fprintf('Epoch %d/%d | validation weighted loss %.5f | accuracy %.2f%%\n',epoch,epochs,loss,100*acc);
 if loss<bestLoss-1e-5
  bestLoss=loss;bestA=A;bestB=B;bestEpoch=epoch;bad=0;
  save(fullfile(runDir,'best_adapter.mat'),'bestA','bestB','bestLoss','bestEpoch','W0','b0','config');
 else,bad=bad+1;end
 save(fullfile(runDir,'last_checkpoint.mat'),'A','B','ma','va','mb','vb','iteration','epoch','history','config');
 if bad>=patience,fprintf('Early stopping: validation loss did not improve for %d epochs.\n',patience);break;end
end
% Merge LoRA into a standard FC layer: existing classify/Grad-CAM bridge can load it.
fc.Weights=cast(W0+scale*bestB*bestA,'like',fc.Weights);
g=replaceLayer(layerGraph(S.net),'dr_fc',fc); net=assembleNetwork(g);
[~,merged]=classify(net,D,'ExecutionEnvironment','cpu');
expected=prob((W0+scale*bestB*bestA)*V(:,1:size(merged,1))+b0);
assert(max(abs(expected'-double(merged)),[],'all')<1e-4,'Merged export equivalence failed.');
S.net=net;S.trainFiles=trainFiles;S.valFiles=valFiles;
S.loraInfo=config;S.loraInfo.bestEpoch=bestEpoch;S.loraInfo.bestValidationLoss=bestLoss;
save(fullfile(runDir,'dr_model_lora.mat'),'-struct','S','-v7.3');
writetable(array2table(history,'VariableNames',{'epoch','validationWeightedLoss','validationAccuracy'}),fullfile(runDir,'validation_history.csv'));
fprintf('Saved merged model: %s\nBest epoch: %d (0 means baseline retained).\n',fullfile(runDir,'dr_model_lora.mat'),bestEpoch);
fprintf('Your active models/dr_model.mat is unchanged. No test images were evaluated.\n');
end
function [files,y]=resolveFiles(old,T,dataRoot)
ids=baseNames(old);[ok,where]=ismember(ids,T.id_code);assert(all(ok),'Saved split ids missing from train.csv.');
files=cellstr(fullfile(dataRoot,'train_images',ids+'.png'));assert(all(isfile(files)),'Actual images missing at new dataset path.');
y=double(T.diagnosis(where))+1;y=y(:);
end
function ids=baseNames(files)
ids=strings(numel(files),1);
for i=1:numel(files),[~,n]=fileparts(files{i});ids(i)=string(n);end
end
function X=features(net,files,layer)
D=imageDatastore(files);D.ReadFcn=@readImage;
X=double(activations(net,D,char(layer),'OutputAs','rows','MiniBatchSize',16,'ExecutionEnvironment','cpu'))';
end
function J=readImage(f),[J,~]=preprocess_fundus(imread(f));end
function P=prob(Z),Z=Z-max(Z,[],1);P=exp(Z);P=P./sum(P,1);end
function [loss,acc]=metrics(W,b,X,y,cw)
Z=W*X+b;Z=Z-max(Z,[],1);logP=Z-log(sum(exp(Z),1));n=numel(y);
loss=-mean(cw(y(:)).*reshape(logP(sub2ind(size(logP),y(:)',1:n)),[],1));
[~,p]=max(Z,[],1);acc=mean(p(:)==y(:));
end
function [p,m,v]=step(p,g,m,v,t,lr)
m=.9*m+.1*g;v=.999*v+.001*(g.^2);
p=p-lr*(m/(1-.9^t))./(sqrt(v/(1-.999^t))+1e-8);
end
