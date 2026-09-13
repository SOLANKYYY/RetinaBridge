function activate_dr_lora(runDir)
% Explicitly activate a merged model for the original website; preserve old model.
root=fileparts(fileparts(mfilename('fullpath')));
source=fullfile(runDir,'dr_model_lora.mat');assert(isfile(source),'Merged LoRA model not found.');
S=load(source,'net','loraInfo');assert(isfield(S,'net') && isfield(S,'loraInfo'),'Not a LoRA export.');
assert(S.loraInfo.bestEpoch>0,'No validation improvement was found. Keep your baseline.');
dest=fullfile(root,'models','dr_model.mat');
backup=[tempname(fullfile(root,'models')) '_before_lora.mat'];
if isfile(dest),copyfile(dest,backup);fprintf('Previous active model backed up: %s\n',backup);end
copyfile(source,dest);fprintf('LoRA model activated. Refresh the website. No server code changes needed.\n');
end
