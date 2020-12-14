clear all 
close all
clc
%%
filename = 'RunGridSearch_NB_5_std.mat'
M = load(filename, '-mat')
%%
filt = table(M.total);
%filt_kernel = filt(filt.Var9 == {0.482608695652174,0.517391304347826},:);
%filt_kernel = filt
%filt_kernel.Var8 = categorical(filt_kernel.Var8);
samp_pri = filt(1:40, :);

samp_pri = splitvars(samp_pri, 'Var1');
samp_pri_kernel = (samp_pri(:,[10,3,11,1,8,13,2,4]));
samp_pri_kernel = sortrows(samp_pri_kernel,'Var1_11','ascend');
samp_pri_kernel = table2cell(samp_pri_kernel)
samp_pri_kernel_box = samp_pri_kernel(1:10,:);
samp_pri_kernel_epa = samp_pri_kernel(11:20,:);
samp_pri_kernel_norm = samp_pri_kernel(21:30,:)
samp_pri_kernel_tri = samp_pri_kernel(31:40,:)
%samp_pri1 = samp_pri(:,[2,7:8]);
 %= grp2idx(samp_pri.Var1_8)
%min_leaf_10 = filt(65:128, :);
%min_leaf_50 = filt(129:192, :);
%min_leaf_100 = filt(193:256, :);
%min_leaf_150 = filt(257:320, :);
%%
figure
errorbar((cell2mat(samp_pri_kernel_box(:,1))), cell2mat(samp_pri_kernel_box(:,2)),cell2mat(samp_pri_kernel_box(:,8)), 'r')
hold on
errorbar((cell2mat(samp_pri_kernel_tri(:,1))), cell2mat(samp_pri_kernel_tri(:,2)),cell2mat(samp_pri_kernel_tri(:,8)),'g')
errorbar((cell2mat(samp_pri_kernel_epa(:,1))), cell2mat(samp_pri_kernel_epa(:,2)),cell2mat(samp_pri_kernel_epa(:,8)),'b')
errorbar((cell2mat(samp_pri_kernel_norm(:,1))), cell2mat(samp_pri_kernel_norm(:,2)),cell2mat(samp_pri_kernel_norm(:,8)),'magenta')
plot((cell2mat(samp_pri_kernel_box(:,1))), cell2mat(samp_pri_kernel_box(:,4)),'r--')
hold on
plot((cell2mat(samp_pri_kernel_tri(:,1))), cell2mat(samp_pri_kernel_tri(:,4)),'g--')
plot((cell2mat(samp_pri_kernel_epa(:,1))), cell2mat(samp_pri_kernel_epa(:,4)),'b--')
plot((cell2mat(samp_pri_kernel_norm(:,1))), cell2mat(samp_pri_kernel_norm(:,4)),'magenta--')
legend({'box crossvalidation', 'tri crossvalidation', 'epa crossvalidation', 'norm crossvalidation', 'box resubstitution', 'tri resubstitution', 'epa resubstitution', 'norm resubstitution'})
xlabel('Kernel Smoothing Width')
ylabel('Error')
title('Crossvalidated Accuracy using Naive Bayes Classifier (Where Sample Prior is default)')
%%
figure
plot((cell2mat(samp_pri_kernel_box(:,1))), cell2mat(samp_pri_kernel_box(:,6)), 'r')
hold on
plot((cell2mat(samp_pri_kernel_tri(:,1))), cell2mat(samp_pri_kernel_tri(:,6)),'g')
plot((cell2mat(samp_pri_kernel_epa(:,1))), cell2mat(samp_pri_kernel_epa(:,6)),'b')
plot((cell2mat(samp_pri_kernel_norm(:,1))), cell2mat(samp_pri_kernel_norm(:,6)),'magenta')
legend({'box', 'tri', 'epa', 'norm'})
xlabel('Kernel Smoothing Width')
ylabel('Time to complete K-Fold Crossvalidation (s)')
title('Crossvalidation time using Naive Bayes Classifier (Where Sample Prior is default)')
