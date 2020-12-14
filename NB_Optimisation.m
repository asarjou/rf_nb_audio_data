close all
clear all
clc
%%
trainingData = csvread('trainingData.csv') %Load in preprepared datasets
trainingLabels = csvread('trainingLabels.csv')
testingData = csvread('testingData.csv')
testingLabels = csvread('testingLabels.csv')
%%
%Create arrays and hyperparameter ranges for hyperparameter tuning
folds = [1:10];
dist = repmat({'kernel'},1,12) ;%Using a Kernel/Normal/Multinomial Distribution
%dist = repmat({'normal'},1,12) ;%Using a Kernel/Normal/Multinomial Distribution
%dist = repmat({'mn'},1,12) ;%Using a Kernel/Normal/Multinomial Distribution
Kernel_smoo_type = {'box', 'epanechnikov', 'normal', 'triangle'}; %Various types of distributions
Kernel_Width = [1:10:100] ;

sample_prior = [1-sum(trainingLabels)/length(trainingLabels), sum(trainingLabels)/length(trainingLabels)]; %Hand calculate the priors for the training data
alt_priors =  [
    sample_prior
    0.9, 0.1
    0.8, 0.2
    0.7, 0.3
    0.6, 0.4
    0.5, 0.5
    0.4, 0.6
    0.3, 0.7
    0.2, 0.8
    0.1, 0.9
    ]; %Sample Prior Experimentation

err_trainNB = [];
err_validNB = [];% Validation error inside each CV 
err_ooblossNB = []; % out of bag error inside each CV

mean_err_trainNB = [];
mean_err_validNB = [];

precision = [];
recall = [];
f1_score = [];

mean_precision = [];
mean_recall = [];
mean_f1_score = [];
total = [];


%%
%Adapted from additional materials circulated by Oleksandr Galkin for Optimisation of Random
%Forests

rng = 'default'; %For reproducibility

foldindex = crossvalind('kfold', size(trainingData, 1), 10); %if KFold is
%to be run
num = 1
tic %Time how long it takes to optimize 

%Nested for loops for manual grid search
for altpri = 1:size(alt_priors,1)
    for width = 1:size(Kernel_Width,2)
        for smoo = 1:size(Kernel_smoo_type,2)
            count = 0;
            
            %for i = 1:500
            
            %hpartition = cvpartition(trainingLabels, 'Holdout', 0.2, 'Stratify', true); %Replicating Crossvalidation used in the paper -- Holdout Validation iterated and averaged over 500 times 
            
            %idxTrain = hpartition.training;
            
            %training = trainingData(idxTrain, :);
            %labels_training = trainingLabels(idxTrain, :);
            
            %idx_val = hpartition.test;
            
            %validation = trainingData(idx_val,:);
            %labels_validation = trainingLabels(idx_val, :);
         tic   
         for i = folds %for 10 Fold K-Fold CrossValidation
            validation = (trainingData(foldindex == i, :)); %Validation fold
            training = (trainingData(foldindex ~= i, :)); %Training Fold 
            labels_training = trainingLabels(foldindex ~= i, :); 
            labels_validation = trainingLabels(foldindex ==i, :);
            
            nbGau = fitcnb(training, labels_training,'ClassNames', {'0','1'}, 'DistributionNames', dist, 'Width', Kernel_Width(width), 'Kernel', char(Kernel_smoo_type(smoo)), 'Prior', alt_priors(altpri, :)); %Crossvalidated Model Production , 'Prior', alt_priors(k,:)

            
            err_trainNB(i) =  resubLoss(nbGau); %Find Resubstitution Loss 
            err_validNB(i) = (loss(nbGau, validation, labels_validation)); %Find Validation Loss
            
            nbGau_pred = predict(nbGau, validation); %predict based off of the validation set in order to get a confusion matrix 
            nbGau_pred_double = str2double(nbGau_pred);
            C = confusionchart(labels_validation, nbGau_pred_double); %calculate confusion matrix 
            C = C.NormalizedValues;
          %  accuracy(i) = (C(1,1)+C(2,2))/(C(1,1)+C(1,2)+C(2,1)+C(2,2)); %use confusion matrix to calculate accuracy
            precision(i) = C(2,2)/(C(2,2)+C(1,2)); %use confusion matrix in order to calculate precision 
            recall(i) = C(2,2)/(C(2,2)+C(2,1)); %use confusion matrix in order to calculate recall 
            f1_score(i) = (2*precision(i)*recall(i))/(precision(i)+recall(i)); %use confusion matrix in order to calculate f1 score 
            count = count+1
            
         end
        time(num) = toc
        mean_err_trainNB(num) = mean(err_trainNB) ; %Mean Resubstitution Loss
        std_err_trainNB(num) = std(err_trainNB); %Standard Deviation of Resubstituted Loss 
        
        mean_err_validNB(num) = mean(err_validNB); %Mean Validation Loss
        std_err_validNB(num) = std(err_validNB); %Standard Deviation of Validated Loss 
        
        mean_accuracy(num) = mean(1-err_validNB); %Crossvalidated Accuracy
        mean_precision(num) = mean(precision); %Crossvalidated Precision
        mean_recall(num) = mean(recall); %Crossvalidated Recall
        mean_f1_score(num) = mean(f1_score); %Crossvalidated F score
        total = [total ; mean_err_trainNB(num), std_err_trainNB(num), mean_err_validNB(num), std_err_validNB(num), mean_accuracy(num), mean_precision(num), mean_recall(num), mean_f1_score(num), dist(1),Kernel_Width(width),Kernel_smoo_type(smoo),alt_priors(altpri, :), time(num),num]; %Save Outputs into a matrix that can be searched for lowest of various values and used for visualisation 
        num = num+1
        
        end
    end
end
toc
%%
eval_metrics = total(:,1:8)
[value, ind_val] = (min((cell2mat(eval_metrics(:,3))))) %Checking for Lowest Cross-Validation Error

opt_dist_val = total(ind_val, 9) 
opt_kern_width_val = total(ind_val, 10)
opt_smoo_val = total(ind_val,11)

%view(Mdl_RF.Trained{1},'Mode','graph');

[value2, ind_f1] = (min(cell2mat(eval_metrics(:, 8)))) %Checking for Lowest F1 Score
opt_dist_f1 = total(ind_f1, 9) 
opt_kern_width_f1 = total(ind_f1, 10)
opt_smoo_f1 = total(ind_f1, 11)

save('RunGridSearch_NB_5_std.mat','total');
%This method of Cross-validation using hold out took 38 hours to optimise my model for a
%method which is supposed to be relatively quick (Naive Bayes). This would
%be unfeasible for a Random Forests Optimisation. Bayesian
%Optimisation was also tried. Results can be seen in
%FirstRunGridSearch_NB.mat
%%
%Bayesian Optimisation
%This method is much quicker but comes with the downside of having less
%hyperparameters to optimise... could be used as a first pass at
%optimisation which is then run through the manual grid search to observe
%effect of variations in sample prior
kernel_smoo_Type = optimizableVariable('kernel_smoo_Type',{'box', 'epanechnikov', 'normal', 'triangle'},'Type','categorical');
kernel_width = optimizableVariable('kernel_width',[1*10-6 100], 'Type', 'real');
%pri = optimizableVariable('pri', {'empirical', 'uniform'}, 'Type', 'categorical');
%kernel_type = optimizableVariable('kernel_type',{'kernel',
%'mn'},'Type','categorical'); %wouldn't work

c = cvpartition(trainingLabels,'Kfold',10);
fun = @(x)kfoldLoss(fitcnb(trainingData,trainingLabels,'CVPartition',c,'ClassNames', {'0','1'},'DistributionNames', dist, 'Width', x.kernel_width, 'Kernel', char(x.kernel_smoo_Type)));
results = bayesopt(fun,[kernel_smoo_Type, kernel_width],'Verbose',2,...
    'AcquisitionFunctionName','expected-improvement-per-second-plus')
%%
%Nested for loops for manual grid search of sample priors ONLY (to test
%after results from Bayesian Optimisation as Bayesian Optimisation wouldn't
%allow altering of priors) 

foldindex = crossvalind('kfold', size(trainingData, 1), 10); %if KFold is to be used

num = 1 
total = [];
tic
for altpri = 1:size(alt_priors,1)
            count = 0;
            %for i = 1:500
            
            %hpartition = cvpartition(trainingLabels, 'Holdout', 0.2, 'Stratify', true); %Replicating Crossvalidation used in the paper -- Holdout Validation iterated and averaged over 500 times 
            
            %idxTrain = hpartition.training;
            
            %training = trainingData(idxTrain, :);
            %labels_training = trainingLabels(idxTrain, :);
            
            %idx_val = hpartition.test;
            
            %validation = trainingData(idx_val,:);
            %labels_validation = trainingLabels(idx_val, :);
            
         for i = folds %for 10 Fold K-Fold CrossValidation
             
            validation = (trainingData(foldindex == i, :)); %Validation fold
            training = (trainingData(foldindex ~= i, :)); %Training Fold 
            labels_training = trainingLabels(foldindex ~= i, :); 
            labels_validation = trainingLabels(foldindex ==i, :);
            
            nbGau = fitcnb(training, labels_training,'ClassNames', {'0','1'}, 'DistributionNames', dist, 'Width', 92.725, 'Kernel', 'triangle', 'Prior', alt_priors(altpri, :)); %Crossvalidated Model Production , 'Prior', alt_priors(k,:)

            
            err_trainNB(i) =  resubLoss(nbGau); %Find Resubstitution Loss 
            err_validNB(i) = (loss(nbGau, validation, labels_validation)); %Find Validation Loss
            
            nbGau_pred = predict(nbGau, validation); %predict based off of the validation set in order to get a confusion matrix 
            nbGau_pred_double = str2double(nbGau_pred);
            C = confusionchart(labels_validation, nbGau_pred_double); %calculate confusion matrix 
            C = C.NormalizedValues;
            accuracy(i) = (C(1,1)+C(1,2))/(C(1,1)+C(1,2)+C(2,1)+C(2,2)); %use confusion matrix to calculate accuracy
            precision(i) = C(2,2)/(C(2,2)+C(1,2)); %use confusion matrix in order to calculate precision 
            recall(i) = C(2,2)/(C(2,2)+C(2,1)); %use confusion matrix in order to calculate recall 
            f1_score(i) = (2*precision(i)*recall(i))/(precision(i)+recall(i)); %use confusion matrix in order to calculate f1 score 
            count = count+1
            
            end
       
        mean_err_trainNB_pri(num) = mean(err_trainNB) ; %Mean Resubstitution Loss
        mean_err_validNB_pri(num) = mean(err_validNB); %Mean Validation Loss
        
        mean_accuracy_pri(num) = mean(accuracy); 
        mean_precision_pri(num) = mean(precision);
        mean_recall_pri(num) = mean(recall);
        mean_f1_score_pri(num) = mean(f1_score);
        total = [total ; mean_err_trainNB_pri(num), mean_err_validNB_pri(num), mean_precision_pri(num), mean_recall_pri(num), mean_f1_score_pri(num), dist(1),alt_priors(altpri,:), num]; %Save Outputs into a matrix that can be searched for lowest of various values and used for visualisation 
        num = num+1
        
end
toc
%%
save('GridSearchNB_sampleprior','total')