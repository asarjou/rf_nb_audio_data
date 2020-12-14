close all
clear all
clc

%% 
trainingData = csvread('trainingData.csv') %Load in preprepared datasets
trainingLabels = csvread('trainingLabels.csv')
testingData = csvread('testingData.csv')
testingLabels = csvread('testingLabels.csv')

%%
%set parameters and initialise matrices for measurements
folds = [1:10];
Trees = [1,10, 50, 100, 500,1000,5000,10000];
MaxNumSplits = [1,10, 50, 100, 500,1000,5000,10000];
MinLeafSize = [1,10,50,100,150];
variables_to_sample = [1,3,6,9,12];

err_trainRF = []; %resubstitution error inside each CV
err_validRF = [] ;% Validation error inside each CV 
err_ooblossRF = []; % out of bag error inside each CV

mean_err_trainRF = [];
mean_err_validRF = [];
mean_err_ooblossRF = [];

accuracy = [];
precision = [];
recall = [];
f1_score = [];

mean_accuracy = []; 
mean_precision = [];
mean_recall = [];
mean_f1_score = [];
total = [];
%% 

rng = 'default'; %For reproducibility
foldindex = crossvalind('kfold', size(trainingData, 1), 10);

num = 1
tic
for minleaf = 1:size(MinLeafSize,2)
    for max_splits = 1:size(MaxNumSplits,2)
        for t = 1:size(Trees,2)
            count = 0;
            tic
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
            
            template = templateTree('MaxNumSplits',MaxNumSplits(max_splits), 'MinLeafSize',MinLeafSize(minleaf), 'Reproducible', true); %from mathworks page to produce template of decision tree -- play with values
            Mdl_RF = fitcensemble(training, labels_training, 'Method', 'Bag','NumLearningCycles', Trees(t), 'Learners', template); % Train the Random Forest Model on the parameters
            
            
            err_trainRF(i) =  resubLoss(Mdl_RF); %Find Resubstitution Loss 
            err_validRF(i) = (loss(Mdl_RF, validation, labels_validation)); %Find Validation Loss
            err_oobloss(i) = (oobLoss(Mdl_RF)); %Find Out of Bag Loss
            
            RF_pred = predict(Mdl_RF, validation); %predict based off of the validation set in order to get a confusion matrix 
            
            C = confusionmat(labels_validation, RF_pred); %calculate confusion matrix 
            %accuracy(i) = (C(1,1)+C(2,1))/(C(1,1)+C(1,2)+C(2,1)+C(2,2)); %use confusion matrix to calculate accuracy
            precision(i) = C(2,2)/(C(2,2)+C(1,2)); %use confusion matrix in order to calculate precision 
            recall(i) = C(2,2)/(C(2,2)+C(2,1)); %use confusion matrix in order to calculate recall 
            f1_score(i) = (2*precision(i)*recall(i))/(precision(i)+recall(i)); %use confusion matrix in order to calculate f1 score 
            count = count+1
         end
       
        time(num) = toc 
        mean_err_trainRF(num) = mean(err_trainRF) ; %Mean Resubstitution Loss
        std_err_trainRF(num) = std(err_trainRF); %Standard Deviation of Resubstitution Loss 
        mean_err_validRF(num) = mean(err_validRF);%Mean Validation Loss
        
        std_err_validRF(num) = std(err_validRF);%Standard Deviation of Validation Loss 
        mean_err_ooblossRF(num) = mean(err_oobloss); %Mean Out of Bag Loss
        
        std_err_ooblossRF(num) = std(err_oobloss);%Standard Deviation of Out of Bag Loss 
        
        mean_accuracy(num) = mean(1-err_validRF); 
        mean_precision(num) = mean(precision);
        mean_recall(num) = mean(recall);
        mean_f1_score(num) = mean(f1_score);
        
        total = [total ; mean_err_trainRF(num), std_err_trainRF(num), mean_err_validRF(num), std_err_validRF(num), mean_err_ooblossRF(num), std_err_ooblossRF(num), mean_accuracy(num), mean_precision(num), mean_recall(num), mean_f1_score(num), Trees(t), MinLeafSize(minleaf),MaxNumSplits(max_splits), num, time(num)]; %Save Outputs into a matrix that can be searched for lowest of various values and used for visualisation 
        num = num+1
        
        end
    end
end
toc
[value, ind_val] = min(total(:, 3)) %Checking for Lowest Validation Error

opt_trees_val = total(ind_val, 11) 
opt_minleafsize_val = total(ind_val, 12)
opt_maxnumsplits_val = total(ind_val, 13)

%view(Mdl_RF.Trained{1},'Mode','graph');

[value2, ind_f1] = min(total(:, 10)) %Checking for Lowest F1 Score
opt_trees_f1 = total(ind_f1, 11) 
opt_minleafsize_f1 = total(ind_f1, 12)
opt_maxnumsplits_f1 = total(ind_f1, 13)

save('GridSearch_RF_5_std.mat','total');
%%
%Testing Best Parameter Model
%rng = 'default'

%template = templateTree('MaxNumSplits',10000, 'MinLeafSize',1, 'Reproducible', true); %from mathworks page to produce template of decision tree -- play with values
%Mdl_RF_testing = fitcensemble(training, labels_training, 'Method', 'Bag','NumLearningCycles', 1000, 'Learners', template); % Train the Random Forest Model on the parameters

%tic
%test_results = predict(Mdl_RF_testing, testingData);
%toc

%tst_loss = loss(Mdl_RF_testing,testingData,testingLabels)
%C = confusionmat(testingLabels, test_results); %calculate confusion matrix 
%accuracy = (C(1,1)+C(1,2))/(C(1,1)+C(1,2)+C(2,1)+C(2,2)); %use confusion matrix to calculate accuracy
%precision = C(2,2)/(C(2,2)+C(1,2)); %use confusion matrix in order to calculate precision  
%recall = C(2,2)/(C(2,2)+C(2,1)); %use confusion matrix in order to calculate recall 
%f1_score = (2*precision*recall)/(precision+recall); %use confusion matrix in order to calculate f1 score 

%%
%Bayesian Optimisation
%This method is much quicker for Random Forests Optimisation but makes it
%harder to track the relationship between parameters and what effect
%changes may have
min_num_leaf = optimizableVariable('min_num_leaf',[1 150],'Type','integer');
max_num_splits = optimizableVariable('max_num_splits',[1 10000], 'Type', 'integer');
max_num_trees = optimizableVariable('max_num_trees',[1 10000], 'Type', 'integer');


c = cvpartition(trainingLabels,'Kfold',10);
fun = @(x)kfoldLoss(fitcensemble(trainingData, trainingLabels, 'Method', 'Bag','NumLearningCycles', x.max_num_trees, 'Learners', templateTree('MaxNumSplits',x.max_num_splits, 'MinLeafSize',x.min_leaf_size, 'Reproducible', true)));
results = bayesopt(fun,[min_num_leaf, max_num_splits, max_num_trees],'Verbose',2,...
    'AcquisitionFunctionName','expected-improvement-per-second-plus')