close all
clear all
clc

%%
trainingData = csvread('trainingData.csv')
trainingLabels = csvread('trainingLabels.csv')
testingData = csvread('testingData.csv')
testingLabels = csvread('testingLabels.csv')

%%
%NB Training

dist = repmat({'kernel'},1,12) ;
Kernel_smoo_type = 'normal';
Kernel_Width = 1;

sample_prior = [1-sum(trainingLabels)/length(trainingLabels), sum(trainingLabels)/length(trainingLabels)]; %Hand calculate the priors for the training data 

%%
rng = 'default';

%final_Mdl_NB = fitcnb(trainingData, trainingLabels,'ClassNames', {'0','1'}, 'DistributionNames', dist, 'Width', Kernel_Width, 'Kernel', Kernel_smoo_type, 'Prior', sample_prior) %Final Model Training on all training data

%save 'NB_final.mat' 'final_Mdl_NB' %Save the model for use later

load('NB_final.mat', 'final_Mdl_NB')
err_trainNB = resubLoss(final_Mdl_NB);

tic 
[final_Mdl_NB_test, score_nb] = predict(final_Mdl_NB, testingData);
time_pred = toc

err_testNB = loss(final_Mdl_NB, testingData, testingLabels)

final_Mdl_NB_test_db = str2double(final_Mdl_NB_test);
C = confusionchart(testingLabels, final_Mdl_NB_test_db)

C = C.NormalizedValues;

accuracy = 1 - err_testNB
accuracy_train = 1-err_trainNB
precision = C(2,2)/(C(2,2)+C(1,2)); %use confusion matrix in order to calculate precision 
recall = C(2,2)/(C(2,2)+C(2,1)); %use confusion matrix in order to calculate recall 
f1_score = (2*precision*recall)/(precision+recall); %use confusion matrix in order to calculate f1 score 

results_nb = [err_trainNB, err_testNB, accuracy_train*100, accuracy*100, precision*100, recall*100, f1_score*100]
%%
%RF Training

MinLeafSize = 1;
MaxNumSplits = 5000;
Trees = 50;

%%

rng = 'default';

%template = templateTree('MaxNumSplits',MaxNumSplits, 'MinLeafSize',MinLeafSize, 'Reproducible', true); %from mathworks page to produce template of decision tree -- play with values
%final_Mdl_RF = fitcensemble(trainingData, trainingLabels, 'Method', 'Bag','NumLearningCycles', Trees, 'Learners', template); % Train the Random Forest Model on the parameters
            
%save 'RF_final.mat' 'final_Mdl_RF' %Save the model for use later

load('RF_final.mat', 'final_Mdl_RF')

err_trainRF = resubLoss(final_Mdl_RF);

tic 
[final_Mdl_RF_test, score_rf] = predict(final_Mdl_RF, testingData);
time_pred = toc

err_testRF = loss(final_Mdl_RF, testingData, testingLabels)

C = confusionmat(testingLabels, final_Mdl_RF_test) %calculate confusion matrix 

accuracy = 1 - err_testRF
accuracy_train = 1 - err_trainRF
precision = C(2,2)/(C(2,2)+C(1,2)); %use confusion matrix in order to calculate precision 
recall = C(2,2)/(C(2,2)+C(2,1)); %use confusion matrix in order to calculate recall 
f1_score = (2*precision*recall)/(precision+recall); %use confusion matrix in order to calculate f1 score 

results_rf = [err_trainRF, err_testRF, accuracy_train*100, accuracy*100, precision*100, recall*100, f1_score*100]
%%

%to get AUC and ROC curve 
rng = 'default' 
%Adapted from https://www.mathworks.com/help/stats/perfcurve.html
[Xnb, Ynb, Tnb, AUCnb] = perfcurve(testingLabels, score_nb(:,2), 1); %Positive Class is US Accent, score(:,2) is contains the posterior probability for each sample being a US accent
[Xrf, Yrf, Trf, AUCrf] = perfcurve(testingLabels, score_rf(:,2), 1);
AUCnb
AUCrf

%%

%plot(Xnb,Ynb, Xrf, Yrf)
%title('ROC Curve')
%xlabel('False Positive Rate')
%ylabel('True Positive Rate')
%legend('Naive Bayes', 'Random Forests')

%%

