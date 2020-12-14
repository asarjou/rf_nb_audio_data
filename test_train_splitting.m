close all
clear all
clc
%%
filename = 'accent-mfcc-data.csv'
M = readtable(filename) %Read in Data

M = table2cell(M)
Labels = M(:,1)
Labels = categorical(Labels)

oldcats = {'ES','GE', 'UK', 'IT', 'FR'};
New_Labels = mergecats(Labels,oldcats) %Merge the Columns in line with the research paper by Ma and Fokoue (2014)

New_Labels = grp2idx(New_Labels) %Convert Labels to numeric values

New_Labels = New_Labels - 1 %0 = Non-US, 1 = US Accent 

Features = cell2mat(M(:,2:13)) %Seperate the Features (X1-X12)

%M_nooutlier = rmoutliers(M, 'percentiles',[00.01 99.99]) %Remove only the most extreme values (5 values)

%M_nooutlier_labels = M_nooutlier(:,1); %extract target
%M_nooutlier = M_nooutlier(:,2:13); %Extract Features

rng('default') %For Reproducibility

[m, n] = size(Features);

idx = randperm(m); %Randomly order the data for random selection

trainingData = Features(idx(1:round(m*0.7)),:) ; 
trainingLabels = New_Labels(idx(1:round(m*0.7)),:) ; 

testingData = Features(idx(round(m*0.7)+1:end),:) ;
testingLabels = New_Labels(idx(round(m*0.7)+1:end),:) ;%Seperate into training and testing datasets, 70% training and 30% testing

writematrix(trainingData,'trainingData.csv') 
writematrix(trainingLabels,'trainingLabels.csv') 
writematrix(testingData,'testingData.csv') 
writematrix(testingLabels,'testingLabels.csv') 

