#Comparing Naive Bayes and Random Forests Algorithms for Speaker Accent Prediction Using MFCCs
Author: Arman Sarjou
Date Created: 05/12/20

This directory contains a number of MatLab files required to perform a comparison of Random Forests and Naive Bayes models for Speaker Accent Prediction using MFCC Data

In this directory you should find the following files: 
1. Comparing Naive Bayes and Random Forests using MFCCs for speaker accent prediction.pdf
	- This pdf contains an A1 poster in which the study is outlined

3. dataset_exploration.m
	- This is a file in which exploratory data analysis of the original dataset is undertaken. This outputs a number of figures to describe the dataset

4. final_models_testing.m
	- This is a script which carries out reproducible final model testing. In order to run this file, ensure that NB_final.mat, RF_final.mat, trainingData.csv, trainingLabels.csv, testingData.csv, testingLabels.csv are all added to path within Matlab. This outputs a number of metrics into the command window. The values from these outputs have been used in the final poster and have been rounded to 3 significant figures. The order of values in thee results_rf and results_nb outputs are: resubstitution error, cross-validation error, resubstitution accuracy, cross-validation accuracy, precision, recall, f1 score

5. GridSearch_RF_5_std.mat
	- This is a .mat file which contains the results from final Grid Search for RF. 

6. NB_final.mat
	- This is a .mat file which contains the final NB model

7. NB_Optimisation.m 
	- This is a script which runs NB manual grid search for hyperparameter optimisation. In this script I show attempts at Bayesian Optimisation also.

8. NB_results_visualisation.m 
	- This is a script which takes RunGridSearch_NB_5_std.mat and visualises the results

9. RF_final.mat
	- This is a .mat file which contains the final RF model

10. RF_Optimisation
	- This is a script which runs RF manual grid search for hyperparameter optimisation. In this script I show attempts at Bayesian Optimisation also. 

11. RF_results_visualisation.m 
	- This is a script which takes GridSearch_RF_5_std.mat and visualises the results

12. RunGridSearch_NB_5_std.mat
	- This is a .mat filee which contains the results from the final NB Grid Search 

13. test_train_splitting.m 
	- This is a script which takes the original dataset and splits it into testing and training sets. Uses random permutations to randomly sort data and then split. 

14-17. testingData.csv , testingLabels.csv , trainingData.csv , trainingLabels.csv 
	These are the split data files which are as labelled
18. accent-mfcc-data.csv
	The original dataset 


These models files have all been produced in Matlab_R2020b using the Statistics and Machine Learning Toolbox, Bioinformatics Toolbox and Optimisation Toolbox. The final model testing should run with the Statistics and Machine Learning Toolbox. 
