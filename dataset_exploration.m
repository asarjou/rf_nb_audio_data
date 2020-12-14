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

%%
%Normalise Features 

norm_Features(:,1) = (Features(:,1) - min(Features(:,1)))/(max(Features(:,1))-min(Features(:,1)));
norm_Features(:,2) = (Features(:,2) - min(Features(:,2)))/(max(Features(:,2))-min(Features(:,2)));
norm_Features(:,3) = (Features(:,3) - min(Features(:,3)))/(max(Features(:,3))-min(Features(:,3)));
norm_Features(:,4) = (Features(:,4) - min(Features(:,4)))/(max(Features(:,4))-min(Features(:,4)));
norm_Features(:,5) = (Features(:,5) - min(Features(:,5)))/(max(Features(:,5))-min(Features(:,5)));
norm_Features(:,6) = (Features(:,6) - min(Features(:,6)))/(max(Features(:,6))-min(Features(:,6)));
norm_Features(:,7) = (Features(:,7) - min(Features(:,7)))/(max(Features(:,7))-min(Features(:,7)));
norm_Features(:,8) = (Features(:,8) - min(Features(:,8)))/(max(Features(:,8))-min(Features(:,8)));
norm_Features(:,9) = (Features(:,9) - min(Features(:,9)))/(max(Features(:,9))-min(Features(:,9)));
norm_Features(:,10) = (Features(:,10) - min(Features(:,10)))/(max(Features(:,10))-min(Features(:,10)));
norm_Features(:,11) = (Features(:,11) - min(Features(:,11)))/(max(Features(:,11))-min(Features(:,11)));
norm_Features(:,12) = (Features(:,12) - min(Features(:,12)))/(max(Features(:,12))-min(Features(:,12)));

%%
xnames = {'X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12'};
figure
[h,ax] = gplotmatrix(Features,[],New_Labels,{'red','blue'}, 'o', [1 1], 'on', 'hist')
title('MFCC Speech Data (X1-X12)')
xlabel('MFCC DATA X1-12')
ylabel('MFCC DATA X1-12')

%%
%Sort into Categories 
idx_US = find(New_Labels);
idx_NonUS = find(~New_Labels);

US_non_norm = Features(idx_US,:);
Non_US_non_norm = Features(idx_NonUS, :);

US = norm_Features(idx_US,:);
Non_US = norm_Features(idx_NonUS, :) ;

%%
figure
tiledlayout(4,3);

nexttile
histogram(US(:,1), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,1), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X1')

nexttile
histogram(US(:,2), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,2), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X2')

nexttile
histogram(US(:,3), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,3), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X3')

nexttile
histogram(US(:,4), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,4), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X4')

nexttile
histogram(US(:,5), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,5), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X5')

nexttile
histogram(US(:,6), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,6), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X6')

nexttile
histogram(US(:,7), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,7), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X7')

nexttile
histogram(US(:,8), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,8), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X8')

nexttile
histogram(US(:,9), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,9), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X9')

nexttile
histogram(US(:,10), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,10), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X10')

nexttile
histogram(US(:,11), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,11), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X11')

nexttile
histogram(US(:,12), 'BinWidth', 0.05, 'Normalization', 'probability')
hold on
histogram(Non_US(:,12), 'BinWidth', 0.05, 'Normalization', 'probability')
title('X12')
lgd = legend({'Non-US Accent', 'US Accent'}, 'Orientation', 'horizontal')
lgd.Layout.Tile = 'south';
%%
%Plot histograms for the non-normalised data, should look similar
figure
tiledlayout(4,3);

nexttile
histogram(US_non_norm(:,1), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,1), 'BinWidth', 1, 'Normalization', 'probability')
title('X1')

nexttile
histogram(US_non_norm(:,2), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,2), 'BinWidth', 1, 'Normalization', 'probability')
title('X2')

nexttile
histogram(US_non_norm(:,3), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,3), 'BinWidth', 1, 'Normalization', 'probability')
title('X3')

nexttile
histogram(US_non_norm(:,4), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,4), 'BinWidth', 1, 'Normalization', 'probability')
title('X4')

nexttile
histogram(US_non_norm(:,5), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,5), 'BinWidth', 1, 'Normalization', 'probability')
title('X5')

nexttile
histogram(US_non_norm(:,6), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,6), 'BinWidth', 1, 'Normalization', 'probability')
title('X6')

nexttile
histogram(US_non_norm(:,7), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,7), 'BinWidth', 1, 'Normalization', 'probability')
title('X7')

nexttile
histogram(US_non_norm(:,8), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,8), 'BinWidth', 1, 'Normalization', 'probability')
title('X8')

nexttile
histogram(US_non_norm(:,9), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,9), 'BinWidth', 1, 'Normalization', 'probability')
title('X9')

nexttile
histogram(US_non_norm(:,10), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,10), 'BinWidth', 1, 'Normalization', 'probability')
title('X10')

nexttile
histogram(US_non_norm(:,11), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,11), 'BinWidth', 1, 'Normalization', 'probability')
title('X11')

nexttile
histogram(US_non_norm(:,12), 'BinWidth', 1, 'Normalization', 'probability')
hold on
histogram(Non_US_non_norm(:,12), 'BinWidth', 1, 'Normalization', 'probability')
title('X12')
lgd = legend({'Non-US Accent', 'US Accent'}, 'Orientation', 'horizontal')
lgd.Layout.Tile = 'south';

%Looks very similar to normalised version, as a first pass sanity check
%that normalisation has worked. Reason that there is some variation is
%because of bin widths (can solve this by adjusting binning settings but
%for our use this isn't helpful)
%%
%Adjusted from https://uk.mathworks.com/help/matlab/creating_plots/customize-pie-chart-labels.html

u = unique(New_Labels) %Find the Unique Groups
counts = histc(New_Labels, u); %Count how many are in each group

figure
p = pie(counts, '%.2f%%') %Plot the Pie Chart with the counts

pText = findobj(p,'Type','text'); %Adjust the Legend to show Percentage and String
percentValues = get(pText,'String'); 
txt = {'US Accent (value = 1): ';'Non-US Accent (value = 0): '}; %The String required
combinedtxt = strcat(txt,percentValues); %Join the String with the Text Value

pText(1).String = combinedtxt(1); 
pText(2).String = combinedtxt(2);

%No Data Imbalances In terms of class sizes (Both are equally sampled in
%the dataset due to how the experiment was carried out

%%
%Correlation is the Normalized Co-variance --> Use this to measure the
%magnitude of co-variance (extent to which random variables will vary
%together)
%Taken from Machine Learning Lecture 1 Written by Artur Garcez

corr_mat = corr(Features, 'Type', 'Pearson') %pairwise (Pearson's) correlation coefficient --> This assumes linearity of Data 
%clims = [-1 -0.6] %To show the highest correlation features
im = imagesc(corr_mat) %  <-- Add this as a parameter to see highest correlation features
set(gca,'XTick',[1:12]);
set(gca,'YTick',[1:12]);
xlabel('X1-X12')

title('Pearson Correlation Matrix of X1-X12')
colorbar
%Highest Pearson correlation (above 0.6) between X5&X10, X9&X4, X6&X7, X6&X11
%%
%Calculate Spearman's to check how it is different
corr_mat = corr(Features, 'Type', 'Spearman') %pairwise (Spearman's Rank) correlation coefficient --> Rank based, Captures monotonic relations between variables
%clims = [0.6 1] %To show the highest correlation features

im = imagesc(corr_mat) % , clims <-- Add this as a parameter to see highest correlation features
set(gca,'XTick',[1:12]);
set(gca,'YTick',[1:12]);
xlabel('X1-X12')

title("Spearman's Rank Correlation Matrix of X1-X12")
colorbar
%%

%Calculate Descriptive Statistics for all normalised and non normalised Features

skew_norm = skewness(norm_Features)
M_norm = mean(norm_Features)
kurt_norm = kurtosis(norm_Features)
med_norm = median(norm_Features)
sd_norm = std(norm_Features)

desc_norm = [skew_norm; M_norm; kurt_norm; med_norm; sd_norm]

skew = skewness(Features)
M = mean(Features)
kurt = kurtosis(Features)
med = median(Features)
sd = std(Features) 

desc = [skew; M; kurt; med; sd]

%%
%Calculate Descriptive Statistics between normalised groups (noting that
%this won't have much meaning due to being normalised -- Use for Visualisation)

skew_US_norm = skewness(US)
M_US_norm = mean(US)
kurt_US_norm = kurtosis(US)
med_US_norm = median(US)
sd_US_norm = std(US)

desc_US = [skew_US_norm; M_US_norm; kurt_US_norm; med_US_norm; sd_US_norm]

skew_Non_US_norm = skewness(Non_US)
M_Non_US_norm = mean(Non_US)
kurt_Non_US_norm = kurtosis(Non_US)
med_Non_US_norm = median(Non_US)
sd_Non_US_norm = std(Non_US)

desc_Non_US = [skew_Non_US_norm; M_Non_US_norm; kurt_Non_US_norm; med_Non_US_norm; sd_Non_US_norm]

%%
%Calculate Descriptive Statistics between non-normalised groups
skew_US_non_norm = skewness(US_non_norm)
M_US_non_norm = mean(US_non_norm)
kurt_US_non_norm = kurtosis(US_non_norm)
med_US_non_norm = median(US_non_norm)
sd_US_non_norm = std(US_non_norm)

desc_US_non_norm = [skew_US_non_norm', M_US_non_norm', kurt_US_non_norm',med_US_non_norm', sd_US_non_norm']

skew_Non_US_non_norm = skewness(Non_US_non_norm)
M_Non_US_non_norm = mean(Non_US_non_norm)
kurt_Non_US_non_norm = kurtosis(Non_US_non_norm)
med_Non_US_non_norm = median(Non_US_non_norm)
sd_Non_US_non_norm = std(Non_US_non_norm)

desc_Non_US_non_norm = [skew_Non_US_non_norm', M_Non_US_non_norm', kurt_Non_US_non_norm', med_Non_US_non_norm', sd_Non_US_non_norm']
 
%%
%Lets try to plot the values for the Non-Normalised Descriptive Statistics
figure
tiledlayout(1,2);

nexttile;
ylabels = {'X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12'};
xlabels = {'Skewness', 'Mean', 'Kurtosis', 'Median', 'Standard Deviation'}
h_US = heatmap(xlabels, ylabels, round(desc_US_non_norm,2))
title('Descriptive Statistics for US Accents')
axp = struct(h_US);       %taken from https://www.mathworks.com/matlabcentral/answers/378670-move-x-axis-labels-on-a-heatmap-to-the-top 
axp.Axes.XAxisLocation = 'top';

nexttile;
ylabels = {'X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12'};
xlabels = {'Skewness', 'Mean', 'Kurtosis', 'Median', 'Standard Deviation'}
h_non_US = heatmap(xlabels, ylabels, round(desc_Non_US_non_norm,2))
title('Descriptive Statistics for Non-US Accents')
axp = struct(h_non_US);       %taken from https://www.mathworks.com/matlabcentral/answers/378670-move-x-axis-labels-on-a-heatmap-to-the-top 
axp.Axes.XAxisLocation = 'top';