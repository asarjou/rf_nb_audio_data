clear all 
close all
clc
%%
filename = 'GridSearch_RF_5_std.mat'
M = load(filename, '-mat')
%%
filt = table(M.total);
%filt_kernel = filt(filt.Var9 == {0.482608695652174,0.517391304347826},:);
%filt_kernel = filt
%filt_kernel.Var8 = categorical(filt_kernel.Var8);
min_leaf_1 = filt(1:64, :);
min_leaf_10 = filt(65:128, :);
min_leaf_50 = filt(129:192, :);
min_leaf_100 = filt(193:256, :);
min_leaf_150 = filt(257:320, :);

min_leaf_1 = splitvars(min_leaf_1, 'Var1');
min_leaf_10 = splitvars(min_leaf_10, 'Var1');
min_leaf_50 = splitvars(min_leaf_50, 'Var1');
min_leaf_100 = splitvars(min_leaf_100, 'Var1');
min_leaf_150 = splitvars(min_leaf_150, 'Var1');
%idx_tri = samplepri.Var8 == 'triangle';
%idx_box = samplepri.Var8 == 'box';
%idx_epa = samplepri.Var8 == 'epanechnikov'
%idx_norm = samplepri.Var8 == 'normal'
% create table based on index
%tri = filt_kernel(idx_tri,:);%%
%box = filt_kernel(idx_box,:);
%epa = filt_kernel(idx_epa,:);
%norm = filt_kernel(idx_norm,:);

%tri = sortrows(tri,'Var7','ascend');
%box = sortrows(box,'Var7','ascend');
%epa = sortrows(epa,'Var7','ascend');
%norm = sortrows(norm,'Var7','ascend');

%figure
%plot((tri.Var7) , tri.Var2)
%hold on 
%plot((box.Var7) , box.Var2)
%hold on
%plot((epa.Var7) , epa.Var2)
%hold on
%plot((norm.Var7) , norm.Var2)
%legend({'tri', 'box', 'epa', 'norm'})
%ylabel('K-Fold Cross-Validation Loss')
%xlabel('Kernel Width')
%%
%%taken from https://www.mathworks.com/matlabcentral/answers/412639-creating-surface-plot-from-a-matrix-with-3-columns
%Comparing Resub, Cross-Validation and OOB Errors where Min Leaf Size = 1 
xi = unique(min_leaf_1.Var1_11) ; yi = unique(min_leaf_1.Var1_13) ;
[X,Y] = meshgrid(xi,yi) ;
Z = reshape(min_leaf_1.Var1_1,size(X)) ;
err_rs = reshape(min_leaf_1.Var1_2,size(X)) ;

xi2 = unique(min_leaf_1.Var1_11) ; yi2 = unique(min_leaf_1.Var1_13) ;
[X2,Y2] = meshgrid(xi2,yi2) ;
Z2 = reshape(min_leaf_1.Var1_3,size(X2)) ;
err_cv = reshape(min_leaf_1.Var1_4,size(X)) ;

xi3 = unique(min_leaf_10.Var1_11) ; yi3 = unique(min_leaf_10.Var1_13) ;
[X3,Y3] = meshgrid(xi3,yi3) ;
Z3 = reshape(min_leaf_10.Var1_1,size(X3)) ;

xi4 = unique(min_leaf_10.Var1_11) ; yi4 = unique(min_leaf_10.Var1_13) ;
[X4,Y4] = meshgrid(xi4,yi4) ;
Z4 = reshape(min_leaf_10.Var1_3,size(X4)) ;

err_rs2 = reshape(min_leaf_10.Var1_2,size(X)) ;
err_cv2 = reshape(min_leaf_10.Var1_4,size(X)) ;

xi5 = unique(min_leaf_50.Var1_11) ; yi5 = unique(min_leaf_50.Var1_13) ;
[X5,Y5] = meshgrid(xi5,yi5) ;
Z5 = reshape(min_leaf_50.Var1_1,size(X5)) ;

xi6 = unique(min_leaf_50.Var1_11) ; yi6 = unique(min_leaf_50.Var1_13) ;
[X6,Y6] = meshgrid(xi6,yi6) ;
Z6 = reshape(min_leaf_50.Var1_3,size(X6)) ;

err_rs3 = reshape(min_leaf_50.Var1_2,size(X)) ;
err_cv3 = reshape(min_leaf_50.Var1_4,size(X)) ;

xi7 = unique(min_leaf_100.Var1_11) ; yi7 = unique(min_leaf_100.Var1_13) ;
[X7,Y7] = meshgrid(xi7,yi7) ;
Z7 = reshape(min_leaf_100.Var1_1,size(X7)) ;

xi8 = unique(min_leaf_100.Var1_11) ; yi8 = unique(min_leaf_100.Var1_13) ;
[X8,Y8] = meshgrid(xi8,yi8) ;
Z8 = reshape(min_leaf_100.Var1_3,size(X8)) ;

err_rs4 = reshape(min_leaf_100.Var1_2,size(X)) ;
err_cv4 = reshape(min_leaf_100.Var1_4,size(X)) ;

xi9 = unique(min_leaf_150.Var1_11) ; yi9 = unique(min_leaf_150.Var1_13) ;
[X9,Y9] = meshgrid(xi9,yi9) ;
Z9 = reshape(min_leaf_150.Var1_1,size(X9)) ;

xi10 = unique(min_leaf_150.Var1_11) ; yi10 = unique(min_leaf_150.Var1_13) ;
[X10,Y10] = meshgrid(xi10,yi10) ;
Z10 = reshape(min_leaf_150.Var1_3,size(X10)) ;

err_rs5 = reshape(min_leaf_150.Var1_2,size(X)) ;
err_cv5 = reshape(min_leaf_150.Var1_4,size(X)) ;

figure
tiledlayout(3,2);

nexttile
surf(log(X),log(Y),Z, 'FaceColor', 'g')
hold on
surf(log(X2),log(Y2),Z2, 'FaceColor','r')
hold on
%surf(log(X3),log(Y3),Z3,'FaceColor', 'b')
%hold on 
%h = scatter3(log(X2(i)),log(Y2(i)),Z2(i),'o', 'filled');
xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlim([0 inf])
zlabel('Error')
title('Where Minimum Leaf Size = 1')

nexttile 
surf(log(X3),log(Y3),Z3, 'FaceColor', 'g')
hold on
surf(log(X4),log(Y4),Z4, 'FaceColor','r')
hold on
%surf(log(X3),log(Y3),Z3,'FaceColor', 'b')
%hold on 
%h2 = scatter3(log(X4(i)),log(Y4(i)),Z4(i),'o', 'filled');
xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlim([0 inf])
zlabel('Error')
title('Where Minimum Leaf Size = 10')

nexttile 
surf(log(X5),log(Y5),Z5, 'FaceColor', 'g')
hold on
surf(log(X6),log(Y6),Z6, 'FaceColor','r')
hold on
%surf(log(X3),log(Y3),Z3,'FaceColor', 'b')
%hold on 
%h3 = scatter3(log(X6(i)),log(Y6(i)),Z6(i),'o', 'filled');
xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlim([0 inf])
zlabel('Error')
title('Where Minimum Leaf Size = 50')

nexttile 
surf(log(X7),log(Y7),Z7, 'FaceColor', 'g')
hold on
surf(log(X8),log(Y8),Z8, 'FaceColor','r')
hold on
%surf(log(X3),log(Y3),Z3,'FaceColor', 'b')
%hold on 
%h4 = scatter3(log(X8(i)),log(Y8(i)),Z8(i),'o', 'filled');
xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlim([0 inf])
zlabel('Error')
title('Where Minimum Leaf Size = 100')

nexttile 
surf(log(X9),log(Y9),Z9, 'FaceColor', 'g')
hold on
surf(log(X10),log(Y10),Z10, 'FaceColor','r')
hold on
%surf(log(X3),log(Y3),Z3,'FaceColor', 'b')
%hold on 
%h5 = scatter3(log(X10(i)),log(Y10(i)),Z10(i),'o', 'filled');
xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlim([0 inf])
zlabel('Error')
title('Where Minimum Leaf Size = 150')

legend({'Average Resubstitution error', 'Cross-validation error'})

%%
figure
tiledlayout(3,2);

nexttile
surf(log(X),log(Y),err_rs, 'FaceColor', 'g')
hold on
surf(log(X2),log(Y2),err_cv, 'FaceColor','r')
hold on

xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlabel('Error Standard Deviation')
zlim([0 inf])
title('Where Minimum Leaf Size = 1')

nexttile
surf(log(X3),log(Y3),err_rs2, 'FaceColor', 'g')
hold on
surf(log(X4),log(Y4),err_cv2, 'FaceColor','r')
hold on

xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlabel('Error Standard Deviation')
zlim([0 inf])
title('Where Minimum Leaf Size = 10')

nexttile
surf(log(X5),log(Y5),err_rs3, 'FaceColor', 'g')
hold on
surf(log(X6),log(Y6),err_cv3, 'FaceColor','r')
hold on

xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlabel('Error Standard Deviation')
zlim([0 inf])
title('Where Minimum Leaf Size = 50')

nexttile
surf(log(X7),log(Y7),err_rs4, 'FaceColor', 'g')
hold on
surf(log(X8),log(Y8),err_cv4, 'FaceColor','r')
hold on

xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlabel('Error Standard Deviation')
zlim([0 inf])
title('Where Minimum Leaf Size = 100')

nexttile
surf(log(X9),log(Y9),err_rs5, 'FaceColor', 'g')
hold on
surf(log(X10),log(Y10),err_cv5, 'FaceColor','r')
hold on

xlabel('log(Number of Trees)')
ylabel('log(Max Number of Splits)')
zlabel('Error Standard Deviation')
zlim([0 inf])
title('Where Minimum Leaf Size = 150')

legend({'Resubstitution error standard deviation', 'Cross-validation error standard deviation'})

%%
clear all
close all
clc
%%
xi = unique(min_leaf_1.Var1_8) ; yi = unique(min_leaf_1.Var1_10) ;
[X,Y] = meshgrid(xi,yi) ;
Z = reshape(min_leaf_1.Var1_12,size(X)) ;

xi2 = unique(min_leaf_10.Var1_8) ; yi2 = unique(min_leaf_10.Var1_10) ;
[X2,Y2] = meshgrid(xi2,yi2) ;
Z2 = reshape(min_leaf_10.Var1_12,size(X2)) ;

xi3 = unique(min_leaf_50.Var1_8) ; yi3 = unique(min_leaf_50.Var1_10) ;
[X3,Y3] = meshgrid(xi3,yi3) ;
Z3 = reshape(min_leaf_50.Var1_12,size(X3)) ;

xi4 = unique(min_leaf_100.Var1_8) ; yi4 = unique(min_leaf_100.Var1_10) ;
[X4,Y4] = meshgrid(xi4,yi4) ;
Z4 = reshape(min_leaf_100.Var1_12,size(X4)) ;

xi5 = unique(min_leaf_150.Var1_8) ; yi5 = unique(min_leaf_150.Var1_10) ;
[X5,Y5] = meshgrid(xi5,yi5) ;
Z5 = reshape(min_leaf_150.Var1_12,size(X5)) ;

%xi6 = unique(min_leaf_10000.Var1_8) ; yi6 = unique(min_leaf_10000.Var1_10) ;
%[X6,Y6] = meshgrid(xi6,yi6) ;
%Z6 = reshape(min_leaf_10000.Var1_12,size(X6)) ;
%xi2 = unique(min_leaf_1.Var1_8) ; yi2 = unique(min_leaf_1.Var1_9) ;
%[X2,Y2] = meshgrid(xi2,yi2) ;
%Z2 = reshape(min_leaf_1.Var1_12,size(X)) ;
%xi3 = unique(min_leaf_1.Var1_8) ; yi3 = unique(min_leaf_1.Var1_10) ;
%[X3,Y3] = meshgrid(xi3,yi3) ;
%Z3 = reshape(min_leaf_1.Var1_12,size(X)) ;
figure

surf(log(X),log(Y),Z, 'FaceColor', 'g')
hold on
surf(log(X2),log(Y2),Z2, 'FaceColor','r')
hold on
surf(log(X3),log(Y3),Z3,'FaceColor', 'b')
surf(log(X4),log(Y4),Z4,'FaceColor', 'magenta')
surf(log(X5),log(Y5),Z5,'FaceColor', 'black')
%surf(log(X6),log(Y6),Z6,'FaceColor', 'yellow')
xlabel('log(number of trees)')
ylabel('log(max number of splits)')
zlabel('Average time for training and Cross-Validation')
title('Recorded Variation in average Training and Cross-Validation Time with  Number of Trees and Maximum Number of Splits')
legend({'Min Leaf Size = 1', 'Min Leaf Size = 10', 'Min Leaf Size = 50','Min Leaf Size = 100','Min Leaf Size = 150'})

