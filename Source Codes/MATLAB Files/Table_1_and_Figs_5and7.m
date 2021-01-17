%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
clc

%% %%%%%%%%% Table 1 %%%%%%%%%%%%% %%

%% %%% Shuffled Test Set (25 and 17 Joints) %%% %%
idx = [1,2,3,4,5,6,7,8,9,10];
for i=1:10
    Overall_MAE = [];
    Overall_MAE_17 = [];
    for j=1:5
        path = sprintf('../../Dataset and Predictions/Data_%s_Frames/MAE/MAE_shuffled_test_%s.npy',num2str(idx(i)),num2str(j));
        MAE = readNPY(path);
        idx_17 = [1,2,3,4,5,6,9,10,13,14,15,16,17,18,19,20,21];
        MAE_17 = MAE(idx_17,:);
        Overall_MAE = cat(1,Overall_MAE,MAE);
        Overall_MAE_17 = cat(1,Overall_MAE_17,MAE_17);
    end
   
    MAE_tot(i,:) = mean(Overall_MAE);
    MAE_tot_17(i,:) = mean(Overall_MAE_17);
    
end
fprintf('\n')
fprintf('################ MAE(cm) for 25 joints for Shuffled Test Set ######################\n')
fprintf('-------------------------------------------------------------------------------\n')
fprintf('         Input Frames  |      Depth    |    Horizontal  |        Vertical \n')
fprintf('-------------------------------------------------------------------------------\n')
for i=1:10
    fprintf('            %s-Frame    |    %s     |      %s    |        %s              \n', num2str(i), num2str(MAE_tot(i,1)*100), num2str(MAE_tot(i,2)*100), num2str(MAE_tot(i,3)*100))
end
fprintf('-------------------------------------------------------------------------------\n')
fprintf('\n')
fprintf('################ MAE(cm) for 17 joints for Shuffled Test Set ######################\n')
fprintf('-------------------------------------------------------------------------------\n')
fprintf('         Input Frames  |      Depth    |    Horizontal  |        Vertical \n')
fprintf('-------------------------------------------------------------------------------\n')
for i=1:10
    fprintf('            %s-Frame    |    %s     |      %s    |        %s              \n', num2str(i), num2str(MAE_tot_17(i,1)*100), num2str(MAE_tot_17(i,2)*100), num2str(MAE_tot_17(i,3)*100))
end
fprintf('-------------------------------------------------------------------------------\n')

%% %%% Continuous Test Set (25 and 17 Joints) %%% %%
idx = [1,2,3,4,5,6,7,8,9,10];
for i=1:10
    Overall_MAE = [];
    Overall_MAE_17 = [];
    for j=1:5
        path = sprintf('../../Dataset and Predictions/Data_%s_Frames/MAE/MAE_cont_test_%s.npy',num2str(idx(i)),num2str(j));
        MAE = readNPY(path);
        idx_17 = [1,2,3,4,5,6,9,10,13,14,15,16,17,18,19,20,21];
        MAE_17 = MAE(idx_17,:);
        Overall_MAE = cat(1,Overall_MAE,MAE);
        Overall_MAE_17 = cat(1,Overall_MAE_17,MAE_17);
    end
    
    MAE_tot_cont(i,:) = mean(Overall_MAE);
    MAE_tot_17_cont(i,:) = mean(Overall_MAE_17);

end
fprintf('\n')
fprintf('################ MAE(cm) for 25 joints for Continuous Test Set ######################\n')
fprintf('-------------------------------------------------------------------------------\n')
fprintf('         Input Frames  |      Depth    |    Horizontal  |        Vertical \n')
fprintf('-------------------------------------------------------------------------------\n')
for i=1:10
    fprintf('            %s-Frame    |    %s     |      %s    |        %s              \n', num2str(i), num2str(MAE_tot_cont(i,1)*100), num2str(MAE_tot_cont(i,2)*100), num2str(MAE_tot_cont(i,3)*100))
end
fprintf('-------------------------------------------------------------------------------\n')
fprintf('\n')
fprintf('################ MAE(cm) for 17 joints for Continuous Test Set ######################\n')
fprintf('-------------------------------------------------------------------------------\n')
fprintf('         Input Frames  |      Depth    |    Horizontal  |        Vertical \n')
fprintf('-------------------------------------------------------------------------------\n')
for i=1:10
    fprintf('            %s-Frame    |    %s     |      %s    |        %s              \n', num2str(i), num2str(MAE_tot_17_cont(i,1)*100), num2str(MAE_tot_17_cont(i,2)*100), num2str(MAE_tot_17_cont(i,3)*100))
end
fprintf('-------------------------------------------------------------------------------\n')

%% %%%%%%%%% Fig 5 %%%%%%%%%%%%% %%
close all
figure(5)
subplot(2,1,1)
plot(idx,MAE_tot(:,1)*100,'Marker','*','LineWidth',2)
hold on
plot(idx,MAE_tot(:,2)*100,':+','LineWidth',2)
hold on
plot(idx,MAE_tot(:,3)*100,'--o','LineWidth',2)
hold off
title('Mean Absolute Error for 25 joints (Before OD)')
legend('Depth Error', 'Azimuth Error', 'Elevation Error')
xlabel('Number of Input Frames')
grid on; ylabel('Error in centimeters')

subplot(2,1,2)
plot(idx,MAE_tot_17(:,1)*100,'Marker','*','LineWidth',2)
hold on
plot(idx,MAE_tot_17(:,2)*100,':+','LineWidth',2)
hold on
plot(idx,MAE_tot_17(:,3)*100,'--o','LineWidth',2)
hold off
title('Mean Absolute Error for 17 joints (After OD)')
legend('Depth Error', 'Azimuth Error', 'Elevation Error')
xlabel('Number of Input Frames')
grid on; ylabel('Error in centimeters')

%% %%%%%%%%% Fig 7 %%%%%%%%%%%%% %%

for i=1:10
    Overall_Timing = [];
    
    for j=1:5
        path = sprintf('../../Dataset and Predictions/Inference_Time/Time_Per_%s_%s.npy',num2str(idx(i)),num2str(j));
        MAE = readNPY(path);
        Overall_Timing = cat(1,Overall_Timing,MAE);
    end
    mean_Timing_tot(i,:) = mean(Overall_Timing);
        
    std_Timing_tot(i,:) = std(Overall_Timing);
    
end

figure(7)

errorbar(idx,mean_Timing_tot*1000,std_Timing_tot*1000,'--','Color',[0 0.45 0.74],'Marker','*','MarkerFaceColor','r','MarkerEdgeColor','r','LineWidth',2)

xlabel('Number of Input Frames')
grid on; ylabel('Inference Time in milliseconds')
title('Inference Time vs Input Frames')