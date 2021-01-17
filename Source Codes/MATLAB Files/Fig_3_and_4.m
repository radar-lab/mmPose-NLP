%close all
clear all
clc

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
path1 = '../../Dataset and Predictions/Data_1_Frames'

path2 = '../../Dataset and Predictions/Data_2_Frames'

path3 = '../../Dataset and Predictions/Data_3_Frames'

path4 = '../../Dataset and Predictions/Data_4_Frames'

path5 = '../../Dataset and Predictions/Data_5_Frames'

path6 = '../../Dataset and Predictions/Data_6_Frames'

path7 = '../../Dataset and Predictions/Data_7_Frames'

path8 = '../../Dataset and Predictions/Data_8_Frames'

path9 = '../../Dataset and Predictions/Data_9_Frames'

path10 = '../../Dataset and Predictions/Data_10_Frames'

iter=5;
%% %%%%%%%%%%%%%%%% Single Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_1, Mean_Y_ovr_1, Mean_Z_ovr_1 ,Mean_X_jnts_1, Mean_Y_jnts_1, Mean_Z_jnts_1, Mean_X_ovr_17_1, Mean_Y_ovr_17_1, Mean_Z_ovr_17_1, Mean_X_jnts_17_1, Mean_Y_jnts_17_1, Mean_Z_jnts_17_1] = analyze_pred(path1,iter);

%% %%%%%%%%%%%%%%%% Two Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_2, Mean_Y_ovr_2, Mean_Z_ovr_2 ,Mean_X_jnts_2, Mean_Y_jnts_2, Mean_Z_jnts_2, Mean_X_ovr_17_2, Mean_Y_ovr_17_2, Mean_Z_ovr_17_2, Mean_X_jnts_17_2, Mean_Y_jnts_17_2, Mean_Z_jnts_17_2] = analyze_pred(path2,iter);

%% %%%%%%%%%%%%%%%% Three Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_3, Mean_Y_ovr_3, Mean_Z_ovr_3 ,Mean_X_jnts_3, Mean_Y_jnts_3, Mean_Z_jnts_3, Mean_X_ovr_17_3, Mean_Y_ovr_17_3, Mean_Z_ovr_17_3, Mean_X_jnts_17_3, Mean_Y_jnts_17_3, Mean_Z_jnts_17_3] = analyze_pred(path3,iter);

%% %%%%%%%%%%%%%%%% Four Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_4, Mean_Y_ovr_4, Mean_Z_ovr_4 ,Mean_X_jnts_4, Mean_Y_jnts_4, Mean_Z_jnts_4, Mean_X_ovr_17_4, Mean_Y_ovr_17_4, Mean_Z_ovr_17_4, Mean_X_jnts_17_4, Mean_Y_jnts_17_4, Mean_Z_jnts_17_4] = analyze_pred(path4,iter);

%% %%%%%%%%%%%%%%%% Five Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_5, Mean_Y_ovr_5, Mean_Z_ovr_5 ,Mean_X_jnts_5, Mean_Y_jnts_5, Mean_Z_jnts_5, Mean_X_ovr_17_5, Mean_Y_ovr_17_5, Mean_Z_ovr_17_5, Mean_X_jnts_17_5, Mean_Y_jnts_17_5, Mean_Z_jnts_17_5] = analyze_pred(path5,iter);

%% %%%%%%%%%%%%%%%% Six Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_6, Mean_Y_ovr_6, Mean_Z_ovr_6 ,Mean_X_jnts_6, Mean_Y_jnts_6, Mean_Z_jnts_6, Mean_X_ovr_17_6, Mean_Y_ovr_17_6, Mean_Z_ovr_17_6, Mean_X_jnts_17_6, Mean_Y_jnts_17_6, Mean_Z_jnts_17_6] = analyze_pred(path6,iter);

%% %%%%%%%%%%%%%%%% Seven Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_7, Mean_Y_ovr_7, Mean_Z_ovr_7 ,Mean_X_jnts_7, Mean_Y_jnts_7, Mean_Z_jnts_7, Mean_X_ovr_17_7, Mean_Y_ovr_17_7, Mean_Z_ovr_17_7, Mean_X_jnts_17_7, Mean_Y_jnts_17_7, Mean_Z_jnts_17_7] = analyze_pred(path7,iter);

%% %%%%%%%%%%%%%%%% Eight Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_8, Mean_Y_ovr_8, Mean_Z_ovr_8 ,Mean_X_jnts_8, Mean_Y_jnts_8, Mean_Z_jnts_8, Mean_X_ovr_17_8, Mean_Y_ovr_17_8, Mean_Z_ovr_17_8, Mean_X_jnts_17_8, Mean_Y_jnts_17_8, Mean_Z_jnts_17_8] = analyze_pred(path8,iter);

%% %%%%%%%%%%%%%%%% Nine Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_9, Mean_Y_ovr_9, Mean_Z_ovr_9 ,Mean_X_jnts_9, Mean_Y_jnts_9, Mean_Z_jnts_9, Mean_X_ovr_17_9, Mean_Y_ovr_17_9, Mean_Z_ovr_17_9, Mean_X_jnts_17_9, Mean_Y_jnts_17_9, Mean_Z_jnts_17_9] = analyze_pred(path9,iter);

%% %%%%%%%%%%%%%%%% Ten Frame Case %%%%%%%%%%%%%%%%%%%%%%%%% %%

[Mean_X_ovr_10, Mean_Y_ovr_10, Mean_Z_ovr_10 ,Mean_X_jnts_10, Mean_Y_jnts_10, Mean_Z_jnts_10, Mean_X_ovr_17_10, Mean_Y_ovr_17_10, Mean_Z_ovr_17_10, Mean_X_jnts_17_10, Mean_Y_jnts_17_10, Mean_Z_jnts_17_10] = analyze_pred(path10,iter);


%% %%%%%%%%%%%%% Stacked %%%%%%%%%%%%%% %%
Mean_ovr = [Mean_X_ovr_1 Mean_Y_ovr_1 Mean_Z_ovr_1; Mean_X_ovr_2 Mean_Y_ovr_2 Mean_Z_ovr_2; Mean_X_ovr_3 Mean_Y_ovr_3 Mean_Z_ovr_3; Mean_X_ovr_4 Mean_Y_ovr_4 Mean_Z_ovr_4; Mean_X_ovr_5 Mean_Y_ovr_5 Mean_Z_ovr_5;Mean_X_ovr_6 Mean_Y_ovr_6 Mean_Z_ovr_6; Mean_X_ovr_7 Mean_Y_ovr_7 Mean_Z_ovr_7; Mean_X_ovr_8 Mean_Y_ovr_8 Mean_Z_ovr_8; Mean_X_ovr_9 Mean_Y_ovr_9 Mean_Z_ovr_9; Mean_X_ovr_10 Mean_Y_ovr_10 Mean_Z_ovr_10];
Mean_ovr_17 = [Mean_X_ovr_17_1 Mean_Y_ovr_17_1 Mean_Z_ovr_17_1; Mean_X_ovr_17_2 Mean_Y_ovr_17_2 Mean_Z_ovr_17_2; Mean_X_ovr_17_3 Mean_Y_ovr_17_3 Mean_Z_ovr_17_3; Mean_X_ovr_17_4 Mean_Y_ovr_17_4 Mean_Z_ovr_17_4; Mean_X_ovr_17_5 Mean_Y_ovr_17_5 Mean_Z_ovr_17_5; Mean_X_ovr_17_6 Mean_Y_ovr_17_6 Mean_Z_ovr_17_6; Mean_X_ovr_17_7 Mean_Y_ovr_17_7 Mean_Z_ovr_17_7; Mean_X_ovr_17_8 Mean_Y_ovr_17_8 Mean_Z_ovr_17_8; Mean_X_ovr_17_9 Mean_Y_ovr_17_9 Mean_Z_ovr_17_9; Mean_X_ovr_17_10 Mean_Y_ovr_17_10 Mean_Z_ovr_17_10];

%% %%%%%%%%%%%%% Plot 3 %%%%%%%%%%%%%%%%%%% %%
h=figure('Name','Figure 3','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
subplot(3,4,1)
plot(Mean_X_jnts_1,'LineWidth',2)
hold on
plot(Mean_Y_jnts_1,'LineWidth',2)
hold on
plot(Mean_Z_jnts_1,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 1 Frames')
hold off

subplot(3,4,2)
plot(Mean_X_jnts_2,'LineWidth',2)
hold on
plot(Mean_Y_jnts_2,'LineWidth',2)
hold on
plot(Mean_Z_jnts_2,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 2 Frames')
hold off

subplot(3,4,3)
plot(Mean_X_jnts_3,'LineWidth',2)
hold on
plot(Mean_Y_jnts_3,'LineWidth',2)
hold on
plot(Mean_Z_jnts_3,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 3 Frames')
hold off

subplot(3,4,4)
plot(Mean_X_jnts_4,'LineWidth',2)
hold on
plot(Mean_Y_jnts_4,'LineWidth',2)
hold on
plot(Mean_Z_jnts_4,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 4 Frames')
hold off

subplot(3,4,5)
plot(Mean_X_jnts_5,'LineWidth',2)
hold on
plot(Mean_Y_jnts_5,'LineWidth',2)
hold on
plot(Mean_Z_jnts_5,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 5 Frames')
hold off

subplot(3,4,6)
plot(Mean_X_jnts_6,'LineWidth',2)
hold on
plot(Mean_Y_jnts_6,'LineWidth',2)
hold on
plot(Mean_Z_jnts_6,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 6 Frames')
hold off

subplot(3,4,7)
plot(Mean_X_jnts_7,'LineWidth',2)
hold on
plot(Mean_Y_jnts_7,'LineWidth',2)
hold on
plot(Mean_Z_jnts_7,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 7 Frames')
hold off

subplot(3,4,8)
plot(Mean_X_jnts_8,'LineWidth',2)
hold on
plot(Mean_Y_jnts_8,'LineWidth',2)
hold on
plot(Mean_Z_jnts_8,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 8 Frames')
hold off

subplot(3,4,9)
plot(Mean_X_jnts_9,'LineWidth',2)
hold on
plot(Mean_Y_jnts_9,'LineWidth',2)
hold on
plot(Mean_Z_jnts_9,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 9 Frames')
hold off

subplot(3,4,10)
plot(Mean_X_jnts_10,'LineWidth',2)
hold on
plot(Mean_Y_jnts_10,'LineWidth',2)
hold on
plot(Mean_Z_jnts_10,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 10 Frames')
hold off

subplot(3,4,[11,12])
hB=bar(Mean_ovr*100)
ylim([0,20])
xlabel('Number of Frames per Prediction')
ylabel('Error in centimeters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Overall Localization Error over 25 joints')
% hT=[];              % placeholder for text object handles
% for i=1:length(hB)  % iterate over number of bar objects
%   hT=[hT text(hB(i).XData+hB(i).XOffset,hB(i).YData,num2str(hB(i).YData.','%.1f'), ...
%                           'VerticalAlignment','bottom','horizontalalign','center','FontSize',6)];
% end
hold off

%% %%%%%%%%%%% Plot 4 %%%%%%%%%%%%%%%%%%% %%
h=figure('Name','Figure 4','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
subplot(3,4,1)
plot(Mean_X_jnts_17_1,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_1,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_1,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 1 Frames')
hold off

subplot(3,4,2)
plot(Mean_X_jnts_17_2,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_2,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_2,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 2 Frames')
hold off

subplot(3,4,3)
plot(Mean_X_jnts_17_3,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_3,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_3,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 3 Frames')
hold off

subplot(3,4,4)
plot(Mean_X_jnts_17_4,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_4,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_4,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 4 Frames')
hold off

subplot(3,4,5)
plot(Mean_X_jnts_17_5,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_5,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_5,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 5 Frames')
hold off

subplot(3,4,6)
plot(Mean_X_jnts_17_6,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_6,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_6,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 6 Frames')
hold off

subplot(3,4,7)
plot(Mean_X_jnts_17_7,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_7,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_7,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 7 Frames')
hold off

subplot(3,4,8)
plot(Mean_X_jnts_17_8,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_8,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_8,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 8 Frames')
hold off

subplot(3,4,9)
plot(Mean_X_jnts_17_9,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_9,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_9,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 9 Frames')
hold off

subplot(3,4,10)
plot(Mean_X_jnts_17_10,'LineWidth',2)
hold on
plot(Mean_Y_jnts_17_10,'LineWidth',2)
hold on
plot(Mean_Z_jnts_17_10,'LineWidth',2)
ylim([0,0.2]); grid on;
xlabel('Joint Index')
ylabel('Error in meters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Prediction in 10 Frames')
hold off

subplot(3,4,[11,12])
hB=bar(Mean_ovr_17*100)
ylim([0,20])
xlabel('Number of Frames per Prediction')
ylabel('Error in centimeters')
legend('Depth Error', 'Horizontal Error', 'Vertical Error')
title('Overall Localization Error over 17 joints')
% hT=[];              % placeholder for text object handles
% for i=1:length(hB)  % iterate over number of bar objects
%   hT=[hT text(hB(i).XData+hB(i).XOffset,hB(i).YData,num2str(hB(i).YData.','%.1f'), ...
%                           'VerticalAlignment','bottom','horizontalalign','center','FontSize',6)];
% end
hold off

%  %% %%%%%%%%%%%%%%%%%%%% Output Gifs %%%%%%%%%%%%%%%%%%%%%%%%% %%
% % 
% 
% for iii=1:5
%     for jjj=1:10
%         
%         close all
%         clear frame
%         clc
%         % 
%         %h=figure('units','normalized','outerposition',[0 0 1 1]);
%         h=figure()
%         Frames_Case = jjj;
%         Iteration = iii;
%         kk=1;
%         GT_path =sprintf('../../DataSet/ML_Training_New_Nov23/Data_%s_Frames/Predictions/GT_test_%s.npy',num2str(Frames_Case),num2str(Iteration));
%         Pred_path =sprintf('../../DataSet/ML_Training_New_Nov23/Data_%s_Frames/Predictions/Pred_test_%s.npy',num2str(Frames_Case),num2str(Iteration));
% 
%         GT = readNPY(GT_path);
%         Pred = readNPY(Pred_path);
%         for kk=1:length(GT(:,1,1))/2
%             i=2*kk;
%             subplot(1,2,1)
%             scatter3(GT(i,:,1),GT(i,:,2),GT(i,:,3),'filled', 'MarkerEdgeColor','k','MarkerFaceColor',[1 0 0])
%             ylim([-2,2])
%             zlim([-2,2])
%             xlim([-3,3])
%             view(-70,11)
%             %view(0,0)
%             xlabel('Depth')
%             grid on; ylabel('Azimuth')
%             zlabel('Elevation')    
%             title('Ground Truth')
% 
%             subplot(1,2,2)
%             scatter3(Pred(i,:,1),Pred(i,:,2),Pred(i,:,3),'filled', 'MarkerEdgeColor','k','MarkerFaceColor',[0 1 0])
%             ylim([-2,2])
%             zlim([-2,2])
%             xlim([-3,3])
%             view(-70,11)
%             %view(0,0)
%             xlabel('Depth')
%             grid on; ylabel('Azimuth')
%             zlabel('Elevation')    
%             title('Prediction')
% 
%             pause(0.01)
%     
%     
% 
%             drawnow
%             frame(kk) = getframe(h);
%     
%             kk=kk+1;
%         end
%         
%         filevid = sprintf('../../DataSet/ML_Training_New_Nov23/Continuous Data Videos/GT_vs_Pred_%s_%s.avi',num2str(Frames_Case),num2str(Iteration));
%         filename = filevid
%         writerObj = VideoWriter(filename);
%         writerObj.FrameRate = 5;
%         
%         open(writerObj);
%         
%         for idx = 1:length(frame)
%             F = frame(idx) ;    
%             writeVideo(writerObj, F);
%         end
%         
%         close(writerObj);
%         
%     end
% end