close all
clear all
clc

Frames_Case = 8;
Iteration = 1;
kk=1;
GT_path =sprintf('../../Dataset and Predictions/Data_%s_Frames/Ground Truth/GT_cont_test_%s.npy',num2str(Frames_Case),num2str(Iteration));
Pred_path =sprintf('../../Dataset and Predictions/Data_%s_Frames/Predictions/Pred_cont_test_%s.npy',num2str(Frames_Case),num2str(Iteration));

GT = readNPY(GT_path);
Pred = readNPY(Pred_path);

frames_plot = [4,8,5,90,103,106,157,185,195,490,519,524];

%%
SkeletonConnectionMap = [ [4 3];  % Neck
                          [3 21]; % Head
                          [21 2]; % Right Leg
                          [2 1];
                          [21 9];
                          [9 10];  % Hip
                          [10 11];
                          [11 12]; % Left Leg
                          [12 24];
                          [12 25];
                          [21 5];  % Spine
                          [5 6];
                          [6 7];   % Left Hand
                          [7 8];
                          [8 22];
                          [8 23];
                          [1 17];
                          [17 18];
                          [18 19];  % Right Hand
                          [19 20];
                          [1 13];
                          [13 14];
                          [14 15];
                          [15 16];
                        ];
%%
h=figure('Name','Figure 6','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
for k=1:length(frames_plot)
    
    j = uint16(frames_plot(k));
    for i=1:24
    joint1 = SkeletonConnectionMap(i,1);
    joint2 = SkeletonConnectionMap(i,2);
    
    X1 = [GT(j,joint1,1),GT(j,joint1,2),GT(j,joint1,3)] ;
    X2 = [GT(j,joint2,1),GT(j,joint2,2),GT(j,joint2,3)]; 
    
    X_comb = [X1;X2];
    subplot(4,6,2*k-1)
    line(X_comb(:,1), X_comb(:,2), X_comb(:,3),'LineWidth',4,'Color','r','Marker','o','MarkerEdgeColor',[0 1 0],'MarkerSize',1,'MarkerFaceColor',[0 1 0])
    ylim([-1,1])
    zlim([-1.5,1.5])
    xlim([-3,3])
    view(-70,11)
    grid on
    xlabel('Depth')
    ylabel('Azimuth')
    zlabel('Elevation')
    title('Ground Truth')
    hold on     
    end
    hold off
    
    for i=1:24
    joint1 = SkeletonConnectionMap(i,1);
    joint2 = SkeletonConnectionMap(i,2);
    
    X1 = [Pred(j,joint1,1),Pred(j,joint1,2),Pred(j,joint1,3)] ;
    X2 = [Pred(j,joint2,1),Pred(j,joint2,2),Pred(j,joint2,3)]; 
    
    X_comb = [X1;X2];
    subplot(4,6,2*k)
    line(X_comb(:,1), X_comb(:,2), X_comb(:,3),'LineWidth',4,'Color','b','Marker','o','MarkerEdgeColor',[1 0 0],'MarkerSize',1,'MarkerFaceColor',[1 0 0])
    ylim([-1,1])
    zlim([-1.5,1.5])
    xlim([-3,3])
    view(-70,11)
    xlabel('Depth')
    ylabel('Azimuth')
    zlabel('Elevation')
    title('mmPose-NLP Prediction')
    grid on
    hold on     
    end
    hold off
end