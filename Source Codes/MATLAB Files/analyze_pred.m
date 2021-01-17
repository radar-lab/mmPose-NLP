function [Mean_X_ovr, Mean_Y_ovr, Mean_Z_ovr ,Mean_X_jnts, Mean_Y_jnts, Mean_Z_jnts, Mean_X_ovr_17, Mean_Y_ovr_17, Mean_Z_ovr_17, Mean_X_jnts_17, Mean_Y_jnts_17, Mean_Z_jnts_17] = analyze_pred(path,iter)
Mean_X_ovr = 0;
Mean_Y_ovr = 0;
Mean_Z_ovr = 0; 
Mean_X_jnts = zeros(1,25);
Mean_Y_jnts = zeros(1,25);
Mean_Z_jnts = zeros(1,25);
Mean_X_ovr_17 = 0;
Mean_Y_ovr_17 = 0;
Mean_Z_ovr_17 = 0;
Mean_X_jnts_17 = zeros(1,17); 
Mean_Y_jnts_17 = zeros(1,17); 
Mean_Z_jnts_17 = zeros(1,17);
for iii=1:iter
    path_GT = sprintf('%s/Ground Truth/GT_cont_test_%s.npy',path,num2str(iii));
    path_Pred = sprintf('%s/Predictions/Pred_cont_test_%s.npy',path,num2str(iii));
    
    GT = readNPY(path_GT);
    Pred = readNPY(path_Pred);
    
    for i = 1:length(GT(:,1,1))
        centroid_GT = mean(GT(i,:,:));
        New_GT(i,:,:) = GT(i,:,:);% - centroid_GT;

        centroid_Pred = mean(Pred(i,:,:));
        New_Pred(i,:,:) = Pred(i,:,:);% - centroid_Pred;
    end

    for i=1:length(New_GT(:,1))
        for j=1:25
            Diff_X(i,j) = abs(New_GT(i,j,1)-New_Pred(i,j,1));
            Diff_Y(i,j) = abs(New_GT(i,j,2)-New_Pred(i,j,2));
            Diff_Z(i,j) = abs(New_GT(i,j,3)-New_Pred(i,j,3));
        end
    end

    Mean_X_ovr = Mean_X_ovr+mean(mean(Diff_X));
    Mean_Y_ovr = Mean_Y_ovr+mean(mean(Diff_Y));
    Mean_Z_ovr = Mean_Z_ovr+mean(mean(Diff_Z));

    Mean_X_jnts = Mean_X_jnts+mean((Diff_X));
    Mean_Y_jnts = Mean_Y_jnts+mean((Diff_Y));
    Mean_Z_jnts = Mean_Z_jnts+mean((Diff_Z));

    idx = [1,2,3,4,5,6,9,10,13,14,15,16,17,18,19,20,21];
    for i=1:length(New_GT(:,1))
        for j=1:17
            Diff_X_17(i,j) = abs(New_GT(i,idx(j),1)-New_Pred(i,idx(j),1));
            Diff_Y_17(i,j) = abs(New_GT(i,idx(j),2)-New_Pred(i,idx(j),2));
            Diff_Z_17(i,j) = abs(New_GT(i,idx(j),3)-New_Pred(i,idx(j),3));
        end
    end

    Mean_X_ovr_17 = Mean_X_ovr_17+mean(mean(Diff_X_17));
    Mean_Y_ovr_17 = Mean_Y_ovr_17+mean(mean(Diff_Y_17));
    Mean_Z_ovr_17 = Mean_Z_ovr_17+mean(mean(Diff_Z_17));

    Mean_X_jnts_17 = Mean_X_jnts_17+mean((Diff_X_17));
    Mean_Y_jnts_17 = Mean_Y_jnts_17+mean((Diff_Y_17));
    Mean_Z_jnts_17 = Mean_Z_jnts_17+mean((Diff_Z_17));
end
Mean_X_ovr = Mean_X_ovr/iter;
Mean_Y_ovr = Mean_Y_ovr/iter;
Mean_Z_ovr = Mean_Z_ovr/iter;

Mean_X_jnts = Mean_X_jnts/iter;
Mean_Y_jnts = Mean_Y_jnts/iter;
Mean_Z_jnts = Mean_Z_jnts/iter;

Mean_X_ovr_17 = Mean_X_ovr_17/iter;
Mean_Y_ovr_17 = Mean_Y_ovr_17/iter;
Mean_Z_ovr_17 = Mean_Z_ovr_17/iter;

Mean_X_jnts_17 = Mean_X_jnts_17/iter;
Mean_Y_jnts_17 = Mean_Y_jnts_17/iter;
Mean_Z_jnts_17 = Mean_Z_jnts_17/iter;



end

