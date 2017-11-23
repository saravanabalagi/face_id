function [ TP_num, condi_P, Pred_P] = evaluate_detector( bbox, pos_prob )
%EVALUATE_DETECTOR 
%==========================================================================
% - This function is to return the true positive number, condition
% positive, predicted positive number for each image with various threshold
% of positive class probability. This function can only evaluate on LFW
% datasets. (You should not modify this function.)
%--------------------------------------------------------------------------
% Output:
%  -TP_num  : True_positive number for this image with different threshold
%  -condi_P : Condition(Actual) Positive
%  -Pred_P  : Predicted Positive number.
% Input:
%  -bbox    : bounding box
%  -bbox    : gt_bbox. It should be a matrix. num_bbox-by-4
%  -pos_prob: positive probability.
%==========================================================================
    
    TP_num = zeros(1,100);
    condi_P = zeros(1,100);
    Pred_P = zeros(1,100);
    list_idx = 100:-1:1;
    
    for i = 100:-1:1
        threshold = 0.01*i;
        threshold_bbox=bbox(pos_prob>=threshold,:);
        prob3=pos_prob(pos_prob>=threshold,:);

        % Remove the redundant boxes via non-maximum supression.
        % - The bbox is the top-left x,y, height,width of the patches.
        if ~isempty(threshold_bbox)
            % Non-maximum suppression
            [selectedBbox,selectedScore] = selectStrongestBbox(threshold_bbox,prob3,'OverlapThreshold',0.5, 'RatioType','Union'); 

            % Overlapping ratio between ground truth and bounding box
            ratio = bboxOverlapRatio(selectedBbox, [83,92,166-83,175-92], 'ratioType','Union');                                   
            pos_num = sum(ratio>=0.5); 

            TP_num(1, list_idx(i)) = pos_num>=1;
            condi_P(1, list_idx(i)) = 1;
            Pred_P(1, list_idx(i)) = length(ratio);
        else
            TP_num(1, list_idx(i)) = 0;
            condi_P(1, list_idx(i)) = 1;
            Pred_P(1, list_idx(i)) = 0;
        end

    end
    
end

