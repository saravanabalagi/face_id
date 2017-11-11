function visualise_verification(va_img_pair,prob,Yva,data_idx,nPairs  )
%VISUALISE_VERIFICATION 
%==========================================================================
% Input:
%   va_img_pair: the raw validation cell array
%   prob:        the probability of negative and positive samples which has
%                dimension Nx2.
%   Yva        : The ground truth of validation set
%   data_idx   : The index of data
%   nPairs     : The number of visualise pair
%==========================================================================


figure;
hold on

count = 0;
uniq_Y = unique(Yva);
label_str = {'Diffferent', 'Same'};
true_false_str = {'False','True'};

for i = 1:length(data_idx)
    count=count+1;
    subplot(nPairs,4,count);
    imshow(va_img_pair{data_idx(i),1});
    title('Img 1')

    count=count+1;
    subplot(nPairs,4,count);
    imshow(va_img_pair{data_idx(i),3});
    title('Img 2')

    count=count+1;
    ax = subplot(nPairs,4,count);
    pred_prob = prob(data_idx(i),:);
    gt_prob = zeros(1,2); % Ground Truth
    if Yva(data_idx(i)) == -1;
        gt_prob(1)=1;
        gt_idx = 1;
    else
        gt_prob(2)=1;
        gt_idx = 2;
    end
    
    [~,pred_l] = max(pred_prob);
    idx = (uniq_Y(pred_l) == Yva(data_idx(i)))+1;
     
    text(0,0.6, 'f(x)' , 'FontSize',14);
    text(0.5,0.6,label_str{pred_l} , 'FontSize',14);
    text(0,0.3, 'y:' , 'FontSize',14);
    text(0.5,0.3,label_str{gt_idx} , 'FontSize',14);
    set ( ax, 'visible', 'off')
    

    count=count+1;
    ax = subplot(nPairs,4,count);
    
    
    text(0.3,0.9, 'Is f(x)=y ?', 'FontSize',16);
    text(0.5,0.5,true_false_str{idx} , 'FontSize',14);
    set ( ax, 'visible', 'off')
end

saveas(gcf, 'visualise_verification.png')

end

