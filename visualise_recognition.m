function visualise_recognition(va_img_sample,prob,Yva,data_idx,nSample  )
%VISUALISE_VERIFICATION 
%==========================================================================
% Input:
%   va_img_sample: the raw validation cell array
%   prob:        the probability of negative and positive samples which has
%                dimension Nx2.
%   Yva          : The ground truth of validation set
%   data_idx     : The index of data
%   nSample      : The number of visualise pair
%==========================================================================


figure;
hold on

count = 0;
uniq_Y = unique(Yva);
label_str = {'Diffferent', 'Same'};
true_false_str = {'False','True'};

for i = 1:length(data_idx)
    count=count+1;
    subplot(nSample,3,count);
    imshow(va_img_sample{data_idx(i),1});
    title('Img')

    count=count+1;
    ax = subplot(nSample,3,count);
    pred_prob = prob(data_idx(i),:);
    
    [~,pred_l] = max(pred_prob);
    pred_label_idx = find(Yva==pred_l);
    pred_label_str = va_img_sample{pred_label_idx(1),2}(1:9);
    pred_label_str = strrep(pred_label_str,'_', ' ');
    label_str = va_img_sample{data_idx(i),2}(1:9);
    label_str = strrep(label_str,'_', ' ');
    text(0,0.6, 'f(x)' , 'FontSize',14);
    text(0.3,0.6,pred_label_str , 'FontSize',14);
    text(0,0.3, 'y:' , 'FontSize',14);
    text(0.3,0.3,label_str , 'FontSize',14);
    set( ax, 'visible', 'off')
    

    count=count+1;
    ax = subplot(nSample,3,count);
    
    idx = sum(pred_l == Yva(data_idx(i)))+1;
    text(0.3,0.9, 'Is f(x)=y ?', 'FontSize',16);
    text(0.5,0.5,true_false_str{idx} , 'FontSize',14);
    set ( ax, 'visible', 'off')
end

saveas(gcf, 'visualise_recognition.png')

end

