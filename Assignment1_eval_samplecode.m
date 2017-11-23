% Image and Visual Computing Assignment 1: Face Detection
%==========================================================================
%   This is the sample code to show how to evaluate your detector. We are
%   using the interpolated average precision to evaluate the final
%   performance. Both average precision calculation and evaluate_detector.m
%   is updated. 
%==========================================================================
%% Evaluating your result on the val_datasets

run ICV_setup % add the library path
load('face_detector.mat') % load your pretrained model

% The test set
val_dir{2} = './data/face_detection/te_raw_images/';            % Test data (For performance evaluation)
val_file2 = dir(val_dir{2});
val_file2(1:2)=[];

% % The validation set
% val_dir{2} = './data/face_detection/val_raw_images/';            % Validation data (For performance evaluation)
% val_file2 = dir(val_dir{2});
% val_file2(1:2)=[];

% Initialization of the true positive, condition positive and prediction
% positive number collection.

total_TP = zeros(length(val_file2),100);
total_condi_P = zeros(length(val_file2),100);
total_Pred_P = zeros(length(val_file2),100);



imset = imageSet(val_dir{2}, 'recursive');
count = 0;

for k=1:length(val_file2)    
    
    for j = 1:length(imset(k).Count)
        count = count+1;
        img = read(imset(k),j);
        plt_img=img;
        if size(img,3)>1, img = rgb2gray(img); end
        window_size=[64 64];

        % Use sliding window to get multiple patches from the original image
        [patches,temp_bbox] = sw_detect_face(img,window_size,0.8,8);    
        Xte=[];
        bbox_ms = [];

        % Extract your features here
        for p=1:length(patches)
            for j = 1:size(patches{p},3)
                face_img = single(patches{p}(:,:,j))/255;
                Xte = [Xte face_img(:)];
                bbox_ms = [bbox_ms;temp_bbox{p}(j,:)];
            end
        end

        % Get the positive probability
        Xte = Xte';
        %[~,~,d] = liblinearpredict(ones(size(Xte,1),1),sparse(double(Xte)),detector);
        %prob2 = 1./(1+exp(-d));
        %prob = [1-prob2, prob2];

        [~,score] = predict(Mdl,Xte);
        prob2 = score(:,2);
    
        % Getting the True positive, condition positive, predicted positive
        % number for evaluating the algorithm performance via Average 
        [ TP_num, condi_P, Pred_P ] = evaluate_detector( bbox_ms, prob2 );
        total_TP(count,:) = TP_num;
        total_condi_P(count,:) = condi_P;
        total_Pred_P(count,:) = Pred_P;
        clear Xte Yte
    end
end


% Summing the statistics over all faces images.
sTP = sum(total_TP);
sCP = sum(total_condi_P);
sPP = sum(total_Pred_P);

% Compute the Precision
% TP is the number of intersection betweem recognized faces and the
% actual faces

Precision = sTP./sPP;       % TP/(The number of recognized faces)
Recall = sTP./sCP;          % TP/(The number of actual faces)

% Ploting the Precision-Recall curve. Normally, the yaxis is the Precision
% and xaxis is the Recall.
figure
plot(Recall, Precision)
xlabel('Recall');
ylabel('Precision');


% Interpolated Average Precision
AP = VOCap(Recall', Precision');
disp(num2str(AP))