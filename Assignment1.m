% Image and Visual Computing Assignment 1: Face Detection-Recognition
%==========================================================================
%   In this assignment, you are expected to use the previous learned method
%   to cope with face detection problem. The vl_feat, libsvm, liblinear and
%   any other classification and feature extraction library are allowed to 
%   use in this assignment. The built-in matlab object-detection function
%   is not allowed. Good luck and have fun!
%
%                                               Released Date:   31/10/2017
%==========================================================================

%% Initialisation
%==========================================================================
% Add the path of used library.
% - The function of adding path of liblinear and vlfeat is included.
% - The use image directory is also included in this part.
% - image_dir{1} is the training positive face images(resize).
% - image_dir{2} is the training negative non-face images(resize).
% - val_dir is the validation set of real images
%==========================================================================
clear all
close all
clc
run ICV_setup
% The relevant data directory
images_dir{1} = './data/face_detection/cropped_faces/';     % positive samples directory
images_dir{2} = './data/face_detection/non_faces_images/';  % negative samples directory
face_images_dir = dir(images_dir{1});
face_images_dir(1:2)=[];

val_dir{1} = './data/face_detection/val_face_detection_images/'; % Validation data (For visualization purpose).
val_file = dir(val_dir{1});
val_file(1:2)=[];
val_dir{2} = './data/face_detection/val_raw_images/';            % Validation data (For performance evaluation)
val_file2 = dir(val_dir{2});
val_file2(1:2)=[];

% Hyperparameter of experiments
resize_size=[64 64];


%% Feature Extraction for Face Detection
%==========================================================================
% Use the HoG features for face detection. 
% - You should read the images and convert any color images into gray. For
% reading all images in subdirectory, you can use matlab function
% 'imageSet('./your data path', 'recursive'). The former quotes is the
% directory to your saved data and the latter 'recursive' is the
% hyparameter of this function.
% - Extract HoG intesert points for training image. You can use either HoG 
% or LBP as features. Generally, using HoG with linear SVM can achieve
% reasonable performance which has already been verified from several
% papers. It is okay to use both vl_feat or your own function. You will get
% bonus points if you are using your own code to get the HoG and LBP
% features.
%                                (You should finish this part by yourself)
%==========================================================================

% Feature Extraction

disp('Extracting features...')


cellSize = 8;
Xtr = [];
% Read and Resize the face images. Extract the features here.
% nFace = 250; 
nFace = length(face_images_dir);  % Uncomment it to get more training data
parfor i=1:nFace                     % Uncomment it to try parallel for-loop to speed up the experiments
% for i=1:nFace
    temp = imresize(imread([images_dir{1},face_images_dir(i).name]),resize_size);   
    temp = single(temp)/255;
    hog = vl_hog(temp, cellSize);
    lbp = vl_lbp(temp, cellSize);
%     rendered = vl_hog('render', temp);
%     imshow(rendered);
    Xtr = [Xtr;[hog(:);lbp(:)]'];
end

% disp(size(Xtr));
% imshow(reshape(Xtr(1,:),[64,64]));

% load non_face_images
non_face_images_dir = dir(images_dir{2});
non_face_images_dir(1:2)=[];
count = 0;
imset = imageSet(images_dir{2}, 'recursive');

% Read and Resize the non-face images. Extract the features here.
for i=1:length(imset)
   parfor j = 1:imset(i).Count     % Uncomment it to get more training data
%    parfor j = 1:imset(i).Count     % Uncomment it to get more training data
%     for j = 1:min(imset(i).Count,5)
        count = count+1;
        temp = imresize(read(imset(i),j),resize_size);
        if size(temp,3)>1, temp = rgb2gray(temp); end
        temp = single(temp)/255;
        hog = vl_hog(temp, cellSize);
        lbp = vl_lbp(temp, cellSize);
        %     rendered = vl_hog('render', temp);
        %     imshow(rendered);
        Xtr = [Xtr;[hog(:);lbp(:)]'];
    end
end

% Create the labelled image
Ytr = [ones(nFace,1);-1*ones(count,1)];
% disp(size(Xtr));
% disp(size(Ytr));

%% Training the Face Detector
%==========================================================================
% Training linear SVM as a face detector. 
% - It is okay to use all matlab, vlfeat, liblinear built-in function to 
% train the SVM. A start hyperparameter of liblinear is '-s 2 -B 1'
% It is free to explore any other hyperparameter combination for getting a 
% better results. The primal form of the SVM parameter '-s 2' is 
% recommended due to the large amount of training data.
%                                (You should finish this part by yourself)
%==========================================================================

disp('Training the face detector..')
% Mdl = fitcknn(Xtr,Ytr, 'NumNeighbors', 3); % The model could be improved by changing the KNN classifier to SVM.
Mdl = fitcsvm(Xtr,Ytr);

% Clear the training X and Y to save memory.
clear Xtr Ytr

% save your trained model for evaluation.
save('face_detector.mat','Mdl')

%% Single/Multi-Scale Sliding Window
%==========================================================================
% Evaluating your detector and the sliding window.
% -It is okay to only use a single-scale sliding window for this assignment.
% However, a better performance would be required a multi-scale sliding
% window due to the different face size in real image.
%                                (You should finish this part by yourself)
%==========================================================================
load('face_detector.mat')
for k=1:length(val_file)
    img = imread([val_dir{1} val_file(k).name]);
    plt_img=img;
    if size(img,3)>1, img = rgb2gray(img); end
    window_size=[64 64];
    
    [patches,temp_bbox] = sw_detect_face(img,window_size,1,32);
    
    Xte=[];
    bbox_ms = [];
    % Extract the feature for each patch
    for p=1:length(patches)
        for j = 1:size(patches{p},3)
            face_img = single(patches{p}(:,:,j))/255;
            hog = vl_hog(face_img, cellSize);
            lbp = vl_lbp(face_img, cellSize);
            Xte = [Xte [hog(:);lbp(:)]];
            bbox_ms = [bbox_ms;temp_bbox{p}(j,:)];
        end
    end

    % Get the positive probability for proposed faces
    Xte = Xte';    
    [l,score] = predict(Mdl,Xte);
    prob2 = score(:,2);
    
%     prob2 = score(:,1);
% %     mu = 0;
% %     sigma = 0.1;
%     for probi = 1:1:length(prob2)
%         prob2(probi) = 1/(1 + exp(-1 * abs(prob2(probi))));
% %         prob2(probi) = (-0.5 * (prob2(probi) - mu)/sigma) / (sigma * sqrt(2 * pi));
%     end
    
%     prob2 = score(:,1);
%     for probi = 1:1:length(prob2)
%         prob2(probi) = 1/(1 + exp(prob2(probi)));
%     end
    
%   Setting a threshold to pick the proposed face images
    threshold = 0.5;
    threshold_bbox=bbox_ms(prob2>threshold,:);
    prob3=prob2(prob2>threshold,:);

    % Remove the redundant boxes via non-maximum supression.
    % - The bbox is the top-left x,y, height,width of the patches.
    % - prob2 is the confidence of the patches
    [selectedBbox,selectedScore] = selectStrongestBbox(threshold_bbox,prob3,'OverlapThreshold',0.3, 'RatioType','Union');

    % Visualise the test images
    bbox_position = selectedBbox;
    figure
    imshow(plt_img)
    hold on
    for i=1:size(bbox_position,1)
    rectangle('Position', [bbox_position(i,2),bbox_position(i,1),bbox_position(i,3:4)],...
        'EdgeColor','b', 'LineWidth', 3)
    
    % This is the bounding box of ground truth. You should not modify this
    % part
    %======================================================================
    rectangle('Position', [83,92,166-83,175-92],...
        'EdgeColor','r', 'LineWidth', 3)
    %======================================================================
    end
    saveas(gcf, ['scratch/', val_file(k).name(1:end-4), '_sw64.png'])
    clear Xte Yte
end

%% Evaluating your result on the val_datasets

load('face_detector.mat')

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
                hog = vl_hog(face_img, cellSize);
                lbp = vl_lbp(face_img, cellSize);
                Xte = [Xte [hog(:);lbp(:)]];
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
        
%         prob2 = score(:,1);
%         for probi = 1:1:length(prob2)
%             prob2(probi) = 1/(1 + exp(prob2(probi)));
%         end
        
        
    
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


% Average Precision
AP = Precision;
AP(isnan(AP)) = 0;
AP = mean(AP); 
disp(num2str(AP))
