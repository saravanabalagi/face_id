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

% if ~exist('net')
% Setup MatConvNet.
addpath(genpath('./library/matconvnet/matlab'))
vl_setupnn();

% Load the VGG-Face model.
modelPath = fullfile(vl_rootnn,'data','models','vgg-face.mat') ;
if ~exist(modelPath)
  fprintf('Downloading the VGG-Face model ... this may take a while\n') ;
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
    'http://www.vlfeat.org/matconvnet/models/vgg-face.mat', ...
    modelPath) ;
end

% Load the model and upgrade it to MatConvNet current version.
net = load(modelPath);
net = vl_simplenn_tidy(net);


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

hog  = false;
lbp  = false;
nn   = true;
pca_ = false;

hog_cellSize   = 8;
lbp_cellSize   = 8;
vocab_size     = 250;
pca_components = 2000;

nPosFace = length(face_images_dir);

nNegFace = 0;
imset = imageSet(images_dir{2}, 'recursive');

for i=1:length(imset)
   parfor j = 1:imset(i).Count
        nNegFace = nNegFace + 1;
    end
end


%%

if true(hog)
    hog_vectors = zeros(nPosFace + nNegFace, hog_cellSize * hog_cellSize * 31);

    for i = 1:nPosFace
        temp = imresize(imread([images_dir{1}, face_images_dir(i).name]), resize_size); 
        temp = single(temp)/255;
        temp = vl_hog(temp, hog_cellSize);
        hog_vectors(i, :) = temp(:)';
    end
    
    hog_iter = 1;

    for i = 1:length(imset)
        for j = 1:imset(i).Count
            temp = imresize(read(imset(i), j), resize_size);
            temp = single(temp)/255;
            temp = vl_hog(temp, hog_cellSize);
            hog_vectors(nPosFace + hog_iter, :) = temp(:)';
            hog_iter = hog_iter + 1;
        end
    end
end


%%

if true(lbp)
    lbp_vectors = zeros(nPosFace + nNegFace, lbp_cellSize * lbp_cellSize * 58);

    for i = 1:nPosFace
        temp = imresize(imread([images_dir{1}, face_images_dir(i).name]), resize_size); 
        temp = single(temp)/255;
        temp = vl_lbp(temp, lbp_cellSize);
        lbp_vectors(i, :) = temp(:)';
    end
    
    lbp_iter = 1;
    
    for i = 1:length(imset)
        for j = 1:imset(i).Count
            temp = imresize(read(imset(i), j), resize_size);
            temp = single(temp)/255;
            temp = vl_lbp(temp, lbp_cellSize);
            lbp_vectors(nPosFace + lbp_iter, :) = temp(:)';
            lbp_iter = lbp_iter + 1;
        end
    end
end


%%

if true(nn)
    if ~exist('nn_vectors', 'var')
        if exist(fullfile('data/face_detection/nn_vectors/training/', 'tr_nn_vectors.mat'), 'file') == 2
            nn_vectors = load(fullfile('data/face_detection/nn_vectors/training/', 'tr_nn_vectors.mat'));
            nn_vectors = nn_vectors.nn_vectors;
            disp('Training neural net vectors loaded from storage');
        else
            nn_vector_size = 2622;

            nn_vectors = zeros(nPosFace + nNegFace, nn_vector_size);

            h = waitbar(0, 'Initializing waitbar...', 'Name', 'Recognition: Extracting features...');

            for i = 1:nPosFace
                temp = imresize(imread([images_dir{1}, face_images_dir(i).name]), resize_size);
                temp = single(temp); % 255 range.
                temp = imresize(temp, net.meta.normalization.imageSize(1:2));
                temp = repmat(temp, [1, 1, 3]);
                temp = bsxfun(@minus, temp, net.meta.normalization.averageImage);
                temp = vl_simplenn(net, temp);
                temp = squeeze(temp(37).x);
                temp = temp./norm(temp,2);
                nn_vectors(i, :, :) = temp(:)';

                perc = i / (nPosFace + nNegFace);
                waitbar(perc, h, sprintf('%1.3f%%  Complete', perc * 100));
            end

            va_iter = 1;

            for i=1:length(imset)
                for j = 1:imset(i).Count
                    temp = imresize(read(imset(i), j), resize_size);
                    temp = single(temp); % 255 range.
                    temp = imresize(temp, net.meta.normalization.imageSize(1:2));
                    temp = repmat(temp, [1, 1, 3]);
                    temp = bsxfun(@minus, temp, net.meta.normalization.averageImage);
                    temp = vl_simplenn(net, temp);
                    temp = squeeze(temp(37).x);
                    temp = temp./norm(temp,2);
                    nn_vectors(nPosFace + va_iter, :, :) = temp(:)';

                    perc = (nPosFace + va_iter) / (nPosFace + nNegFace);
                    waitbar(perc, h, sprintf('%1.3f%%  Complete', perc * 100));

                    va_iter = va_iter + 1;
                end
            end

            close(h);
            
            % Save output
            save('data/face_detection/nn_vectors/training/tr_nn_vectors.mat', 'nn_vectors');
        end
    end
end


%%

Xtr = [];
Xva = [];

if true(hog)
    Xtr = cat(2, Xtr, hog_vectors);
end

if true(lbp)
    Xtr = cat(2, Xtr, lbp_vectors);
end

if true(nn)
    Xtr = cat(2, Xtr, nn_vectors);
end

Ytr = [ones(nPosFace, 1); -1 * ones(nNegFace,1)];


%%

if true(pca_)
    [coeff, score, latent, ~, explained] = pca(Xtr, 'NumComponents', pca_components);

    Xtr = score;
end


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
% Mdl = fitcknn(Xtr, Ytr, 'NumNeighbors', 3);
Mdl = fitcsvm(Xtr, Ytr);

% model = train(double(Ytr), sparse(double(Xtr)));

% Clear the training X and Y to save memory.
clear Xtr Ytr

% save your trained model for evaluation.
save('face_detector.mat', 'Mdl')


%% Single/Multi-Scale Sliding Window
%==========================================================================
% Evaluating your detector and the sliding window.
% -It is okay to only use a single-scale sliding window for this assignment.
% However, a better performance would be required a multi-scale sliding
% window due to the different face size in real image.
%                                (You should finish this part by yourself)
%==========================================================================

load('face_detector.mat')
for k = 1:length(val_file)
    img = imread([val_dir{1} val_file(k).name]);
    plt_img = img;
    if size(img, 3)>1, img = rgb2gray(img); end
    window_size = [64 64];
    
    % Sliding window function
    [patches, temp_bbox] = sw_detect_face(img, window_size, 1, 32);   
    
    % Extract the features for each patch
    total = 0;
    
    for p = 1:length(patches)
        for j = 1:size(patches{p}, 3)
            total = total + 1;
        end
    end
    
    if true(hog)
        te_hog_vectors = zeros(total, hog_cellSize * hog_cellSize * 31);
        
        hog_iter = 1;

        for p = 1:length(patches)
            for j = 1:size(patches{p}, 3)
                temp = single(patches{p}(:, :, j))/255;
                temp = vl_hog(temp, hog_cellSize);
                te_hog_vectors(hog_iter, :) = temp(:)';
                hog_iter = hog_iter + 1;
            end
        end
    end
    
    if true(lbp)
        te_lbp_vectors = zeros(total, lbp_cellSize * lbp_cellSize * 58);
        
        lbp_iter = 1;

        for p = 1:length(patches)
            for j = 1:size(patches{p}, 3)
                temp = single(patches{p}(:, :, j))/255;
                temp = vl_lbp(temp, lbp_cellSize);
                te_lbp_vectors(lbp_iter, :) = temp(:)';
                lbp_iter = lbp_iter + 1;
            end
        end
    end
    
    if true(nn)
        if exist(strcat('data/face_detection/nn_vectors/visualization/vi_nn_vectors_', int2str(k), '.mat'), 'file') == 2
            nn_vectors = load(strcat('data/face_detection/nn_vectors/visualization/vi_nn_vectors_', int2str(k), '.mat'));
            te_nn_vectors = nn_vectors.te_nn_vectors;
            disp(strcat('Visualization neural net vectors_', int2str(k), '_loaded from storage'));
        else
            nn_vector_size = 2622;

            te_nn_vectors = zeros(total, nn_vector_size);

            nn_iter = 1;

            for p = 1:length(patches)
                for j = 1:size(patches{p}, 3)
                    temp = single(patches{p}(:, :, j)); % 255 range.
                    temp = imresize(temp, net.meta.normalization.imageSize(1:2));
                    temp = repmat(temp, [1, 1, 3]);
                    temp = bsxfun(@minus, temp, net.meta.normalization.averageImage);
                    temp = vl_simplenn(net, temp);
                    temp = squeeze(temp(37).x);
                    temp = temp./norm(temp, 2);
                    te_nn_vectors(nn_iter, :, :) = temp(:)';
                    nn_iter = nn_iter + 1;
                end
            end
            
            % Save output
            save(strcat('data/face_detection/nn_vectors/visualization/vi_nn_vectors_', int2str(k), '.mat'), 'te_nn_vectors');
        end
    end
    
    Xte = [];

    if true(hog)
        Xte = cat(2, Xte, te_hog_vectors);
    end

    if true(lbp)
        Xte = cat(2, Xte, te_lbp_vectors);
    end

    if true(nn)
        Xte = cat(2, Xte, te_nn_vectors);
    end
    
    bbox_ms = [];
    
    for p = 1:length(patches)
        for j = 1:size(patches{p}, 3)
            bbox_ms = [bbox_ms; temp_bbox{p}(j, :)];
        end
    end
    
    if true(pca_)
        Xte = bsxfun(@minus, Xte, mean(Xte));
        Xte = Xte * coeff;
    end

    % Get the positive probability for proposed faces
    [l, score] = predict(Mdl, Xte);
    prob2 = score(:, 2);
    
    % Setting a threshold to pick the proposed face images
    threshold = 0.5;
    threshold_bbox = bbox_ms(prob2 > threshold, :);
    prob3 = prob2(prob2 > threshold, :);

    % Remove the redundant boxes via non-maximum supression.
    % - The bbox is the top-left x, y, height, width of the patches.
    % - prob2 is the confidence of the patches
    [selectedBbox, selectedScore] = selectStrongestBbox(threshold_bbox, prob3, 'OverlapThreshold', 0.3, 'RatioType', 'Union');

    % Visualise the test images
    bbox_position = selectedBbox;
    figure
    imshow(plt_img)
    hold on
    for i = 1:size(bbox_position, 1)
    rectangle('Position', [bbox_position(i, 2),bbox_position(i, 1),bbox_position(i, 3:4)],...
        'EdgeColor', 'b', 'LineWidth', 3)
    
    % This is the bounding box of ground truth. You should not modify this
    % part
    %======================================================================
    rectangle('Position', [83, 92, 166-83, 175-92],...
        'EdgeColor', 'r', 'LineWidth', 3)
    %======================================================================
    end
    saveas(gcf, [val_file(k).name(1:end-4), '_sw64.png'])
    clear Xte Yte
end


%% Evaluating your result on the val_datasets

load('face_detector.mat')

% Initialization of the true positive, condition positive and prediction
% positive number collection.

total_TP = zeros(length(val_file2), 100);
total_condi_P = zeros(length(val_file2), 100);
total_Pred_P = zeros(length(val_file2), 100);

imset = imageSet(val_dir{2}, 'recursive');
count = 0;

total = 0;

for k = 1:length(val_file2)
    for u = 1:length(imset(k).Count)
        total = total + 1;
    end
end

h = waitbar(0, 'Initializing waitbar...', 'Name', 'Validation: Extracting features...');

for k = 1:length(val_file2)
    for u = 1:length(imset(k).Count)
        count = count + 1;
        img = read(imset(k), u);
        plt_img = img;
        if size(img, 3)>1, img = rgb2gray(img); end
        window_size=[64 64];

        % Use sliding window to get multiple patches from the original image
        [patches, temp_bbox] = sw_detect_face(img, window_size, 1, 8);    
        
        % Extract the features for each patch
        total = 0;

        for p = 1:length(patches)
            for j = 1:size(patches{p}, 3)
                total = total + 1;
            end
        end

        if true(hog)
            te_hog_vectors = zeros(total, hog_cellSize * hog_cellSize * 31);

            hog_iter = 1;

            for p = 1:length(patches)
                for j = 1:size(patches{p}, 3)
                    temp = single(patches{p}(:, :, j))/255;
                    temp = vl_hog(temp, hog_cellSize);
                    te_hog_vectors(hog_iter, :) = temp(:)';
                    hog_iter = hog_iter + 1;
                end
            end
        end

        if true(lbp)
            te_lbp_vectors = zeros(total, lbp_cellSize * lbp_cellSize * 58);

            lbp_iter = 1;

            for p = 1:length(patches)
                for j = 1:size(patches{p}, 3)
                    temp = single(patches{p}(:, :, j))/255;
                    temp = vl_lbp(temp, lbp_cellSize);
                    te_lbp_vectors(lbp_iter, :) = temp(:)';
                    lbp_iter = lbp_iter + 1;
                end
            end
        end

        if true(nn)
            if exist(strcat('data/face_detection/nn_vectors/validation/va_nn_vectors_', int2str(k), '_', int2str(u), '.mat'), 'file') == 2
                nn_vectors = load(strcat('data/face_detection/nn_vectors/validation/va_nn_vectors_', int2str(k), '_', int2str(u), '.mat'));
                te_nn_vectors = nn_vectors.te_nn_vectors;
                disp(strcat('Validation neural net vectors_', int2str(k), '_', int2str(u), '_loaded from storage'));
            else
                nn_vector_size = 2622;

                te_nn_vectors = zeros(total, nn_vector_size);

                nn_iter = 1;

                for p = 1:length(patches)
                    for j = 1:size(patches{p}, 3)
                        temp = single(patches{p}(:, :, j)); % 255 range.
                        temp = imresize(temp, net.meta.normalization.imageSize(1:2));
                        temp = repmat(temp, [1, 1, 3]);
                        temp = bsxfun(@minus, temp, net.meta.normalization.averageImage);
                        temp = vl_simplenn(net, temp);
                        temp = squeeze(temp(37).x);
                        temp = temp./norm(temp, 2);
                        te_nn_vectors(nn_iter, :, :) = temp(:)';
                        nn_iter = nn_iter + 1;
                    end
                end
                
                % Save output
                save(strcat('data/face_detection/nn_vectors/validation/va_nn_vectors_', int2str(k), '_', int2str(u), '.mat'), 'te_nn_vectors');
            end
        end

        Xte = [];

        if true(hog)
            Xte = cat(2, Xte, te_hog_vectors);
        end

        if true(lbp)
            Xte = cat(2, Xte, te_lbp_vectors);
        end

        if true(nn)
            Xte = cat(2, Xte, te_nn_vectors);
        end

        bbox_ms = [];

        for p = 1:length(patches)
            for j = 1:size(patches{p}, 3)
                bbox_ms = [bbox_ms; temp_bbox{p}(j, :)];
            end
        end

        if true(pca_)
            Xte = bsxfun(@minus, Xte, mean(Xte));
            Xte = Xte * coeff;
        end

        % Get the positive probability
%         Xte = Xte';
        %[~,~,d] = liblinearpredict(ones(size(Xte,1),1),sparse(double(Xte)),detector);
        %prob2 = 1./(1+exp(-d));
        %prob = [1-prob2, prob2];

        [~,score] = predict(Mdl, Xte);
        prob2 = score(:, 2);
    
        % Getting the True positive, condition positive, predicted positive
        % number for evaluating the algorithm performance via Average 
        [ TP_num, condi_P, Pred_P ] = evaluate_detector( bbox_ms, prob2 );
        total_TP(count, :) = TP_num;
        total_condi_P(count, :) = condi_P;
        total_Pred_P(count, :) = Pred_P;
        
        perc = count / total;
        waitbar(perc, h, sprintf('%1.3f%%  Complete', perc * 100));
                
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
