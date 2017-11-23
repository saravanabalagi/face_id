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
val_dir{2} = './data/face_detection/te_raw_images/';            % Validation data (For performance evaluation)
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

hog  = true;
lbp  = true;
nn   = false;
pca_ = false;
normalise = false;

hog_cellSize   = 8;
lbp_cellSize   = 4;
pca_components = 1000;

max_resize = 1.0;
min_resize = 0.8;
threshold = 0.0;

nPosFace = length(face_images_dir);

nNegFace = 0;
imset = imageSet(images_dir{2}, 'recursive');

for i=1:length(imset)
   for j = 1:imset(i).Count
        nNegFace = nNegFace + 1;
   end
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

overall_total = 0;

for k = 1:length(val_file2)
    for u = 1:length(imset(k).Count)
        overall_total = overall_total + 1;
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

        % Sliding window function
        [patches, temp_bbox] = sw_detect_face(img, window_size, max_resize, 8);

        for p = max_resize - 0.1:-0.1:min_resize
            [temp_patches, temp_bbox2] = sw_detect_face(img, window_size, p, 8);
            patches = cat(1, patches, temp_patches);
            temp_bbox = cat(1, temp_bbox, temp_bbox2);
        end
        
        % Extract the features for each patch
        total = 0;

        for p = 1:length(patches)
            for j = 1:size(patches{p}, 3)
                total = total + 1;
            end
        end

        if true(hog)
            v = resize_size(1) / hog_cellSize;
            te_hog_vectors = zeros(total, v * v * 31);

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
            v = resize_size(1) / lbp_cellSize;
            te_lbp_vectors = zeros(total, v * v * 58);

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
        
        if true(normalise)
            normr(Xte);
        end

        % Get the positive probability for proposed faces
        [predicted_label, ~, prob_estimates] = predict(zeros(size(Xte, 1), 1), sparse(Xte), Mdl);
        l = predicted_label;
        score = prob_estimates;
        prob2 = score(:, 1);
    
        % Getting the True positive, condition positive, predicted positive
        % number for evaluating the algorithm performance via Average 
        [ TP_num, condi_P, Pred_P ] = evaluate_detector( bbox_ms, prob2 );
        total_TP(count,:) = TP_num;
        total_condi_P(count,:) = condi_P;
        total_Pred_P(count,:) = Pred_P;
        
        perc = count / overall_total;
        waitbar(perc, h, sprintf('%1.3f%%  Complete', perc * 100));
                
        clear Xte Yte
    end
end

close(h);


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