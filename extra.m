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


%%

disp('Extracting features...')

hog  = true;
lbp  = true;

hog_cellSize   = 8;
lbp_cellSize   = 4;

max_resize = 1.0;
min_resize = 0.8;
threshold = 0.0;


%% Single/Multi-Scale Sliding Window
%==========================================================================
% Evaluating your detector and the sliding window.
% -It is okay to only use a single-scale sliding window for this assignment.
% However, a better performance would be required a multi-scale sliding
% window due to the different face size in real image.
%                                (You should finish this part by yourself)
%==========================================================================

load('face_detector.mat')

img = imread('test.png');
plt_img = img;
if size(img, 3)>1, img = rgb2gray(img); end
window_size = [64 64];

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

Xte = [];
Xte = cat(2, Xte, te_hog_vectors);
Xte = cat(2, Xte, te_lbp_vectors);

bbox_ms = [];

for p = 1:length(patches)
    for j = 1:size(patches{p}, 3)
        bbox_ms = [bbox_ms; temp_bbox{p}(j, :)];
    end
end

addpath('library/liblinear-2.1/windows/');

% Get the positive probability for proposed faces
[predicted_label, ~, prob_estimates] = predict(zeros(size(Xte, 1), 1), sparse(Xte), Mdl);
l = predicted_label;
score = prob_estimates;
prob2 = score(:, 1);

% Setting a threshold to pick the proposed face images
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
% rectangle('Position', [83, 92, 166-83, 175-92],...
%     'EdgeColor', 'r', 'LineWidth', 3)
%======================================================================
end


%%

disp('Recognition: Extracting features...')

Xtr = []; Ytr = [];
Xva = []; Yva = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% -tr_img_sample/va_img_sample:
% The data is store in a N-by-3 cell array. The first dimension of the cell
% array is the cropped face images. The second dimension is the name of the
% image and the third dimension is the class label for each image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hog  = false;
lbp  = false;
bof  = false;
nn   = true;
pca_ = true;

hog_cellSize   = 8;
lbp_cellSize   = 8;
vocab_size     = 250;
pca_components = 250;


%% Part I: Face Recognition: Who is it?
%==========================================================================
% The aim of this task is to recognize the person in the image(who is he).
% We train a multiclass classifer to recognize who is the person in this
% image.
% - Propose the patches of the images
% - Recognize the person (multiclass)
%==========================================================================


disp('Recognition: Extracting features...')

Xva = []; 
Yva = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% -tr_img_sample/va_img_sample:
% The data is store in a N-by-3 cell array. The first dimension of the cell
% array is the cropped face images. The second dimension is the name of the
% image and the third dimension is the class label for each image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./models/fr_model.mat');
load('./data/face_recognition/face_recognition_data_te.mat');

lbp_cellSize   = 8;
pca_components = 250;

va_lbp_vectors = zeros(length(va_img_sample), lbp_cellSize * lbp_cellSize * 58);
for i =1:length(va_img_sample)
    temp = single(va_img_sample{i,1})/255;
    temp = vl_lbp(temp, lbp_cellSize);
    va_lbp_vectors(i, :) = temp(:)';
end

nn_vector_size = 2622;
va_nn_vectors = zeros(length(va_img_sample), nn_vector_size);

h = waitbar(0, 'Initializing waitbar...', 'Name', 'Recognition: Extracting features...');

for i =1:length(va_img_sample)
    temp = single(va_img_sample{i,1}); % 255 range.
    temp = imresize(temp, net.meta.normalization.imageSize(1:2));
    temp = repmat(temp, [1, 1, 3]);
    temp = bsxfun(@minus, temp, net.meta.normalization.averageImage);
    temp = vl_simplenn(net, temp);
    temp = squeeze(temp(37).x);
    temp = temp./norm(temp,2);
    va_nn_vectors(i, :, :) = temp(:)';

    perc = i / length(va_img_sample);
    waitbar(perc, h, sprintf('%1.3f%%  Complete', perc * 100));
end

close(h);

Yva = zeros(length(va_img_sample), 1);
for i =1:length(va_img_sample)
    Yva(i) = va_img_sample{i, 3};
end

%% Build data for training from extracted features
Xva = [Xva va_nn_vectors];

% PCA
Xva = bsxfun(@minus ,Xva, mean(Xva));
Xva = Xva * coeff;

Xva = double(Xva);

%% Train the recognizer and evaluate the performance

% model = train(double(Ytr), sparse(double(Xtr)));
[predicted_label, ~, prob_estimates] = predict(zeros(size(Xva, 1), 1), sparse(Xva), model);
l = predicted_label;
prob = prob_estimates;

% Compute the accuracy
acc = mean(l==Yva)*100;

fprintf('The accuracy of face recognition is:%.2f \n', acc)