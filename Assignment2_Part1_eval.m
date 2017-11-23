% Image and Visual Computing Assignment 2: Face Verification & Recognition
%==========================================================================
%   In this assignment, you are expected to use the previous learned method
%   to cope with face recognition and verification problem. The vl_feat, 
%   libsvm, liblinear and any other classification and feature extraction 
%   library are allowed to use in this assignment. The built-in matlab 
%   object-detection functionis not allowed. Good luck and have fun!
%
%                                               Released Date:   31/10/2017
%==========================================================================

%% Initialisation
%==========================================================================
% Add the path of used library.
% - The function of adding path of liblinear and vlfeat is included.
%==========================================================================
clear all
clc

run ICV_setup

% Hyperparameter of experiments
resize_size=[64 64];

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
addpath('library/liblinear-2.1/windows/');

% model = train(double(Ytr), sparse(double(Xtr)));
[predicted_label, ~, prob_estimates] = predict(zeros(size(Xva, 1), 1), sparse(Xva), model);
l = predicted_label;
prob = prob_estimates;

% Compute the accuracy
acc = mean(l==Yva)*100;

fprintf('The accuracy of face recognition is:%.2f \n', acc)
