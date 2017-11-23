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
clc
clear all

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

%% Part II: Face Verification: 
%==========================================================================
% The aim of this task is to verify whether the two given people in the
% images are the same person. We train a binary classifier to predict
% whether these two people are actually the same person or not.
% - Extract the features
% - Get a data representation for training
% - Train the verifier and evaluate its performance
%==========================================================================

disp('Verification:Extracting features..')

cellSize = 8;
Xtr = [];
Xva = [];

nn_1_train = [];
nn_2_train = [];
nn_1_val = [];
nn_2_val = [];

lbp_1_train = [];
lbp_2_train = [];
lbp_1_val = [];
lbp_2_val = [];

load('./data/face_verification/face_verification_tr.mat')
% load('./data/face_verification/face_verification_va.mat')
load('./data/face_verification/face_verification_te.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% -tr_img_pair/va_img_pair:
% The data is store in a N-by-4 cell array. The first dimension of the cell
% array is the first cropped face images. The second dimension is the name 
% of the image. Similarly, the third dimension is another image and the
% fourth dimension is the name of that image.
% -Ytr/Yva: is the label of 'same' or 'different'
%%%%%%%%%%%%%%%%%

% Ytr2 = zeros(1800,2);

%% Extract Features

h = waitbar(0,'Name','Extracting features...','Initializing waitbar...');

% You should construct the features in here. (read, resize, extract)
for i =1:length(tr_img_pair)
    
    % First Image
    im_ = single(tr_img_pair{i,1}) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
    res = vl_simplenn(net, im_) ;
    output1 = squeeze(res(37).x);
    output1 = output1./norm(output1,2);
    nn_1_train = [nn_1_train; output1(:)'];
    
    temp = single(tr_img_pair{i,1})/255;
    lbp_1 = vl_lbp(temp, cellSize);
    lbp_1_train = [lbp_1_train; lbp_1(:)'];
    
    % Second Image
    im_ = single(tr_img_pair{i,3}) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
    res = vl_simplenn(net, im_) ;
    output2 = squeeze(res(37).x);
    output2 = output2./norm(output2,2);
    nn_2_train = [nn_2_train; output2(:)'];
    
    temp = single(tr_img_pair{i,3})/255;
    lbp_2 = vl_lbp(temp, cellSize);
    lbp_2_train = [lbp_2_train; lbp_2(:)'];
    
    perc = (i * 100) / (length(tr_img_pair) + length(va_img_pair));
    waitbar(perc/100,h,sprintf('%0.5f%%  Complete',perc))
end

for i =1:length(va_img_pair)
    
    % First Image
    im_ = single(va_img_pair{i,1}) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
    res = vl_simplenn(net, im_) ;
    output1 = squeeze(res(37).x);
    output1 = output1./norm(output1,2);
    nn_1_val = [nn_1_val; output1(:)'];
    
    temp = single(va_img_pair{i,1})/255;
    lbp_1 = vl_lbp(temp, cellSize);
    lbp_1_val = [lbp_1_val; lbp_1(:)'];
    
    % Second Image
    im_ = single(va_img_pair{i,3}) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
    res = vl_simplenn(net, im_) ;
    output2 = squeeze(res(37).x);
    output2 = output2./norm(output2,2);
    nn_2_val = [nn_2_val; output2(:)'];
    
    temp = single(va_img_pair{i,3})/255;
    lbp_2 = vl_lbp(temp, cellSize);
    lbp_2_val = [lbp_2_val; lbp_2(:)'];
    
    perc = ((length(tr_img_pair) + i) * 100) / (length(tr_img_pair) + length(va_img_pair));
    waitbar(perc/100,h,sprintf('%0.5f%%  Complete',perc))
end

%% Build data for training from extracted features
Xtr = [sqrt(sum((lbp_1_train-lbp_2_train)'.^2))' sqrt(sum((nn_1_train-nn_2_train)'.^2))'];
Xva = [sqrt(sum((lbp_1_val-lbp_2_val)'.^2))' sqrt(sum((nn_1_val-nn_2_val)'.^2))'];

Xtr = double(Xtr);
Xva = double(Xva);

%% PCA
pca_components = min(size(Xtr,2));
[coeff,score,latent,~,explained] = pca(Xtr, 'NumComponents', pca_components);

Xtr = score;

Xva = bsxfun(@minus ,Xva, mean(Xva));
Xva = Xva * coeff;

%% Train the verifier and evaluate the performance

% Train the recognizer and evaluate the performance
addpath('library/liblinear-2.1/windows/');

model = train(double(Ytr), sparse(double(Xtr)));

[predicted_label, ~, prob_estimates] = predict(zeros(size(Xva, 1), 1), sparse(Xva), model);
l = predicted_label;
prob = prob_estimates;

% Compute the accuracy
acc = mean(l==Yva)*100;
fprintf('The accuracy of face verification is:%.2f \n', acc)

%% Visualization the result of face verification

data_idx = [100,200,300]; % The index of image in validation set
nPairs = 3; % number of visualize data. maximum is 3
% nPairs = length(data_idx); 
visualise_verification(va_img_pair,prob,Yva,data_idx,nPairs )
