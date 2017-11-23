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

Xtr = []; Ytr = [];
Xva = []; Yva = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% -tr_img_sample/va_img_sample:
% The data is store in a N-by-3 cell array. The first dimension of the cell
% array is the cropped face images. The second dimension is the name of the
% image and the third dimension is the class label for each image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./data/face_recognition/face_recognition_data_tr.mat');
load('./data/face_recognition/face_recognition_data_te.mat');

hog  = false;
lbp  = false;
bof  = false;
nn   = true;
pca_ = true;

hog_cellSize   = 8;
lbp_cellSize   = 8;
vocab_size     = 250;
pca_components = 250;


%%

if true(hog)
    tr_hog_vectors = zeros(length(tr_img_sample), hog_cellSize * hog_cellSize * 31);
    va_hog_vectors = zeros(length(va_img_sample), hog_cellSize * hog_cellSize * 31);

    for i =1:length(tr_img_sample)
        temp = single(tr_img_sample{i,1})/255;
        temp = vl_hog(temp, hog_cellSize);
        tr_hog_vectors(i, :) = temp(:)';
    end

    for i =1:length(va_img_sample)
        temp = single(va_img_sample{i,1})/255;
        temp = vl_hog(temp, hog_cellSize);
        va_hog_vectors(i, :) = temp(:)';
    end
end


%%

if true(lbp)
    tr_lbp_vectors = zeros(length(tr_img_sample), lbp_cellSize * lbp_cellSize * 58);
    va_lbp_vectors = zeros(length(va_img_sample), lbp_cellSize * lbp_cellSize * 58);

    for i =1:length(tr_img_sample)
        temp = single(tr_img_sample{i,1})/255;
        temp = vl_lbp(temp, lbp_cellSize);
        tr_lbp_vectors(i, :) = temp(:)';
    end

    for i =1:length(va_img_sample)
        temp = single(va_img_sample{i,1})/255;
        temp = vl_lbp(temp, lbp_cellSize);
        va_lbp_vectors(i, :) = temp(:)';
    end
end


%%

if true(bof)
    if ~exist('data/face_recognition/images/training/')
        for i =1:length(tr_img_sample)
            temp = single(tr_img_sample{i,1})/255;
            name = strsplit(tr_img_sample{i,2}, '_');
            number = strsplit(name{3}, '.');
            foldername = ['data/face_recognition/images/training/', name{1}, '_', name{2}];
            w = warning('query','last');
            id = w.identifier;
            warning('off',id)
            mkdir(foldername);
            filename = strcat(foldername, '/', name(1), '_', name(2), '_', number{1}, '.png');
            imwrite(temp, filename{1});
        end
    end

    if ~exist('data/face_recognition/images/validation/')
        for i =1:length(va_img_sample)
            temp = single(va_img_sample{i,1})/255;
            name = strsplit(va_img_sample{i,2}, '_');
            number = strsplit(name{3}, '.');
            foldername = ['data/face_recognition/images/validation/', name{1}, '_', name{2}];
            mkdir(foldername);
            filename = strcat(foldername, '/', name(1), '_', name(2), '_', number{1}, '.png');
            imwrite(temp, filename{1});
        end
    end

    tr_imset = imageSet('data/face_recognition/images/training/', 'recursive');

    bag = bagOfFeatures(tr_imset, 'VocabularySize', vocab_size);

    tr_bof_vectors = zeros(length(tr_img_sample), vocab_size);
    va_bof_vectors = zeros(length(va_img_sample), vocab_size);

    for i =1:length(tr_img_sample)
        temp = single(tr_img_sample{i,1})/255;
        temp = encode(bag, temp);
        tr_bof_vectors(i, :) = temp(:)';
    end

    for i =1:length(va_img_sample)
        temp = single(va_img_sample{i,1})/255;
        temp = encode(bag, temp);
        tr_bof_vectors(i, :) = temp(:)';
    end
end


%%

if true(nn)
    if ~exist('tr_nn_vectors')
        if exist(fullfile('data/face_recognition/', 'nn_vectors.mat'), 'file') == 2
            nn_vectors = load(fullfile('data/face_recognition/', 'nn_vectors.mat'));
            tr_nn_vectors = nn_vectors.tr_nn_vectors;
            va_nn_vectors = nn_vectors.va_nn_vectors;
            disp('Neural net vectors loaded from storage');
        else
            nn_vector_size = 2622;

            tr_nn_vectors = zeros(length(tr_img_sample), nn_vector_size);
            va_nn_vectors = zeros(length(va_img_sample), nn_vector_size);

            h = waitbar(0, 'Initializing waitbar...', 'Name', 'Recognition: Extracting features...');

            for i =1:length(tr_img_sample)
                temp = single(tr_img_sample{i,1}); % 255 range.
                temp = imresize(temp, net.meta.normalization.imageSize(1:2));
                temp = repmat(temp, [1, 1, 3]);
                temp = bsxfun(@minus, temp, net.meta.normalization.averageImage);
                temp = vl_simplenn(net, temp);
                temp = squeeze(temp(37).x);
                temp = temp./norm(temp,2);
                tr_nn_vectors(i, :, :) = temp(:)';

                perc = i / (length(tr_img_sample) + length(va_img_sample));
                waitbar(perc, h, sprintf('%1.3f%%  Complete', perc * 100));
            end

            for i =1:length(va_img_sample)
                temp = single(va_img_sample{i,1}); % 255 range.
                temp = imresize(temp, net.meta.normalization.imageSize(1:2));
                temp = repmat(temp, [1, 1, 3]);
                temp = bsxfun(@minus, temp, net.meta.normalization.averageImage);
                temp = vl_simplenn(net, temp);
                temp = squeeze(temp(37).x);
                temp = temp./norm(temp,2);
                va_nn_vectors(i, :, :) = temp(:)';

                perc = (length(tr_img_sample) + i) / (length(tr_img_sample) + length(va_img_sample));
                waitbar(perc, h, sprintf('%1.3f%%  Complete', perc * 100));
            end

            close(h);
            
            % Save output
            save('data/face_recognition/nn_vectors.mat', 'tr_nn_vectors', 'va_nn_vectors');
        end
    end
end


%% Prepare Training set and validation set for required features

if true(hog)
    Xtr = cat(2, Xtr, tr_hog_vectors);
    Xva = cat(2, Xva, va_hog_vectors);
end

if true(lbp)
    Xtr = cat(2, Xtr, tr_lbp_vectors);
    Xva = cat(2, Xva, va_lbp_vectors);
end

if true(bof)
    Xtr = cat(2, Xtr, tr_bof_vectors);
    Xva = cat(2, Xva, va_bof_vectors);
end

if true(nn)
    Xtr = cat(2, Xtr, tr_nn_vectors);
    Xva = cat(2, Xva, va_nn_vectors);
end

Ytr = zeros(length(tr_img_sample), 1);

for i =1:length(tr_img_sample)
    Ytr(i) = tr_img_sample{i, 3};
end

Yva = zeros(length(va_img_sample), 1);

for i =1:length(va_img_sample)
    Yva(i) = va_img_sample{i, 3};
end


%% PCA
if true(pca_)
    [coeff,score,latent,~,explained] = pca(Xtr, 'NumComponents', pca_components);

    Xtr = score;

    Xva = bsxfun(@minus ,Xva, mean(Xva));
    Xva = Xva * coeff;
    disp('Finished feature extraction.')
end

%% Train the recognizer and evaluate the performance
Xtr = double(Xtr);
Xva = double(Xva);

% Train the recognizer
% model = fitcknn(Xtr,Ytr,'NumNeighbors',3);
% [l,prob] = predict(model,Xva);

addpath('library/liblinear-2.1/windows/');

model = train(double(Ytr), sparse(double(Xtr)));
[predicted_label, ~, prob_estimates] = predict(zeros(size(Xva, 1), 1), sparse(Xva), model);
l = predicted_label;
prob = prob_estimates;

% model = fitcecoc(Xtr, Ytr);
% [l,prob] = predict(model, Xva);

% Compute the accuracy
acc = mean(l == Yva) * 100;

fprintf('The accuracy of face recognition is:%.2f \n', acc)

%% Visualization the result of face recognition

data_idx = [1,30,50]; % The index of image in validation set
nSample = 3; % number of visualize data. maximum should be 3
% nPairs = length(data_idx); % unconment to get full size of 
visualise_recognition(va_img_sample,prob,Yva,data_idx,nSample)


% Image and Visual Computing Assignment 2 Part 2: Face Verification
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
