function [ features, metrics ] = custom_extractor( image )
%CUSTOM_EXTRACTOR Summary of this function goes here
%   Detailed explanation goes here

cellSize = 8;

hog = vl_hog(single(image)/255, cellSize);

features = hog;
metrics  = zeros(0,1);

end

