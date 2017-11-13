function [patches, bbox_locations] = sw_detect_face(img, window_size, scale, stride)
% sw_multiscale_detect_face
% - This is a function to proposed the potential face images via moving the
% sliding window. 
%==========================================================================
% Output:
%   - patches: a cell to store every window_size proposed images. The size
%               of save images are H*W*N, where N is the number of sliding
%   - bbox_location: bounding box [x,y,height,width]
%--------------------------------------------------------------------------
% Input:
%   - real_image : The original images without resize
%   - window_size: The proposed sliding window size
%   - scale      : The scale of for each original image
%   - stride     : The steps between each save images
%==========================================================================

img = imresize(img, scale);

% Image dimensions
[image_y, image_x] = size(img);

% Window dimensions
window_y = window_size(1);
window_x = window_size(2);

% Number of windows to return
number_of_windows = floor(((image_y - window_y) / stride) + 1) * floor(((image_x - window_x) / stride) + 1);

% Initialise patches and bounding box locations for speed
s_patches = zeros(window_y, window_x, number_of_windows, 'uint8');
s_bbox_locations = zeros(number_of_windows, 4, 'uint8');

% Set count to 1, to keep track of the number of patches processed
count = 1;

% Iteratively save the patches.
for y = 1:stride:image_y - window_y + 1
    for x = 1:stride:image_x - window_x + 1
        % Image patch at bbox location
        s_patches(:,:,count) = img(y:y+window_y-1, x:x+window_x-1);
        
        % Bbox cords, top-left y, x, height, width
        s_bbox_locations(count, :) = [int16(y / scale), int16(x / scale), int16(window_y / scale), int16(window_x / scale)];     
        
        % Iterate processed count
        count = count + 1;
    end
end

% Set output
patches{1} = s_patches;
bbox_locations{1} = s_bbox_locations;

end

