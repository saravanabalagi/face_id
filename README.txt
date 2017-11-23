######################################
##        Files for marking         ##
######################################

Assignment_1.m:
	Part 1: Face Detection

Assignment_2.m: 
	Part 2: Face Recognition and Verification

bonus.m:
	Bonus marks: Linked detector and recogniser, indentifies multiple individuals (drawn from validation dataset which the model has not seen before) correctly.
	

######################################
##            Dev Files             ##
######################################

We split Assignment_2.m into two files for developmental convenience

Assignment_2_Part1.m: 
	Face Recognition

Assignment_2_Part2.m: 
	Face Verification

VGG Face model Location: ./library/matconvnet/data/models/vgg-face.mat

######################################
##            Demo Files            ##
######################################

We made separate files for demo which loads the trained models

Assignment_1_eval.m: 
	Face Detection
	Loads model from ./face_detector.mat

Assignment_2_Part1_eval.m: 
	Face Recognition
	Loads model from ./models/fr_model.mat

Assignment_2_Part2_eval.m: 
	Face Verification
	Loads models from ./models/fv_model.mat