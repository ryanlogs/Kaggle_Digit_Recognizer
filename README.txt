After going to the directory where the code is
(use the cd '<path_to_directory>')

1. Generate the data.
   To generate the trainData.mat and testData.mat use generate_mat_files script.
   example: 
      $> generate_mat_files

   This step needs to be done only once. The testData.mat and trainData.mat will be saved in data/ directory
	
2. Run the Neural Network.
   Before you run, edit the digit_rec script. Change the values of the following variables:
    a. hidden_layer1_size  - Size of hidden layer (the NN contains only 1 hidden layer).
    b. lambda - Regularization parameter. Changing this value will yield different results. Keep tuning it till u find the best result.
    c. iterations - The number of iteratons the NN should train. Initial test different values of lambda for 50 to 100 iterations. Once a suitable lambda value is found increase the number of iterations (depending on your hardware).			 
    d.	submission_name - The name of the output file. Saved in output directory by default.
	
   example:
      $> digit_rec
