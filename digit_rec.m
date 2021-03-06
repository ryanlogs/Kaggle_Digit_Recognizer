%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Kaggle Digit Recognizer using %%%
%%% Neural Network                %%%
%%% Author - Ryan Sequeira        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%================= Update these values ================= %%
hidden_layer1_size = 200;
lambda = 1.2;
iterations = 300;

%adding functions directory to load path
addpath("functions");
submission_name = "latest_submission.csv";
output_dir = "output";

disp("Starting Digit Recognizer");


%%% ================ Loaging training data ================ %%%
%data = load_csv('F:\coursera\MC learning\Kaggle\Digit Regognizer\train.csv',1000,784);
%y = data(:,1);
%X = data(:,2:end);
%save -binary 'trainData.mat' X,y;
load -binary 'data\trainData.mat'
y = trainData(:,1);
X = trainData(:,2:end);

%%% ================ Display Data ========================= %%% 
disp("\nDisplaying first 100 digits from training set \n");
display_data(X(1:100,:),28);
labels = reshape(y(1:100,1),10,10)';
disp(labels);

disp("\nHit Enter to start!!!\n");
pause;
disp("Starting Neural Network Training...\n");

%%% ================ Initializing neural network ========== %%%
disp("\nInitializing Neural Network Parameters ...\n");

%neural network [784, 50, 50, 10]

ip_layer_size = 784;
%hidden_layer1_size = 200;
op_layer_size = 10;

initial_Theta1 = randInitializeWeights(ip_layer_size,hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size,op_layer_size);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%%% ================ Training NN ========================== %%%
disp("\nTraining Neural Network... \n");

options = optimset('MaxIter', iterations);
%lambda = 1.2;
network = [ip_layer_size; hidden_layer1_size; op_layer_size];

% Obtain Theta1 and Theta2 back from nn_params

costFunction = @(p) nnCostFunction(p, ...
                                   network, ...
								   X, y, lambda);
								   
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);	


%obtain parameters
Theta1 = reshape(nn_params(1:hidden_layer1_size * (ip_layer_size+1)), ...
						hidden_layer1_size, ip_layer_size+1);
read = 	hidden_layer1_size * (ip_layer_size+1);
						
Theta2 = reshape(nn_params(read + 1 : end ), ...
						op_layer_size, hidden_layer1_size+1);	

%%% ================ Test NN accuracy ===================== %%%
pred = predict(Theta1, Theta2, X);						
fprintf('\nTraining Set Accuracy: %f lambda: %f\n', mean(double(pred == y)) * 100, lambda);

%%% ================ Test Data ============================ %%%
%data = load_csv('F:\coursera\MC learning\Kaggle\Digit Regognizer\train.csv',100,783);
%X = data(1:100,1:end);
load -binary 'data\testData.mat';
%X = testData
pred = predict(Theta1, Theta2, testData);

%display some samples
disp("\nDisplaying first 100 digits from test set \n");
display_data(testData(1:100,:),28);
labels = reshape(pred(1:100,1),10,10)';
disp(labels);

disp("Writing Test Output... \n");
%writing the headers first
output = sprintf("%s/%s",output_dir,submission_name);
out_id = fopen(output,"w+");
fprintf(out_id,"%s","ImageId,Label\n");
fclose(out_id);

out = (1:28000)';
out = [out pred];
dlmwrite (output, out, ",","-append");
%display_data(X(1:100,:),28);
%labels = reshape(pred,10,10)';
%disp(labels);	