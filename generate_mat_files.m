% Run this script to generate testData and trainData

train_split_files = 21
test_split_files = 14

disp("Generating Data...\n");

%generating trainData.mat
disp("working on training data...\n");
tic
trainData = [];

for i = 1:train_split_files
	file = sprintf("data/train_split/train%d.csv",i);
	temp = dlmread(file,',');
	trainData = [trainData ; temp];
end

disp("Size of trainData");
disp(size(trainData));
disp("Saving to data/trainData.mat\n");
save -binary 'data/trainData.mat' trainData;
toc



%generating testData.mat
disp("working on test data...\n");
tic
testData = [];

for i = 1:test_split_files
	file = sprintf("data/test_split/test%d.csv",i);
	temp = dlmread(file,',');
	testData = [testData ; temp];
end

disp("Size of testData");
disp(size(testData));
disp("Saving to data/testData.mat\n");
save -binary 'data/testData.mat' testData;
toc;
