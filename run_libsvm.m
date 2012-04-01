% AT&T dataset
%dataset = FaceDataset('att_faces', 7/10, [38, 31], 30);
%save att dataset;
clear
load att

% LibSVM
[X, y] = dataset.generateData('euclidean');
tic;
model = svmtrain(y{1}, X{1}, '-t 0 -e 0.05');
toc;
save libsvm model;
[Csvm, acc] = svmpredict(y{2}, X{2}, model);


% Yale dataset
%dataset = FaceDataset('yale_faces', 8/11, [50, 38], 35);
%save yale dataset;
clear
load yale

% LibSVM
[X, y] = dataset.generateData('euclidean');
tic;
model = svmtrain(y{1}, X{1}, '-t 0 -e 0.05');
toc;
save libsvm2 model;
[Csvm, acc] = svmpredict(y{2}, X{2}, model);
