% AT&T dataset
%dataset = FaceDataset('att_faces', 7/10, [38, 31], 30);
%save att dataset;
clear
load dataset

% SDPAD
[X, y] = dataset.generateData('difference');
spm = SPM(X{1}, y{1}, 1);
SS = SDPAD(spm.A, spm.b, spm.c);
SSA = SS(K.s - dataset.pca_dim + 1:K.s, K.s - dataset.pca_dim + 1:K.s);
save sdpad SS;

X2 = bsxfun(@rdivide, X{2}, sqrt(sum(X{2}.^2, 2)));
Csdpad = zeros(size(X2, 1), 1);
for i = 1:size(X2, 1)
   Csdpad(i) = (X2(i, :) * SSA * X2(i, :)' < 0.5) + 1;
end
acc = sum(Csdpad == y{2}) / size(y{2}, 1);
disp(acc);


% Yale dataset
%dataset = FaceDataset('yale_faces', 8/11, [50, 38], 35);
%save yale dataset;
clear
load yale

% SDPAD
[X, y] = dataset.generateData('difference');
spm = SPM(X{1}, y{1}, 1);
SS = SDPAD(spm.A, spm.b, spm.c);
SSA = SS(K.s - dataset.pca_dim + 1:K.s, K.s - dataset.pca_dim + 1:K.s);
save sdpad2 SS;

X2 = bsxfun(@rdivide, X{2}, sqrt(sum(X{2}.^2, 2)));
Csdpad = zeros(size(X2, 1), 1);
for i = 1:size(X2, 1)
   Csdpad(i) = (X2(i, :) * SSA * X2(i, :)' < 0.5) + 1;
end
acc = sum(Csdpad == y{2}) / size(y{2}, 1);
disp(acc);

% Original
%pairs = [dataset.origtest(dataset.same{2}); dataset.origtest(dataset.diff{2});];
%disp(pairs(Csvm == y{2} & Csvm ~= Csedumi, :));
