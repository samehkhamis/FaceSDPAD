% AT&T dataset
%dataset = FaceDataset('att_faces', 7/10, [38, 31], 30);
%save att dataset;
clear
load att

% SeDuMi
[X, y] = dataset.generateData('difference');
spm = SPM(X{1}, y{1}, 1);
K.s = [sqrt(size(spm.c, 1))];
pars.eps = 1e-6;
S = mat(sedumi(spm.A, spm.b, spm.c, K, pars));
SA = S(K.s - dataset.pca_dim + 1:K.s, K.s - dataset.pca_dim + 1:K.s);
save sedumi S;

X2 = bsxfun(@rdivide, X{2}, sqrt(sum(X{2}.^2, 2)));
Csedumi = zeros(size(X2, 1), 1);
for i = 1:size(X2, 1)
   Csedumi(i) = (X2(i, :) * SA * X2(i, :)' < 0.5) + 1;
end
acc = sum(Csedumi == y{2}) / size(y{2}, 1);
disp(acc);


% Yale dataset
%dataset = FaceDataset('yale_faces', 8/11, [50, 38], 35);
%save yale dataset;
clear
load yale

% SeDuMi
[X, y] = dataset.generateData('difference');
spm = SPM(X{1}, y{1}, 1);
K.s = [sqrt(size(spm.c, 1))];
pars.eps = 1e-6;
S = mat(sedumi(spm.A, spm.b, spm.c, K, pars));
SA = S(K.s - dataset.pca_dim + 1:K.s, K.s - dataset.pca_dim + 1:K.s);
save sedumi2 S;

X2 = bsxfun(@rdivide, X{2}, sqrt(sum(X{2}.^2, 2)));
Csedumi = zeros(size(X2, 1), 1);
for i = 1:size(X2, 1)
   Csedumi(i) = (X2(i, :) * SA * X2(i, :)' < 0.5) + 1;
end
acc = sum(Csedumi == y{2}) / size(y{2}, 1);
disp(acc);
