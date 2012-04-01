classdef FaceDataset < handle
%FACEDATASET Face image dataset class
%   Face image dataset class that works with both the AT&T and the Yale
%   datasets. It can also export data to difference of vector format, or
%   to Euclidean distance format.

   properties
       dim;
       count;
       train_split;
       pair_train = 10;
       pair_test = 3;
       new_size;
       pca;
       pca_dim;
       Xtrain_pca;
       Xtest_pca;
       origtrain;
       origtest;
       same = cell(1, 2);
       diff = cell(1, 2);
   end

   methods
       % Constructor, takes the initial directory, the data split
       % percentage, the size to sample the images to, and dimensionality
       % of the PCA subspace to project to.
       function obj = FaceDataset(directory, split, newsize, pcadim)
           obj.train_split = split;
           obj.new_size = newsize;
           obj.pca_dim = pcadim;
           
           images = obj.readImages(directory);
           [Xtrain, Xtest, obj.origtrain, obj.origtest] = obj.generateSplit(images);
           
           pca = PCA(Xtrain, obj.pca_dim);
           obj.Xtrain_pca = pca.project(Xtrain);
           obj.Xtest_pca = pca.project(Xtest);
           
           [obj.same{1}, obj.diff{1}] = obj.generatePairs(length(obj.Xtrain_pca), obj.pair_train, obj.train_split);
           [obj.same{2}, obj.diff{2}] = obj.generatePairs(length(obj.Xtest_pca), obj.pair_test, 1 - obj.train_split);
       end
       
       % Generate the data from the images, either as differences of
       % feature vectors, or the Euclidean distance between them.
       function [X, y] = generateData(obj, dist)
           X = cell(1, 2);
           y = cell(1, 2);
           
           if strcmp(dist, 'euclidean')
               func = @(X1, X2) sum((X1 - X2).^2, 2);
           elseif strcmp(dist, 'difference')
               func = @(X1, X2) X1 - X2;
           end
           
           [X{1}, y{1}] = obj.generateDataParams(obj.Xtrain_pca, obj.same{1}, obj.diff{1}, func);
           [X{2}, y{2}] = obj.generateDataParams(obj.Xtest_pca, obj.same{2}, obj.diff{2}, func);
       end
       
       function images = readImages(obj, directory)
           images = [];
           obj.dim = prod(obj.new_size);
           
           fulldir = fullfile(fileparts(mfilename('fullpath')), directory);
           subdirs = dir(fulldir);
           obj.count = zeros(sum([subdirs.isdir]) - 2, 1);
           
           i = 1;
           for d = 3:length(subdirs)
               if subdirs(d).isdir
                   subdir = fullfile(fulldir, subdirs(d).name);
                   field = strrep(subdirs(d).name, '-', '_');

                   files = dir(subdir);
                   images.(field) = cell(1, length(files) - 2);

                   for f = 3:length(files)
                       images.(field){f - 2} = reshape(imresize(imread(fullfile(subdir, files(f).name)), obj.new_size), [], 1);
                       obj.count(i) = obj.count(i) + 1;
                   end
                   i = i + 1;
               end
           end
       end
       
       function [Xtrain, Xtest, origtrain, origtest] = generateSplit(obj, images)
           total = sum(obj.count);
           Xtrain = zeros(total * obj.train_split, obj.dim);
           Xtest = zeros(total * (1 - obj.train_split), obj.dim);
           origtrain = zeros(total * obj.train_split, 1);
           origtest = zeros(total * (1 - obj.train_split), 1);
           
           itrain = 1;
           itest = 1;
           names = fieldnames(images);
           count = 0;
           for i = 1:length(names)
               n = obj.count(i);
               p = randperm(n);
               t = n * obj.train_split;
               
               origtrain(itrain:itrain + t - 1, :) = p(1:t) + count;
               origtest(itest:itest + n - t - 1, :) = p(t + 1:n) + count;
               
               Xtrain(itrain:itrain + t - 1, :) = cell2mat(images.(names{i})(p(1:t)))';
               Xtest(itest:itest + n - t - 1, :) = cell2mat(images.(names{i})(p(t + 1:n)))';
               
               itrain = itrain + t;
               itest = itest + n - t;
               count = count + n;
           end
       end
       
       function [same, diff] = generatePairs(obj, n, pair_count, ratio)
           same = zeros(length(obj.count) * pair_count, 2);
           diff = zeros(length(obj.count) * pair_count, 2);
           
           index = 1;
           step = pair_count;
           val_index = 0;
           
           for i = 1:length(obj.count)
               s = combnk(1:floor(obj.count(i) * ratio), 2);
               p = randperm(size(s, 1));
               same(index:index + step - 1, :) = ceil(val_index + s(p(1:pair_count), :));
               
               d = [1:val_index, val_index + floor(obj.count(i) * ratio) + 1:n]';
               p = randperm(size(d, 1));
               diff(index:index + step - 1, :) = ceil([val_index + ceil(floor(obj.count(i) * ratio) .* rand(pair_count, 1)), d(p(1:pair_count))]);
               
               index = index + step;
               val_index = val_index + floor(obj.count(i) * ratio);
           end
       end
       
       function [X, y] = generateDataParams(obj, X_values, s, d, distFunc)
           X = [distFunc(X_values(s(:, 1), :), X_values(s(:, 2), :));
               distFunc(X_values(d(:, 1), :), X_values(d(:, 2), :))];
           y = [ones(length(s), 1);
               repmat(2, length(d), 1)];
       end
   end
end 
