classdef PCA
%PCA Summary of this class goes here
%   Detailed explanation goes here

   properties
       X_proj;
       X_mean;
   end

   methods
       function obj = PCA(X, dim)
           obj.X_mean = mean(X, 1);
           Xm = bsxfun(@minus, X, obj.X_mean);
           
           S = Xm * Xm';
           [V, D] = eig(S);
           obj.X_proj = Xm' * V;
           
           s = size(obj.X_proj, 2);
           obj.X_proj = obj.X_proj(:, s - dim + 1:s);
           obj.X_proj = obj.X_proj ./ repmat(sqrt(sum(obj.X_proj.^2, 1)), size(obj.X_proj, 1), 1);
       end
       
       function [X_pca] = project(obj, X_orig)
           X_pca = bsxfun(@minus, X_orig, obj.X_mean) * obj.X_proj;
       end
       
       function [X_orig] = reconstruct(obj, X_pca)
           X_orig = bsxfun(@plus, X_pca * obj.X_proj', obj.X_mean);
       end
   end
end 
