classdef SPM < handle
%SPM Generates the matrices for the SPM problem
%   Generates the matrices A, b, and c for the Semidefinite Probabilistic
%   Model using the feature vectors X and the labels y in standard form SDP
%   that can be solved using any standard solver.

   properties
       A;
       b;
       c;
   end

   methods
       function obj = SPM(X, y, beta)
           % Normalize X to unit length
           X = bsxfun(@rdivide, X, sqrt(sum(X.^2, 2)));
           X(isnan(X)) = 0;
           
           % Number of varialbes is (m)
           [n, d] = size(X);
           dd = d * d;
           m = 1 + 2 * n + d;
           
           % The location of the projection submatrix inside A
           [mi, mj] = meshgrid(1 + 2 * n + 1:m, 1 + 2 * n + 1:m);
           coeff = reshape(sub2ind([m, m], mi, mj), [], 1);
           
           % Number of constraints is (an)
           atri = 1;
           astep = 3 + dd;
           an = n * astep;
           ai = zeros(an, 1);
           aj = zeros(an, 1);
           as = zeros(an, 1);
           
           obj.b = zeros(n, 1);
           
           % For all training instances
           for i = 1:n
               XXTi = X(i, :)' * X(i, :);
               sgn = 3 - 2 * y(i);
               
               % Add a constraint involving Eta, Zai_i, Lambda_i, and the
               % projection submatrix
               ai(atri:atri + astep - 1) = repmat(i, astep, 1);
               aj(atri:atri + astep - 1) = [1; sub2ind([m, m], [1 + i; 1 + n + i], [1 + i; 1 + n + i]); coeff];
               as(atri:atri + astep - 1) = [-1; 1; -1; sgn * 2 * reshape(XXTi, [], 1)];
               atri = atri + astep;
               
               obj.b(i) = sgn * trace(XXTi);
           end
           
           obj.A = sparse(ai, aj, as, n, m * m, an);
           obj.c = sparse(sub2ind([m, m], (1:n + 1)', (1:n + 1)'), ones(n + 1, 1), [-1; repmat(beta, n, 1)], m * m, 1, n + 1);
       end
   end
end 
