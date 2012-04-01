function x = SDPAD(A, b, c)
% c(nn, 1), A(m, nn), b(m, 1)

% Main work variables
mu = 5; posEig = true; rho = 1.6;

% Stagnation variables
it_stag = 0; h1 = 20; h2 = 150; h3 = 300; ref = inf;

% Mu update variables
it_pinf = 0; it_dinf = 0; h4 = 20; eta1 = 1; eta2 = 1;
gamma = 0.5; mumin = 1e-4; mumax = 1e4;

% Termination variables
err = 1e-5; maxit = 2000;

disp('it        b*y        c*x        gap        pinf        dinf');

% Before we start, calculate AATI
tic;
[m, nn] = size(A);
n = sqrt(nn);
AATI = inv(A * A');
x = reshape(eye(n), [], 1);
Ax = A * x;
s = zeros(nn, 1);
tpre = toc;

tic;
for k = 1:maxit
    % S1: main work
    % Calculate y
    y = -AATI * (mu * (Ax - b) + A * (s - c));
    
    % Calculate v, then Eigen-decompose it
    ATy = A' * y;
    v = c - ATy - mu * x;
    [Q, E] = eig(reshape(v, n, n));
    neg = (diag(E) < 0);
    
    % Next, calculate s and x from the more sparse part of v
    % Not the best way to do it, because I cannot iteratively find the
    % next Eigen value in Matlab without writing my own C code to do it
    if posEig
        Q(:, neg) = 0;
        E(:, neg) = 0;
        s = reshape(Q * E * Q', [], 1);
        x = (1 - rho) * x + rho * (s - v) / mu;
        if sum(neg) < n / 2
            posEig = false;
        end
    else
        Q(:, ~neg) = 0;
        E(:, ~neg) = 0;
        vn = reshape(- Q * E * Q', [], 1);
        s = vn + v;
        x = (1 - rho) * x + rho * vn / mu;
        if sum(neg) >= n / 2
            posEig = true;
        end
    end
    
    % S2: stagnation detection and termination conditions
    Ax = A * x;
    pinf = norm(Ax - b) / (1 + norm(b));
    dinf = norm(ATy + s - c) / (1 + norm(reshape(c, n, n), 1));
    by = b' * y;
    cx = c' * x;
    gap = abs(by - cx) / (1 + abs(by) + abs(cx));
    delta = max([pinf, dinf, gap]);
    
    disp(sprintf('%d        %.5g        %.5g        %.5g        %.5g        %0.5g', k, by, cx, gap, pinf, dinf));
    
    % Terminate because error is too small
    if delta <= err
        break;
    end
    
    if delta <= ref
        ref = delta;
        it_stag = 0;
    else
        it_stag = it_stag + 1;
    end
    
    % Terminate if error is small enough and stagnation is occuring
    if (it_stag > h1 && delta <= 10 * err) || (it_stag > h2 && delta <= 100 * err) || (it_stag > h3 && delta <= 1000 * err)
        break;
    end
    
    % S3: updating mu
    if pinf / dinf <= eta1
        it_pinf = it_pinf + 1;
        it_dinf = 0;
        if it_pinf >= h4
            mu = max(gamma * mu, mumin);
            it_pinf = 0;
        end
    elseif pinf / dinf > eta2
        it_dinf = it_dinf + 1;
        it_pinf = 0;
        if it_dinf >= h4
            mu = min(mu / gamma, mumax);
            it_dinf = 0;
        end
    end
end
x = reshape(x, n, n);
tmain = toc;

disp(sprintf('Time: pre = %.5g, post = %.5g', tpre, tmain));
