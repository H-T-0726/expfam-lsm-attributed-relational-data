function Z = EffcalcEtaNewton(X, Y, F, Z, varZ, Sigma, w0, w) 
% EffcalcEtaNweton
    % eta = randn(size(Z, 1), size(Z, 2));
    maxIte = 10;
    tol = 10e-5;
    diff = 10;
    for i = 1 : size(Z, 1)
        iter = 1;
        alpha = 0.01;
        Ainit = calcInitA(Z, F, Sigma, w, w0, varZ);
        while (maxIte > iter) && (tol < diff)
            eta_prev = Z(i, :);
            lambda = calcGrad(X, Y, Z, Sigma, F, varZ, w0, w, i);
            %alpha = alpha * 0.5;
            Ai = calcAi(F, varZ, Sigma, w0, w, Z, i);
            invAi = inv(((Ai + Ai') ./ 2));
            Z(i, :) = Z(i, :) - alpha .* (invAi * lambda)';
            if norm(lambda) < tol
                disp("break");
                break;
            end
            diff = mean(abs(Z(i, :) - eta_prev));
            iter = iter + 1;
        end
        Z(i, :) = mvnrnd(Z(i, :), invAi);
    end
end

function A = calcInitA(F, varZ, Sigma, w0, w, Z, ind)
    % Sを算出
    % Sのi行目とzi zi^Tの積を計算
    FSF = (F' / Sigma) * F;
    sig = diag(calcsigmoid(w0 + w .* Z * Z'));
    
end

function partE = calcGrad(X, Y, Z, Sigma, F, varZ, w0, w, ind)
    FS = (F' / Sigma) * (X(ind, :)' - F * Z(ind, :)');
    S = calcsigmoid(w0 + w .* Z(ind, :) * Z');
    % S = tmpY(ind, :);
    YSWZ = (Y(ind, :) - S) * Z - (Y(ind, ind) - S(ind))* Z(ind, :);
    partE = (1 / varZ) * Z(ind, :)' - FS - YSWZ';
end

function s = calcsigmoid(x)
    s = 1 ./ (1 + exp(-x));
end

% calc covariance matrix of posterior of Z
function Ai = calcAi(F, varZ, Sigma, w0, w, Z, ind)
    FSF = (F' / Sigma) * F;
    sig = diag(calcsigmoid(w0 + w .* Z(ind, :) * Z'));
    sig = w^2 .* sig .* (1 - sig);
    tmp = Z' * sig * Z;
    tmp = tmp - sig(ind, ind) * (1 - sig(ind, ind)) .* w^2 .* Z(ind, :)' * Z(ind, :);
    Ai = (1 / varZ) * eye(size(Z, 2)) + FSF + tmp;
end

% 勾配計算用の関数
% ここの修正が効率化のためには必要！
%{
function partE = calcGrad(X, Y, Z, Sigma, F, varZ, w0, w, ind)
    FS = (F' / Sigma) * (X(ind, :)' - F * Z(ind, :)');
    tmp = zeros(size(Z(ind, :), 2), 1);    
    for j = 1 : size(Z, 1)
        if j ~= ind
            tmp = tmp + (Y(ind, j) - calcsigmoid(w0 + w * Z(ind, :) * Z(j, :)'))...
                * Z(j, :)';
        end
    end
    partE = (1 / varZ) .* Z(ind, :)' - FS - tmp;
end
%}
