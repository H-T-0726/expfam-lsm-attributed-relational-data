% NOLTAの実験に合わせてプログラムを修正．
% ver4 : outputにhistoryを追加
% e_params : 最終的な出力結果のみ
% history : 繰り返しごとの結果出力
% Xは正規化されているという前提のもと，muX(=0)を削除

function [e_params, history] = calcdescmetric_ver4(X, Y, i, params, tparams)
    rng('shuffle');
    [n, d] = size(X);
    m = i; % あまり頭の良い方法ではないので後で検討を行うこと．
    % e_params.muX = mean(X);
    w0 = randn(); %params.w0true; w0est; %params.w0true; %
    w = randn(); %params.wtrue; %
    % muX = zeros(size(X, 2));
    % parameter initialization
    Z = params.Z; % normrnd(0, 1, [n, m]); % params.Z_true; %
    F = params.F; % normrnd(0, varF^0.5, [d, m]); %params.F_true; %zeros(params.d, params.m);  % %
    sigma = params.sigma; % diag(ones(d, 1)); %params.sigma_true; %
    Z_new = zeros(size(Z, 1), size(Z, 2), params.L);
    for a = 1 : params.numIter % a is the number of trial
        % fprintf('%d\n', a);
        % varZ = rand(); %params.varZtrue; %
        % varF = rand();
        % generate postrior probability
        for l = 1 : params.L
            Z = calcEtaNewton(X, Y, F, Z, 1, sigma, w0, w);
            Z_new(:, :, l) = Z;
        end
        Z_new2 = scaleZ(Z_new);
        % Z_new2 = Z_new;
        Z = Z_new2(:, :, params.L);
        F = calcF(X, Z_new2);
        w0 = calcw0(Y, Z_new2, w0);
        [w, difflist] = calcw(Y, Z_new2, w0, w);
        % varZ = calcVarZ(Z_new2, params.L);
        sigma = calcSigma(X, Z_new2, F, params.L);
        Q = 0;
        % calc Q function
        for l = 1 : params.L
            Q = Q + calcp_X(X, F, Z_new2(:, :, l), sigma) ...
                + calcp_Y(Y, Z_new2(:, :, l), w0, w) ...
                + calcp_Z(Z_new2(:, :, l),  1); % varZ); %
        end
        fprintf('%d times finished\t%d\n', a, Q / params.L);
        history(a) = calcRmseDesc(tparams, Z, F, w0, w, sigma, Q / params.L);
    end
    e_params.Q = Q / params.L;
    e_params.Z = Z_new2(:, :, params.L);
    tmpY2 = calcsigmoid(w0 + w .* Z_new2(:, :, params.L) * Z_new2(:, :, params.L)');
    % s_iiの結果とs_ijの値に違いがあるかどうか？(tmp2)
    tmpY = round(tmpY2);
    e_params.Y = tmpY - diag(diag(tmpY));
    e_params.F = F;
    % e_params.varZ(a) = varZ; % Zは分散が1になるように正規化してある．
    e_params.X = Z * F';
    e_params.sigma = sigma;
    e_params.w0 = w0;
    e_params.w = w;
end
%toc;



%% functions %%
function Z_new2 = scaleZ(Z)
% z_iのl2ノルムと次元数で標準化してやる
% 以下，修正しておくこと．
a = 0;
for l = 1 : size(Z, 3)
    for i = 1 : size(Z, 1)
        for k = 1 : size(Z, 2)
            a = a + Z(i, k, l)^2;
        end
    end
end
a = a / (size(Z, 3) * size(Z, 1));
%Z_new2 = (Z - mean(Z, 1)).* sqrt(size(Z, 2) / a);
Z_new2 = Z.* sqrt(size(Z, 2) / a);
end


function F = calcF(X, Z)
tmp = pagemtimes(pagetranspose(Z), Z);
ZZT2 = sum(tmp, 3);
XZT2 = sum(pagemtimes(X', Z), 3);
F = XZT2 / ZZT2;
end

function sigma = calcSigma(X, Z, F, L)
sigma = zeros(size(X, 2));
for i = 1 : L
    sigma = sigma + (X - Z(:, :, i) * F')' * (X - Z(:, :, i) * F');
end
sigma = sigma ./ (L * size(X, 1));
sigma = diag(diag(sigma));
end

function varZ = calcVarZ(Z, L)
varZ = norm(Z(:))^2 ./ (L * size(Z, 1) * size(Z, 2));
end

function p_X = calcp_X(X, F, Z, sigma)
% p_X = - (size(X, 2)/ 2) * log(2 * pi) - (1 / 2) * log(trace(sigma))...
%     -(1 / 2) * trace(((X - Z * F')'* ...
%     (X - Z * F')) /diag(diag(sigma)));
    [n, d] = size(X);
    p_X = - (n * d/ 2) * log(2 * pi) - (n / 2) * sum(log(diag(sigma)))...
        -(1 / 2) * trace(((X - Z * F')'* ...
        (X - Z * F')) / diag(diag(sigma)));
end

function p_Y = calcp_Y(Y, Z, w0, w)
S = 1 ./ (1 + exp(-w0 - w .* Z * Z'));
tmp = (Y .* log(S + 10e-7)) + ((1 - Y) .* log((1 - S) + 10e-7));
tmp = tmp - diag(diag(tmp));
p_Y = sum(tmp, 'all') / 2;
end

function p_Z = calcp_Z(Z, sigma_z)
% p_Z = - (size(Z, 2)/ 2) * log(2 * pi) - (1 / 2) * log(sigma_z * size(Z, 2))...
%     - (1 / (2 * sigma_z)) * norm(Z, 'fro')^2;
[n, d] = size(Z);
p_Z = - (n * d/ 2) * log(2 * pi) ...
    - (n / 2) * d * log(sigma_z)...
    - (1 / (2 * sigma_z)) * sum(Z(:).^2);
end

function s = calcsigmoid(x)
s = 1 ./ (1 + exp(-x));
end

function history = calcRmseDesc(tParams, Z, F, w0, w, sigma, Q)
% rotate parameter F, Z
[F2, Z2] = rotVal(F, Z);

% calculate RMSE of variances
% RMSE.varY = rmse(tParams.varY, estParams.varY); % sqrt((tParams.varY - estParams.varY)^2);
% RMSE.varZ = rmse(tParams.varZ, estParams.varZ); % sqrt((tParams.varZ - estParams.varZ)^2);
history.sigma = sqrt(mean((diag(tParams.sigma) - diag(sigma)).^2));
% rmse(diag(tParams.sigma), diag(estParams.sigma)); % norm(tParams.sigma - estParams.sigma)^2 ./ size(tParams.sigma, 1);
% RMSE.muX = rmse(tParams.muX, estParams.muX, "all"); % norm(tParams.muX - estParams.muX)^2 ./ size(tParams.muX, 2);
history.Z = rmse(diag(sqrt(tParams.Z * tParams.Z')), diag(sqrt(Z * Z'))); 
history.Z2 = rmse(diag(sqrt(tParams.Z * tParams.Z')), diag(sqrt(Z2 * Z2')));
history.F = rmse(sqrt(tParams.F * tParams.F'), sqrt(F * F'), "all"); 
history.F2 = rmse(sqrt(tParams.F * tParams.F'), sqrt(F2 * F2'), "all"); 
tmpY = 1 ./ (1 + exp(-w0 - w .* Z * Z'));
Yhat = tmpY - diag(diag(tmpY));
%RMSE.Y = norm(triu(tParams.Y - Yhat), 'fro') / sqrt(nnz(triu(Yhat, 1))); %rmse(tParams.Y, Yhat, "all"); %
history.Y = sum(abs(tParams.Y - Yhat), 'all') / (size(tParams.Y, 1)^2 - size(tParams.Y, 1));
history.X = rmse(tParams.X, Z * F', "all");% + estParams.muX, "all");
history.w0 = rmse(tParams.w0, w0);
history.w = rmse(tParams.w, w);
history.Q = Q;
end

function [normF, normZ] = rotVal(F, Z)
    [n, ~] = size(Z);
    [U, S, V] = svd(Z, "econ");
    % U : \tilde{Z}, S * V' : Lambda
    normZ = sqrt(n) .* U; % 修正した
    lambda = (1 / sqrt(n)) .* S * V';
    normF = F * lambda';
end

