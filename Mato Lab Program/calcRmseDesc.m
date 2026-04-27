function RMSE = calcRmseDesc(tParams, estParams)
% calculating RMSE and prediction error
% input : 
%   tParams     : true parameters
%   estParams   : estimated parameters
% output :
%   RMSE        : calculated value of each parameters

% rotate parameter F, Z
[F, Z] = rotVal(estParams.F, estParams.Z);

% calculate RMSE of variances
% RMSE.varY = rmse(tParams.varY, estParams.varY); % sqrt((tParams.varY - estParams.varY)^2);
% RMSE.varZ = rmse(tParams.varZ, estParams.varZ); % sqrt((tParams.varZ - estParams.varZ)^2);
RMSE.sigma = sqrt(mean((diag(tParams.sigma) - diag(estParams.sigma)).^2));
% rmse(diag(tParams.sigma), diag(estParams.sigma)); % norm(tParams.sigma - estParams.sigma)^2 ./ size(tParams.sigma, 1);
% RMSE.muX = rmse(tParams.muX, estParams.muX, "all"); % norm(tParams.muX - estParams.muX)^2 ./ size(tParams.muX, 2);
RMSE.Z = rmse(diag(sqrt(tParams.Z * tParams.Z')), diag(sqrt(estParams.Z * estParams.Z'))); 
RMSE.Z2 = rmse(diag(sqrt(tParams.Z * tParams.Z')), diag(sqrt(Z * Z')));
RMSE.F = rmse(sqrt(tParams.F * tParams.F'), sqrt(estParams.F * estParams.F'), "all"); 
RMSE.F2 = rmse(sqrt(tParams.F * tParams.F'), sqrt(F * F'), "all"); 
tmpY = 1 ./ (1 + exp(-estParams.w0 - estParams.w .* estParams.Z * estParams.Z'));
Yhat = tmpY - diag(diag(tmpY));
%RMSE.Y = norm(triu(tParams.Y - Yhat), 'fro') / sqrt(nnz(triu(Yhat, 1))); %rmse(tParams.Y, Yhat, "all"); %
RMSE.Y = sum(abs(tParams.Y - Yhat), 'all') / (size(tParams.Y, 1)^2 - size(tParams.Y, 1));
RMSE.X = rmse(tParams.X, estParams.X, "all");% + estParams.muX, "all");
RMSE.w0 = rmse(tParams.w0, estParams.w0);
RMSE.w = rmse(tParams.w, estParams.w);
end

function [normF, normZ] = rotVal(F, Z)
    [n, ~] = size(Z);
    [U, S, V] = svd(Z, "econ");
    % U : \tilde{Z}, S * V' : Lambda
    normZ = sqrt(n) .* U; % 修正した
    lambda = (1 / sqrt(n)) .* S * V';
    normF = F * lambda';
end
