function params = setParamsDesc(tparams, k, L)
    rng('shuffle');
    params.L = L;
    params.numIter = 10;
    % params.maxite = maxite; % 直列に計算しているのでこれはnumIterの統合してしまう
    % params.numj = numj; % これも
    params.varZ = tparams.varZ * 2;
    params.varF = tparams.varF * 2;
    params.Z = randn(size(tparams.X, 1), k);
    params.F = randn(size(tparams.F, 1), k); % zeros(size(tparams.Fx, 1), params.mx);% 
    % params.muX = randn(size(tparams.muX)) .* 0.1; %zeros(size(tparams.muX)); % 
    params.sigma = diag(ones(size(tparams.sigma, 1), 1)); % zeros(size(tparams.sigma, 1)); % 
    % params.mask = tparams.mask; % 虫食いにするとき用に取っておく
end