function [minInd, BIC_X, loglikely] = DecideNumFactor(X)
    df = zeros(1, size(X, 2));
    [n, d] = size(X);
    i_vec = 1:d;
    df = ((d - i_vec).^2 - (d + i_vec)) / 2;
    df(df < 0) = 99999;
    [~, dfInd] = min(df); % dfIndが最大因子数になる
    loglikely = zeros(1, dfInd);
    BIC_X = zeros(1, dfInd);
    for i = 1 : length(loglikely)
        [~,~,~,stats] = factoran(X, i, 'Rotate', 'none');
        loglikely(1, i) = stats.loglike;
        t = (i + 1) * d - .5 * i * (i - 1);
        BIC_X(1, i) = -2 * n * loglikely(1, i) + t * log(n);
    end
    [~, minInd] = min(BIC_X); % 最小値のインデックス取得
end