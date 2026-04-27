% 覚書：とりあえず，misRateは使用しない．
% 使うときは引数の最後にmisRateをつける．

function [X, Y, params] = setTrueParams(n, d, k)
% to generate true value of each parameters
% input:
% n : the number of data (X)
% d : dimensionality of X
% m : dimensionality of z
% output : 
% params : true parameters (structure)
   rng(1980);
    
   % set true parameters
   uniq = 0.1;
   params.varF = 5;
   params.varZ = 1;
   params.varY = 0.1;
   % params.sigma = 0.3 * diag(gamrnd(2, 2, [1 d])); % 
   params.sigma = diag(uniq * ones(1, d)); % と元のプログラムでは記述．
   % params.muX = normrnd(0, 4, [1, d]);
   params.Z = normalize(normrnd(0, params.varZ^0.5, [n, k]));
   params.F = normrnd(0, params.varF^0.5, [d, k]);
   
   % これ，因子分析の独自成分とかを設定するための処理．いったんコメントアウト
   for i = 1 : size(params.F, 1)
        params.F(i, :) = (params.F(i, :) ./ norm(params.F(i, :))) * sqrt(1 - params.sigma(i, i));
   end

   params.w0 = randn(); % -2;%
   params.w = 3 * randn(); % 5; % 
   % params.mask = rand(n) >= misRate;
   % params.Nmask = ~params.mask;
    
   % generate X and Y
   X = zeros(n, d);
   % X = mvnrnd(params.Z * params.F' + params.muX, params.sigma);
   X = mvnrnd(params.Z * params.F', params.sigma);
   % params.X = X;
   params.X = normalize(X); % 必要に応じてXを正規化する．
   X = params.X;
   Y = zeros(n);
   % for i = 1 : n
   %     for j = i + 1 : n
   %         Y(i, j) = binornd(1, 1 ./ (1 + exp(- params.w0 - params.w * (params.Z(i, :) * params.Z(j, :)'))));
   %     end
   % end
   % params.Y = Y + Y';
   Y = triu(binornd(1, 1 ./ (1 + exp(- params.w0 - params.w .* (params.Z * params.Z')))), 1);
   
   Y = Y + Y';
   params.Y = Y;
   % params.Ydense = Y;
   % Y = Y .* params.mask;   
end