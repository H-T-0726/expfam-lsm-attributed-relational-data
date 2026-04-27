
function w0 = calcw0(Y, Z, w0)
% calculating mean vector(using Adam)
    iter = 1;
    alpha = 0.01;
    beta1 = 0.9;
    beta2= 0.999;
    epsilon = 10e-9;
    maxIte = 50;
    tol = 10e-8;
    m = 0;%zeros([size(Z, 2), 1]);%randn([size(X, 2), 1]);
    V = 0;%zeros([size(Z, 2), 1]);%randn([size(X, 2), 1]);
    diff = 10;
    difflist = diff;
%    fprintf('\n result of Adam\n');
    while (maxIte > iter) && (tol < diff)
        % fprintf('inner loop of w0 : %d\n', iter);
        w0_prev = w0;
        lambda = - calcGrad(Y, Z, w0);
        m = beta1 * m + (1 - beta1^iter) * lambda;
        V = beta2 * V + (1 - beta2^iter) * lambda.^2;
        mhat = m ./ (1 - beta1^iter);
        %mhat = m;
        Vhat = V ./ (1 - beta2^iter);
        %Vhat = V;        
        vfStep = alpha * mhat / (sqrt(Vhat + epsilon));
        w0 = w0 - vfStep;
        diff = mean(abs(w0 - w0_prev));
        difflist = [difflist , diff];
        iter = iter + 1;
    end
end

% 勾配計算用の関数
function gradW0 = calcGrad(Y, Z, w0)
    %gradW0 = 0;
    diagS = zeros(size(Z, 1),size(Z, 1), size(Z, 3)); 
    T = pagemtimes(Z, pagetranspose(Z));
    S = calcsigmoid(w0 + T);
    for i = 1 : size(Z, 3)
        diagS(:, :, i) = diag(diag(S(:, :, i)));
    end
    S = S - diagS;
    gradW0 = sum((Y - S), 'all') / (2 * size(Z, 3));
    %gradW0 = - sum((S - Y), 'all') / (2 * size(Z, 3) * size(Z, 1) * nnz(Y));
end

function s = calcsigmoid(x)
    s = 1 ./ (1 + exp(-x));
end

%{
function w0 = calcw0(Y, Z, w0)
% calculating mean vector(using Adam)
iter = 1;
alpha = 0.01;
beta1 = 0.9;
beta2= 0.999;
epsilon = 10e-9;
maxIte = 50;
tol = 10e-8;
m = 0;
V = 0;
diff = 10;
difflist = diff;
% 計算の重複を削減する
gradient = calcGrad(Y, Z, w0);
while (maxIte > iter) && (tol < diff)
    w0_prev = w0;
    % 同じ値を再計算しないようにする
    lambda = - gradient;
    m = beta1 * m + (1 - beta1^iter) * lambda;
    V = beta2 * V + (1 - beta2^iter) * lambda.^2;
    mhat = m ./ (1 - beta1^iter);
    Vhat = V ./ (1 - beta2^iter);
    vfStep = alpha * mhat / (sqrt(Vhat + epsilon));
    w0 = w0 - vfStep;
    diff = mean(abs(w0 - w0_prev));
    difflist = [difflist , diff];
    iter = iter + 1;
end

end
function gradW0 = calcGrad(Y, Z, w0)
T = pagemtimes(Z, pagetranspose(Z));
S = calcsigmoid(w0 + T);
% Yの非ゼロ要素のインデックスを取得
[row, col, val] = find(Y);
% gradW0をベクトル演算で計算
gradW0 = - sum(val .* log(S(sub2ind(size(S), row, col))) + (1 - val) .* log(1 - S(sub2ind(size(S), row, col)))) / size(Z, 3);
end

function s = calcsigmoid(x)
s = 1 ./ (1 + exp(-x));
end
%}