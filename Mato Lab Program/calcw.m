function [w, difflist] = calcw(Y, Z, w0, w)
% calculating mean vector(using Adam)
    iter = 1;
    alpha = 0.01;
    beta1 = 0.9;
    beta2= 0.999;
    epsilon = 10e-9;
    maxIte = 50;
    tol = 10e-8;
    m = 0;%randn([size(X, 2), 1]);
    V = 0;%randn([size(X, 2), 1]);
    diff = 10;
    difflist = diff;
%    fprintf('\n result of Adam\n');
    while (maxIte > iter) && (tol < diff)
        % fprintf('inner loop of w : %d\n', iter);
        w_prev = w;
        lambda = - calcGrad(Y, Z, w0, w);
        m = beta1 * m + (1 - beta1^iter) * lambda;
        V = beta2 * V + (1 - beta2^iter) * lambda.^2;
        mhat = m ./ (1 - beta1^iter);
        Vhat = V ./ (1 - beta2^iter);
        vfStep = alpha * mhat / (sqrt(Vhat) + epsilon);
        w = w - vfStep;
        diff = mean(abs(w - w_prev));
        difflist = [difflist , diff];
        iter = iter + 1;
    end
end


% 勾配計算用の関数
function gradW = calcGrad(Y, Z, w0, w)
    tmp = zeros(size(Y));
    for l = 1 :size(Z, 3)
        YS = Y - calcsigmoid(w0 + w * Z(:, :, l) * Z(:, :, l)');
        ZZ = Z(:, :, l) * Z(:, :, l)';
        tmp = tmp + YS .* ZZ;
    end
    tmp = tmp - diag(diag(tmp));
    sumtmp = sum(tmp, 'all');
    gradW = sumtmp / (2 * size(Z, 3));
end
%{
function gradW = calcGrad(Y, Z, w0, w)
    tmp = 0;
    for l = 1 :size(Z, 3)
        for i = 1 : size(Z, 1)
            for j = 1 : size(Z, 2)
                if j ~= i
                    tmp = tmp + (Y(i, j) - calcsigmoid(w0 + w ...
                        * Z(i, :, l) * Z(j, :, l)')) * Z(i, :, l) * Z(j, :, l)';
                end
            end
        end
    end
    gradW = tmp / (2 * size(Z, 3));
end
%}
function s = calcsigmoid(x)
    s = 1 ./ (1 + exp(-x));
end