%% NOLTA投稿用実験（その2）
% kは固定したもとで，Lを変化させてみる．
% 
% 実験の条件は一番最初のn, dがそれぞれ小さい場合で行う．（余裕があれば増やしても良いかも．）
% 1. パラメータの設定
% 1.1 真の値の設定

clear; clc; clf;
tic;
rng('shuffle');
% set true latent dimensions
kT = 3; % the number of true dimension
% misRate = 0.1; % missing value rate of Y
[X, Y, trueParams] = setTrueParams(150, 15, kT);
% 1.2 初期値の設定
% まずはLを変化させるた眼，配列に確保

% L = [1, 3, 5, 7, 9, 15];
L = 11;
% numLatVal = 8; % the number of latent variable dimension
% k = 3; % set true latent variable dimension
% できればここで初期値の推定用アルゴリズムを入れたい．→別プログラムで．
% ただし，実験の条件が変わってしまうので注意．
% 
% 今回は潜在次元数を変化させて，真の値をきちんと捉えられているかを確認する．
% 2. 潜在構造の推定
% 真の値と設定した値を別に入力としないといけないので，プログラムを修正する必要あり．

m = 1;
for k = 1 : 10
    for i = L
        params = setParamsDesc(trueParams, kT, i); % Lの個数を可変に
        [estParams(m), history] = calcdescmetric_ver4(X, Y, kT, params, trueParams);
        filename = "num_of_L" + num2str(i) + ".mat";
        main_save_func_parfor(filename, history);
        %    save(filename, "history");
        result(m) = calcRmseDesc(trueParams, estParams(m));
        historyTable = struct2table(history);
        filenameTable = "num_of_L" + num2str(i) + "_" + num2str(k) + ".csv";
        writetable(historyTable, filenameTable);
        m = m + 1;
    end
end
% 繰り返し毎にRMSEを計算するための方法に修正する必要あり！
%{
for i = 1 : size(L, 2)
    result(i) = calcRmseDesc(trueParams, estParams(i));
end
%% 
% 

% ファイルの読み込み
for i = L
    str = ['history', num2str(i), '=load("10-resultL/num_of_L', num2str(i), '.mat");'];
    eval(str);
end
% sigmaの描画
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.sigma])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;
%% 
% 

% Zの描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.Z])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;

% Z2の描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.Z2])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;
% Fの描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.F])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;

% F2の描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.F2])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;

% Yの描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.Y])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;

% Xの描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.X])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;

% w0の描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.w0])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;

% wの描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.w])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;

% Q関数の描画
clf;
hold on;
for i = L
    str = ['plot([history', num2str(i), '.history.Q])'];
    eval(str);
    grid on;
end
legend('1', '3', '5', '7', '9', '15');
hold off;
%}
toc;