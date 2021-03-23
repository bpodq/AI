%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 改进学习速率
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, L] = estimate_quadratic_form2()
%% 二次型
w0 = [1 1];

%% 画出二次型曲面
m0 = 51;
n0 = 51;
[x0, y0] = meshgrid(linspace(-3, 3, m0), linspace(-3, 3, n0));
xx = [x0(:) y0(:)];
z0 = f(xx, w0);

z0 = reshape(z0, m0, n0);

%% 产生随机点
n = 1000;
x = randn(n, 2);
r = randn(n, 1) * 1.5;
z = f(x, w0) + r;

X = x.^2;
w_hat = regress(z, X);
w_hat = w_hat';

%%
nIter = 50;
w = zeros(nIter+1, 2);
w(1, :) = w0 + [0, 2];  % 初始参数

L = zeros(50, 1);
L(1) = cal_res(x, z, w(1, :));  % 记录最初目标函数

% 速率
nIter2 = 20;  % 内部循环
tol = 10;     % 每一步目标函数的改进要超过tol
for i = 1:nIter
    eta = cal_grad(x, z, w(i, :));
    rate = 0.1;  % 每次重置rate，还可以考虑根据上次的rate进行调整
    flag = 0;
    for j = 1:nIter2
        w(i+1, :) = w(i, :) - eta*rate;
        L(i+1) = cal_res(x, z, w(i+1, :));  % 记录最初目标函数
        if L(i+1) < L(i) - tol;
            flag = 1;
            break;
        else
            rate = rate / 2;
        end
    end
    if ~flag  % 无法改进了
        w = w(1:i, :);
        L = L(1:i);
        break;
    end
end

disp('最小二乘法结果：')
disp(w_hat)
disp('梯度下降法结果：')
disp(w(end, :))

%% 画图
figure(1)
hold off
mesh(x0, y0, z0); hold on
plot3(x(:, 1), x(:, 2), z, '.r')
cameratoolbar('Show')

figure(2)
plot_contour(x, z, w0); hold on
plot(w0(1), w0(2), 'b.', 'markersize', 20)
plot(w(:,1), w(:,2), 'r.-');
plot(w(end, 1), w(end, 2), 'r.', 'markersize', 15)
hold off

figure(3)
subplot(2,1,1); plot(0:length(L)-1, L)
subplot(2,1,2); plot(0:length(L)-1, log(L))

%% 函数
function z = f(x, w)
% n = size(x, 1);
% z = zeros(n, 1);
% for i = 1:n
%     z(i) = w(1)*x(i,1)^2 + w(2)*x(i,2)^2;
% end

z = x.^2 * w';   % 注意w是行向量


function eta = cal_grad(x, z, w)
res = z - f(x, w);
eta1 = -2 * sum(res .* x(:, 1).^2);
eta2 = -2 * sum(res .* x(:, 2).^2);

eta = [eta1 eta2];
% eta = eta / norm(eta);


function z = cal_res(x, z, w)
res = z - f(x, w);

z = res' * res;

function plot_contour(x, y, w0)
m1 = linspace(w0(1)-3, w0(1)+3, 101);
m2 = linspace(w0(2)-3, w0(2)+3, 101);
% m1 = linspace(min(w(:,1)), max(w(:,1)), 31);
% m2 = linspace(min(w(:,2)), max(w(:,2)), 31);
[wx, wy] = meshgrid(m1, m2);
w = [wx(:) wy(:)];
z = zeros(size(wx));
for i = 1:size(w, 1)
    z(i) = cal_res(x, y, w(i, :));
end

contour(wx, wy, z, 30)
