function [w, L] = estimate_quadratic_form()
%% 二次型
w0 = [1 5];

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



%%
nIter = 50;
w = zeros(nIter+1, 2);
w(1, :) = w0 + [0, 2];  % 初始参数

L = zeros(50, 1);

% 速率
% rate = 0.0003;  % 发散
% rate = 0.0002;  % 来回跑
% rate = 0.00015; % 稍好
rate = 0.0001;    % 合适
% rate = 0.000001;  % 过小
for i = 1:nIter
    L(i) = cal_res(x, z, w(i, :));  % 记录最初目标函数
    eta = cal_grad(x, z, w(i, :));
    w(i+1, :) = w(i, :) - eta*rate;
end

%% 画图
figure(1)
hold off
mesh(x0, y0, z0); hold on
plot3(x(:, 1), x(:, 2), z, '.r')

figure(2)
plot_contour(x, z, w0, w); hold on
% plot(w(end-20:end,1), w(end-20:end,2), '.-'); 
plot(w(:,1), w(:,2), '.-'); 

hold off

function z = f(x, w)
n = size(x, 1);
z = zeros(n, 1);
for i = 1:n
    z(i) = w(1)*x(i,1)^2 + w(2)*x(i,2)^2;
end


function eta = cal_grad(x, z, w)
res = z - f(x, w);
eta1 = -2 * sum(res .* x(:, 1).^2);
eta2 = -2 * sum(res .* x(:, 2).^2);

eta = [eta1 eta2];
% eta = eta / norm(eta);


function z = cal_res(x, z, w)
res = z - f(x, w);

z = res' * res;

function plot_contour(x, y, w0, w)
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
