%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ���ݶ��½������Ʋ�������ʧ������Ϊ������ѧϰ���ʷ����仯
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, L] = estimate_quadratic_form1_2()
%% ������
w0 = [1 5];

%% ��������������
m0 = 51;
n0 = 51;
[x0, y0] = meshgrid(linspace(-3, 3, m0), linspace(-3, 3, n0));

% �����������������Ϊ�˼���
xx = [x0(:) y0(:)];
z0 = f(xx, w0);
z0 = reshape(z0, m0, n0);  % ����֮���ٱ�ؾ���

%% ���������
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
w(1, :) = w0 + [0, 2];  % ��ʼ����

L = zeros(50, 1);
L(1) = loss(x, z, w(1, :));  % ��¼���Ŀ�꺯��

% ����
% rate = 0.28;  % ��ɢ
% rate = 0.2;   % ������
% rate = 0.1;   % �Ժ�
rate = 0.03;    % ����
% rate = 0.001;   % ��С
for i = 1:nIter
    eta = cal_grad(x, z, w(i, :));
    w(i+1, :) = w(i, :) - eta*rate;
    L(i+1) = loss(x, z, w(i+1, :));  % ��¼���Ŀ�꺯��
end

disp('��С���˷������')
disp(w_hat)
disp('�ݶ��½��������')
disp(w(nIter+1, :))

%% ��ͼ
figure(1)
hold off
% mesh(x0, y0, z0); hold on
surf(x0, y0, z0); hold on
shading interp
alpha(0.8)
plot3(x(:, 1), x(:, 2), z, '.r')

figure(2)
plot_contour(x, z, w0); hold on
% plot(w(end-20:end,1), w(end-20:end,2), '.-'); 
plot(w(:,1), w(:,2), '.-'); 

hold off


figure(3)
plot(0:nIter, L)



%% ����
function z = f(x, w)

z = x.^2 * w';   % ע��w��������


function eta = cal_grad(x, z, w)
res = z - f(x, w);
n = length(z);
eta1 = -2 * sum(res .* x(:, 1).^2) / n;
eta2 = -2 * sum(res .* x(:, 2).^2) / n;

eta = [eta1 eta2];
% eta = eta / norm(eta);


function z = loss(x, z, w)
res = z - f(x, w);
n = length(z);
z = res' * res / n;  % ƽ�����

function plot_contour(x, y, w0)
m1 = linspace(w0(1)-3, w0(1)+3, 101);
m2 = linspace(w0(2)-3, w0(2)+3, 101);
% m1 = linspace(min(w(:,1)), max(w(:,1)), 31);
% m2 = linspace(min(w(:,2)), max(w(:,2)), 31);
[wx, wy] = meshgrid(m1, m2);
w = [wx(:) wy(:)];
z = zeros(size(wx));
for i = 1:size(w, 1)
    z(i) = loss(x, y, w(i, :));
end

contour(wx, wy, z, 30)
