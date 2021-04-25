% https://blog.csdn.net/zouxy09/article/details/45276053

% Face recognition using eigenfaces
 
close all, clear, clc;
 
%% 20 random splits
num_trainImg = 5;
showEigenfaces = true;
 
%% load data
disp('loading data...');
dataDir = './CroppedYale';
datafile = 'Yale.mat';
if ~exist(datafile, 'file')
	readYaleDataset(dataDir, datafile);
end
load(datafile);
 
%% Five images per class are randomly chosen as the training
%% dataset and remaining images are used as the test dataset
disp('get training and testing data...');
num_class = size(unique(labels), 2);
trainIdx = [];
testIdx = [];
for i=1:num_class
	label = find(labels == i);  % 找到下标
	indice = randperm(numel(label));
	trainIdx = [trainIdx label(indice(1:num_trainImg))];  % 取出下标
	testIdx = [testIdx label(indice(num_trainImg+1:end))];
end

%% get train and test data
% 从下标得到训练集，32x32=1024, 38个人，每人5张
% 数据为1024*190, 每个人的数据是一列
train_x = double(data(:, trainIdx));
train_y = labels(trainIdx);
test_x = double(data(:, testIdx));
test_y = labels(testIdx);

% 查看第几个人的图片
% imagesc(reshape(train_x(:, 2), 32, 32)); colormap('gray')

%% computing eigenfaces using PCA
disp('computing eigenfaces...');
tic;
[num_dim, num_imgs] = size(train_x);   %% A: #dim x #images

% computing the average face, 计算平均脸，求每一行的均值
avg_face = mean(train_x, 2);

% computing the difference images, 每一行与均值做差
X = bsxfun(@minus, train_x, avg_face); %

X = X / 255;  % 这里处理一下后面特征值就不会太大 by Yang

%% PCA
if num_dim <= num_imgs
	C = X * X';
	[V, D] = eig(C);
    % 这里缺少 V = X * U 的语句
else
	C = X' * X;
	[U, D] = eig(C);  % U'*C*U = D
	V = X * U;  % ？？？
end
eigenfaces = V;
eigenfaces = eigenfaces ./ (ones(size(eigenfaces,1),1) * sqrt(sum(eigenfaces.*eigenfaces)));
toc;
 
%% visualize the average face
P = sqrt(numel(avg_face));
Q = numel(avg_face) / P;
imagesc(reshape(avg_face, P, Q)); title('Mean face');
colormap('gray');
 
%% visualize some eigenfaces
figure;
num_eigenfaces_show = 9;
for i = 1:num_eigenfaces_show
	subplot(3, 3, i)
	imagesc(reshape(eigenfaces(:, end-i+1), P, Q));  % 特征值大的在后面，特征值好大
	title(['Eigenfaces ' num2str(i)]);
end
colormap('gray');
 
%% transform all training images to eigen space (each column for each image)
disp('transform data to eigen space...');
X = bsxfun(@minus, train_x, avg_face);
T = eigenfaces' * X;

%% transform the test image to eigen space
X_t = bsxfun(@minus, test_x, avg_face);
T_t = eigenfaces' * X_t;

%% find the best match using Euclidean distance
disp('find the best match...');
AB = -2 * T_t' * T;       % N x M
BB = sum(T .* T);         % 1 x M
distance = bsxfun(@plus, AB, BB);        % N x M
[score, index] = min(distance, [], 2);   % N x 1

%% compute accuracy
matchCount = 0;
for i=1:numel(index)
	predict = train_y(index(i));
	if predict == test_y(i)
		matchCount = matchCount + 1;
	end
end
fprintf('**************************************\n');
fprintf('accuracy: %0.3f%% \n', 100 * matchCount / numel(index));
fprintf('**************************************\n');

