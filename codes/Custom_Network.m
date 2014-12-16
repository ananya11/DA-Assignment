% training data
x=(load('C:\Users\student\Documents\MATLAB\input.csv'))';

% target data
t=(load('C:\Users\student\Documents\MATLAB\target_1.csv'))';

% create a custom network
net = network;

% define number of layers
net.numInputs = 1;
net.numLayers = 4;

% configure the network
net.biasConnect(1) = 1;
net.biasConnect(3) = 1;
net.biasConnect(4) = 1;

net.inputConnect(1,1) = 1;
net.inputConnect(2,1) = 1;
net.inputConnect(3,1) = 1;
%net.inputConnect(2,2) = 1;


net.layerConnect = [0 0 0 0; 0 0 0 0; 0 0 0 0; 1 1 1 1];
net.outputConnect = [0 0 0 1];

%net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};

net.layers{1}.size = 30;
net.layers{1}.transferFcn = 'tansig';
net.layers{1}.initFcn = 'initnw';

net.layers{2}.size = 10;
net.layers{2}.transferFcn = 'tansig';
net.layers{2}.initFcn = 'initnw';

net.layers{3}.size = 20;
net.layers{3}.transferFcn = 'logsig';
net.layers{3}.initFcn = 'initnw';

net.layers{4}.initFcn = 'initnw';

net.inputWeights{2,1}.delays = [0 1];
%net.inputWeights{2,2}.delays = 1;
net.layerWeights{4,4}.delays = 1;

net.initFcn = 'initlay';

%net.performFcn = 'mse';
net.trainFcn = 'trainrp';
%net.trainParam.lr = 0.05;

net.divideFcn = 'divideint';
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 20/100;


net = init(net);
[net,tr] = train(net,x, t);
view(net)

tic;
testX = x(:, tr.testInd);
testT = t(:,tr.testInd);
testY = net(testX);
toc;

figure
plotconfusion(testT, testY);


% overall percentages of correct and incorrect classification
[c,cm, ind, per] = confusion(testT, testY);
fprintf('Percentage Correct Classification : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

pb = cm(1,1)/(cm(1,1) + cm(2,1));
rb = cm(1,1)/(cm(1,1) + cm(1,2));
fb = 2*pb*rb / (pb+ rb);
fprintf('Precision of background : %f%%\n' ,pb);
fprintf('Recall of background : %f%%\n' , rb);
fprintf(' f1-score background : %f%%\n' , fb);

ps = cm(2,2)/(cm(2,2) + cm(1,2));
rs = cm(2,2)/(cm(2,2) + cm(2,1));
fs = 2*ps*rs / (ps+ rs);
fprintf('Precision of signal : %f%%\n' ,ps);
fprintf('Recall of signal : %f%%\n' , rs);
fprintf(' f1-score signal : %f%%\n' , fs);


% show false positives and false negatives rate
figure
plotroc(testT, testY);
