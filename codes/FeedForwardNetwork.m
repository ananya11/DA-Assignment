% This classification is done based on Feed Forward Network. 
% This script here creates a model of two-layer (one hidden layer). 
% This is a single hidden layer of 70 neurons. Input contains 29 features and
% target contains only 1

% create a input file containing rows 2 - 30 from training.csv. 
% Ignore columns 'EventId', 'PRI_jet_num', 'Label' and 'Weight' 
x=(load('C:\Users\student\Documents\MATLAB\input.csv'))';

% create a target file containing row #33 from training.csv , Background: 0 , Signal : 1
t=(load('C:\Users\student\Documents\MATLAB\target_1.csv'))';


% neural network starts with random weights. Add a seed to avoid this randomness
setdemorandstream(111)

% create a single hidden network with 40 neurons
net = feedforwardnet(40);
view(net)

% split data into train and test
net.divideParam.trainRatio = 80/100;
net.divideParam.testRatio = 20/100;

% scaled conjugate gradient backpropagation
net.trainFcn = 'trainscg';
[net,tr] = train(net,x,t);


tic;
% test samples
testX = x(:, tr.testInd);
testT = t(:,tr.testInd);

testY = net(testX);
toc;

figure
plotconfusion(testT, testY)

% overall percentages of correct and incorrect classification
[c,cm, ind, per] = confusion(testT, testY);
fprintf('Percentage Correct Classification : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

% background
pb = cm(1,1)/(cm(1,1) + cm(2,1));
rb = cm(1,1)/(cm(1,1) + cm(1,2));
fb = 2*pb*rb / (pb+ rb);
fprintf('Precision of background : %f%%\n' ,pb);
fprintf('Recall of background : %f%%\n' , rb);
fprintf(' f1-score background : %f%%\n' , fb);

% signal
ps = cm(2,2)/(cm(2,2) + cm(1,2));
rs = cm(2,2)/(cm(2,2) + cm(2,1));
fs = 2*ps*rs / (ps+ rs);
fprintf('Precision of signal : %f%%\n' ,ps);
fprintf('Recall of signal : %f%%\n' , rs);
fprintf(' f1-score signal : %f%%\n' , fs);

% show false positives and false negatives rate 
figure
plotroc(testT, testY)



