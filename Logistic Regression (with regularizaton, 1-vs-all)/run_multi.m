%% Logistic Regression (multi-class, linear, regularized)
%
% Motivating Task:
%   Recognize handwritten digits (from 0 to 9).
% Data:
%   hand_written_digits.mat
%       - X: each row is an "unrolled" 20x20 grid of pixels
%       - y: labels
%

%% Initialization
clear all; close all;

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% Loading and Visualizing Data
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('../Data/hand_written_digits.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% One-vs-all Logistic Regression
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

%% Accuracies
pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;