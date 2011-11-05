%% Logistic Regression (binary, not linear, regularized)
%
% Motivating Task:
%   Suppose that you want to predict whether microchips from a fabrication
%   plant passes quality assurance (QA), during which each microchip goes
%   through various tests to ensure it is functioning correctly. You have
%   the test results for some microchips on two different tests, and
%   whether they were accepted or rejected.
% Data:
%   data_bi_reg.txt
%       - column 1 & 2: scores on two tests
%       - column 3: accepted or rejected
%

%% Initialization
clear all; close all;

%% Load Data
data = csvread('data_bi_reg.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% Regularized Logistic Regression
%  Data points in the given dataset with are not linearly separable. In
%  order to use logistic regression to classify the data points, we
%  introduce more features to use -- in particular, we add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Accuracies
% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
