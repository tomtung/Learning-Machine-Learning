%% Multivariate Linear Regression (with feature normalization)
%
% Motivating Task:
%   Predeicting housing prices.
% Data:
%   housing_prices.txt
%       - column 1: the size of the house (in square feet)
%       - column 2: is the number of bedrooms
%       - column 3: the price of the house
%

%% Initialization
clear all; close all;

%% Load & Normalize Data
fprintf('Loading data ...\n');
data = load('../Data/housing_prices.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% Linear Regression via Gradient Descent
alpha = 0.6;
num_iters = 100;
theta = zeros(3, 1);
fprintf('Running gradient descent ...\n');
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('Cost = %d\n\n', computeCost(X,y,theta));

% Estimate the price of a 1650 sq-ft, 3 br house
fprintf('Predicting using trained model:\n')
x = [1; (1650-mu(1))/sigma(1); (3-mu(2))/sigma(2)];
price = theta' * x;
fprintf('- For a 1650 sq-ft, 3 br house, predicted price = $%f\n' , price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Linear Regression via Normal Equations

% Load Data
data = csvread('data_multi.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
fprintf('Solving with normal equations...\n');
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('Cost = %d\n ', computeCost(X,y,theta));
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
fprintf('Predicting using trained model:\n')
x = [1; 1650; 3];
price = theta' * x;
fprintf('- For a 1650 sq-ft, 3 br house, predicted price = $%f\n' , price);
