%% Univariate Linear Regression (& Visualization)
%
% Motivating Task:
%   Suppose you are the CEO of a restaurant franchise and are considering
%   different cities for opening a new outlet. The chain already has trucks 
%   in various cities and you have data for profits and populations from
%   the cities.
%   You would like to use this data to help you select which city to expand
%   to next.
% Data:
%   data_uni.txt
%   - column 1: the population size of a city in (10,000s)
%   - column 2: the profit of a truck in that city (in $10,000s)
%

%% Initialization
clear all; close all; clc

%% Data Plotting
fprintf('Plotting Data ...\n')
data = load('data_uni.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('\n');

%% Linear Regression via Gradient Descent

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
% compute and display initial cost
fprintf('Initial cost = %f\n', computeCost(X, y, theta))

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('Running Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
fprintf('Predicting using trained model:\n')
fprintf('- For population = 35,000, predicted profit = %f\n',...
    [1, 3.5] * theta * 10000);
fprintf('- For population = 70,000, predicted profit = %f\n',...
    [1, 7] * theta * 10000);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('\n');

%% Visualization of Cost Function J(theta_0, theta_1)
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
