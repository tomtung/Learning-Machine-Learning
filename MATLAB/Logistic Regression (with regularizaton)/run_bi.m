%% Logistic Regression (binary, linearly separable)
%
% Motivating Task:
%   Suppose that you want to determine each applicant's chance of admission
%   based on their results on two exams. For each training example, you 
%   have the applicant's scores on two exams and the admissions decision.
% Data:
%   data_ad.txt
%       - column 1 & 2: scores on two exams
%       - column 3: admissions decision
%

%% Initialization
clear all; close all;

%% Load Data
data = csvread('data_bi.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% Plotting
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);
plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Optimizing using fminunc

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Predict and Accuracies

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 
fprintf('Predicting using trained model:\n')
prob = sigmoid([1 45 85] * theta);
fprintf(['- For a student with scores 45 and 85, predicted admission ' ...
         'probability = %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f%%\n', mean(double(p == y)) * 100);
