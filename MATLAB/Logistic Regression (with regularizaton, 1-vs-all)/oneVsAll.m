function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

[m, n] = size(X);
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

options = optimset('GradObj', 'on', 'MaxIter', 50);
for c = 1:num_labels
    % fmincg works similarly to fminunc, but is more efficient when dealing 
    % with large number of parameters.
    theta = ...
        fmincg (@(t)(costFunctionReg(t, X, (y == c), lambda)), ...
                 zeros(n + 1, 1), options);
     all_theta(c,:) = theta';
end

end
