function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

h = sigmoid(X*theta);
J = mean(-y.*log(h)-(1-y).*log(1-h)) ...
    + lambda/(2*m)*sum(theta(2:end).^2);
grad = ((X'* (h - y) + lambda*[0;theta(2:end)])) / m;

end
