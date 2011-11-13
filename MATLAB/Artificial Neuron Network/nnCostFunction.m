function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1); % number of training examples

% Feedforward
A1 = [ones(m, 1) X];
Z2 = A1 * Theta1'; A2 = [ones(m, 1) sigmoid(Z2)];
Z3 = A2 * Theta2'; A3 = sigmoid(Z3);
% Label matrix
Y = bsxfun(@eq, y, 1:num_labels);

% Cost function J(theta)
sumn = @(M) sum(M(:));
J = sumn(-Y.*log(A3)-(1-Y).*log(1-A3)) / m + ...
    lambda / (2*m) * (sumn(Theta1(:,2:end).^2) + sumn(Theta2(:,2:end).^2));

% Backpropagation
Delta3 = A3 - Y;
Theta2_grad = Delta3' * A2 / m + ...
    lambda / m * [zeros(num_labels,1) Theta2(:,2:end)];

Delta2 = Delta3*Theta2 .* [ones(m, 1) sigmoidGradient(Z2)];
Theta1_grad = Delta2(:, 2:end)' * A1 / m + ...
    lambda / m * [zeros(hidden_layer_size,1) Theta1(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
