function [J, grad] = costFunctionReg(theta, X, y, lambda)
%   Compute cost and gradient for logistic regression 
%   with regularization


m = length(y); % number of training examples

predictions = sigmoid(X*theta);
first_term = -y .* log(predictions);
second_term = (1 - y) .* log(1-predictions);
reg_theta = [0; theta(2:size(theta))]; % setting theta(1) = 0
third_term = lambda/(2 * m) * reg_theta' * reg_theta;
J = 1/m * sum( first_term - second_term ) + third_term; 
grad = (1/m * (X' * (predictions-y))) + (lambda/m) * reg_theta;


end
