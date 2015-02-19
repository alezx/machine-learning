function [J, grad] = costFunction(theta, X, y)

%   Compute cost and gradient for logistic regression
%   computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   this two values can then be used in fminunc to calculate
%   the theta values that minimize the cost function.

m = length(y); % number of training examples


predictions = sigmoid(X*theta);
first_term = -y .* log(predictions);
second_term = (1 - y) .* log(1-predictions);
% J = 1/m * sum( first_term - second_term ); % equivalent to following line
J = 1/m * (first_term - second_term)' * ones(m,1);
grad = 1/m * (X' * (predictions-y));

end
