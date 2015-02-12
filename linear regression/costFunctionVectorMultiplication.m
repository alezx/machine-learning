function J = costFunctionVectorMultiplication(X, y, theta)

%	X is a matrix m x n
%	y is a vector of size m
%	theta is a vector of size n

%	computes the cost of using theta as the
%	parameter for linear regression to fit the data points in X and y
%	this version uses vector multiplication 


m = length(y); % number of training examples


predictions = X * theta;

% equivalent to sum((predictions - y).^2) but faster
J = 1/(2*m) * (predictions - y)' * (predictions - y);


end
