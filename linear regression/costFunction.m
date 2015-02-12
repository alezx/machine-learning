function J = costFunction(X, y, theta)

%   X is a matrix m x n
%   y is a vector of size m
%   theta is a vector of size n

%   computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y


m = length(y); % number of training examples


predictions = X * theta; 			% predictions of hypothesis on all m
sqrErrors = (predictions-y).^2;	 	% squared errors

J = 1/(2*m) * sum(sqrErrors);

end
