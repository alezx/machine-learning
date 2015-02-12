function [theta] = normalEqn(X, y)

%   X is a matrix m x n
%   y is a vector of size m
%	computes theta using normal equation.

theta = pinv(X'*X) * X' * y;


end
