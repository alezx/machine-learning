function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%   X is a matrix m x n
%   y is a vector of size m
%   theta is a vector of size n
%   alpha is the learing rate
%   num_iters number of iteration we want to perform

%   Performs gradient descent to learn theta
%   updates theta by taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    predictions = X * theta;      % predictions of hypothesis on all m
    errors = (predictions-y);     % squared errors
    
    delta =  (1/m) * (X' * errors);

    theta = theta - (alpha * delta); % decrementing/incrementig teta

    % Save the cost J in every iteration    
    J_history(iter) = costFunctionVectorMultiplication(X, y, theta);

end

end
