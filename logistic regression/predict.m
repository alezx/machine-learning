function p = predict(theta, X)

% Predict whether the label is 0 or 1 using learned logistic 
% regression parameters theta.
%   i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

p = sigmoid(X * theta) >= 0.5;


end
