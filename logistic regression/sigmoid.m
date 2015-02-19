function g = sigmoid(z)

% J = SIGMOID(z) computes the sigmoid of z.
% (z can be a matrix, vector or scalar).


g = 1 ./ (1+(e.^-z));

end
