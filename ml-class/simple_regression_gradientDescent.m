function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
tmp = 0;

for iter = 1:num_iters
  x = X(:,2);
  h = theta(1) + (theta(2)*x);
  theta_zero = theta(1) - alpha * (1/m) * sum(h-y);
  theta_one  = theta(2) - alpha * (1/m) * sum((h - y) .* x);

  theta = [theta_zero; theta_one];
  
 
    J_history(iter) = computeCost(X, y, theta);

end

end
