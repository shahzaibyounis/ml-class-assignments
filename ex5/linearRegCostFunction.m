function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



h = X * theta;
J = sum((h-y).^2)/(2*m);

% regularized cost
thetaMinusFirstUnit = theta(2:length(theta), :);
J = J + (lambda/(2*m)) * sum(thetaMinusFirstUnit.^2);

% gradient
regGradientFirstUnit = sum((h-y) .* X(:,1)) / m;

if(length(theta) > 1)
    regGradientMinusFirstUnit = sum((h-y) .* X(:,2:end)) / m;
    regGradientMinusFirstUnit = regGradientMinusFirstUnit + (lambda/m) * theta(2:end,:)';

    grad = [regGradientFirstUnit; regGradientMinusFirstUnit'];
else
    grad = regGradientFirstUnit;
end



% =========================================================================

grad = grad(:);

end
