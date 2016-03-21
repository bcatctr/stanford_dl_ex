function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  y_hat1= theta'*X;
  y_hat = [y_hat1;zeros(1,size(X,2))];
  I = sub2ind(size(y_hat),y,1:size(y_hat,2));
  values = y_hat(I);
  f = -sum(log(exp(values)./sum(exp(y_hat),1)));
  %gradient matrix
  weights = bsxfun(@eq,y,(1:(num_classes-1))')-bsxfun(@rdivide,exp(y_hat1),sum(exp(y_hat1),1));
  for i = 1:1:(num_classes-1)
      g(:,i) = -sum(bsxfun(@times,X,weights(i,:)),2);
  end
  g=g(:); % make gradient a vector for minFunc

