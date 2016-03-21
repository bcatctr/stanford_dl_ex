function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
fun_Act = @sigmoid;
%% forward prop
%%% YOUR CODE HERE %%%
for d=1:numHidden
    if d == 1 
        hAct{d}.Z = stack{d}.W*data;
    else
        hAct{d}.Z = stack{d}.W*hAct{d-1}.A;
    end
    hAct{d}.Z = bsxfun(@plus,hAct{d}.Z,stack{d}.b);
    hAct{d}.A = fun_Act(hAct{d}.Z);
end
% Output layer should be normalized and is similar to softmax regression
hAct{numHidden+1}.Z = bsxfun(@plus,stack{numHidden+1}.W*hAct{numHidden}.A,stack{numHidden+1}.b);
hAct{numHidden+1}.A = bsxfun(@rdivide,exp(hAct{numHidden+1}.Z),sum(exp(hAct{numHidden+1}.Z),1));
pred_prob = hAct{numHidden+1}.A;


%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
% the cost calculation is similar to softmax_regression_vec but here is
% another implementation
ground_truth = full(sparse(labels,1:size(labels,1),1,10,size(labels,1)));
ceCost = - sum(sum(ground_truth.*log(pred_prob)));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
error_set= pred_prob - ground_truth;%output_dim*sample_num
%here all the gradient is the sum-form, not average-form
for l=numHidden+1:-1:1
    %sum up errors in different samples
    gradStack{l}.b = sum(error_set,2);
    %matrix multiplication automatically sum up deltaW in different sample
    if l==1
        gradStack{l}.W = error_set*data';
        break;
    else
        gradStack{l}.W = error_set*hAct{l-1}.A';
    end
    %error set can be updated column by column simultaneously in matrix
    %the derivative of f(z) is f(a)*(1-f(a)), the dot product means do 
    %element-wise update simultaneously
    error_set = (stack{l}.W)'*(error_set).*(hAct{l-1}.A).*(1-hAct{l-1}.A);
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for i = 1 : numHidden+1
    wCost = wCost + sum(stack{i}.W(:).^2);
end
cost = ceCost + 0.5*ei.lambda*wCost;

%penalty for no bias terms
for l=numHidden+1:-1:1
    gradStack{l}.W = gradStack{l}.W + ei.lambda*stack{l}.W;
end
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



