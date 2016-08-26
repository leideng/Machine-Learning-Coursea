function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1

% get h_theta(x), i.e. a3

% Add ones to the X data matrix
X = [ones(m, 1) X];

%come out the matrix notation by hand in papers for the following
%procedure

z2 = Theta1*X'; %size: num_labels-by-m, all_prob(k,j) is the probability that the j-th example belongs to k-th label.
a2 = sigmoid(z2);

a2 = [ones(1,m);a2]; %remmeber to add 1 here

z3 = Theta2*a2;
a3 = sigmoid(z3); %size num_labels-by-m, each column is the corresonding value h_theta(x) for one particualr trainning example


% get corresponding Y matrixs
Y = zeros(m, num_labels);
for kk=1:num_labels
    Y(:,kk) = (y == kk);
end

Y = Y'; %size num_labels-by-m, each column is the corresonding vector y for one particualr trainning example

JM = -Y.*log(a3) - (1-Y).*log(1-a3);
J =  sum(sum(JM))/m;

% Part 3 add the regularization

% we should not be regularizing the terms that correspond to the bias
Theta1_temp = Theta1;
Theta1_temp(:,1) = 0;
Theta2_temp = Theta2;
Theta2_temp(:,1) = 0;
J = J + (lambda/(2*m))*( sum(sum(Theta1_temp.^2)) + sum(sum(Theta2_temp.^2)) );


% Part 2, use backpropogation to get the graidient

% without regularization
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for tt = 1:m % each loop is for one particular trainning example
    % step 1
    aa1 = X(tt,:)'; % the tt-th trainning example, we aleady add the 1's
    zz2 = Theta1*aa1;
    aa2 = sigmoid(zz2);
    aa2 = [1;aa2]; 
    zz3 = Theta2*aa2;
    aa3 = sigmoid(zz3); 
    
    % step 2
    yy = Y(:,tt);
    delta3 = aa3-yy;
    
    
    % step 3
    %kind of tricky here, I artifically add a 1 for zz2, i.e., zz2= [1,zz2]
    % it doest not matter to add what value, because we will remove
    % delta2(1) in step 4;
    delta2 = (Theta2'*delta3).*sigmoidGradient([1;zz2]);
    
    
    % step 4
    delta2 = delta2(2:end);
    Delta1 = Delta1 + delta2*aa1';
    Delta2 = Delta2 + delta3*aa2';
end

% step 5
Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

% add regularization term for the gradient
regular_term1 = lambda/m*Theta1;
regular_term2= lambda/m*Theta2;
regular_term1(:,1) = 0;
regular_term2(:,1) = 0;
Theta1_grad = Theta1_grad + regular_term1;
Theta2_grad = Theta2_grad + regular_term2;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
