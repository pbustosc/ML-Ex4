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
Theta1_grad = zeros(size(Theta1)); %25x401
Theta2_grad = zeros(size(Theta2)); %10x26

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

%----------------------------
%Parte 1 : Feedforward y cost function

%transformar y en matriz
f=ones(m,num_labels);
a=1:num_labels;
f=f.*a;
y=f==y; %5000x10
y=y';%10*5000

% Add ones to the X data matrix
X = [ones(m, 1) X];%5000x401
s=0;
w=0;
gradiente1=0;
gradiente2=0;

for a=1:m
Z2=Theta1*X(a,:)'; %25x1
A2=sigmoid(Z2);%25x1
A2=[ones(1,1); A2];%26x1
Z3=Theta2*A2; %10x1
A3=sigmoid(Z3); %10x1  

w=sum(y(:,a).*log(A3)+(1-y(:,a)).*log(1-A3)); 
s=s+w;
J=(-1/m)*s+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

delta3 = A3-y(:,a);%10x1
delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(Z2);%25x1

gradiente1=gradiente1+delta2*X(a,:);%25x401
gradiente2=gradiente2+delta3*A2'; %10x26

end

Theta1_grad(:,1)=gradiente1(:,1)*(1/m);
Theta1_grad(:,2:end)=gradiente1(:,2:end)*(1/m)+(lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,1)=gradiente2(:,1)*(1/m);
Theta2_grad(:,2:end)=gradiente2(:,2:end)*(1/m)+(lambda/m)*(Theta2(:,2:end));
% -------------------------------------------------------------
%Parte 2: Backpropagation algorithm



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
