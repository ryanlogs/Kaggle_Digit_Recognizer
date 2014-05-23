function [J grad] = nnCostFunction(nn_params, ...
                                   network, ...
                                   X, y, lambda)

	
	m = size(X,1);	
	num_layers = length(network);
	num_lables = network(3);
	
	input_layer_size = network(1);
	hidden_layer1_size = network(2);
	output_layer_size = network(3);
	
	Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size+1)), ...
						hidden_layer1_size, input_layer_size+1);
	read = 	hidden_layer1_size * (input_layer_size+1);
	
	
	Theta2 = reshape(nn_params(read + 1 : end ), ...
						output_layer_size, hidden_layer1_size+1);						
	
	
% You need to return the following variables correctly 
	J = 0;
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));	
	
	
	% computing cost
	a1 = [ones(m,1), X];

	z2 = (a1 * Theta1');
	a2 = sigmoid(z2);	
	a2 = [ones(m,1), a2];

	z3 = (a2 * Theta2');
	a3 = sigmoid(z3);	
	

	Y = zeros(m, num_lables);
	for i = 1:m,
		Y(i,y(i)+1) = 1;
	end;
	
	p1 = Y .* log(a3);
	p2 = (1 - Y) .* log(1 - a3);
	
	J = sum(p1 + p2) ;
	J = sum(J) / (-1 * m);
	
	T1 = Theta1(:,2:input_layer_size+1);
	T2 = Theta2(:,2:hidden_layer1_size+1);

	reg = sum(T1(:) .^ 2) + sum(T2(:) .^ 2);
	reg = (reg * lambda) / (2*m);
	
	J = J + reg;
	
	%computing gradient
	del3 = a3 - Y;
	
	del2 = (del3 * Theta2) ;
	del2 = del2(:,2:hidden_layer1_size+1);
	del2 = del2 .* sigmoidGradient(z2);
	
	delta2 = del3' * a2;  
	delta1 = del2' * a1;
	
	Theta1_grad = delta1 ./ m;
	Theta2_grad = delta2 ./ m;

	
	Theta1_grad(:,2:input_layer_size+1) +=  Theta1(:,2:input_layer_size+1) .* (lambda / m);
	Theta2_grad(:,2:hidden_layer1_size+1) +=  Theta2(:,2:hidden_layer1_size+1) .* (lambda / m);
	
	% Unroll gradients
	grad = [Theta1_grad(:) ; Theta2_grad(:)];
	
end