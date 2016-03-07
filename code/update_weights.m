function updated_model = update_weights(model,grad,hyper_params)

num_layers = length(grad);
a = hyper_params.learning_rate;
batch_size = hyper_params.batch_size; 
mu = hyper_params.mu;
% regularization
lmda = hyper_params.weight_decay;

step_size = 1e-3; % a normal stepsize should be in the range of 1e-2 to 1e-5
upperBound = 1e-2; lowerBound = 1e-5;

% Update the weights of each layer in your model based on the calculated gradients
if exist('momentum.mat','file') load momentum.mat
else 
    momentum = [];
    for i=1:num_layers
        momentum = [momentum, struct('W',0,'b',0)];
    end
end

for i=1:num_layers
    % regular SGD
    updateW = -a.*grad{i}.W./batch_size;
    updateB = -a.*grad{i}.b./batch_size;
    uw1 = norm(updateW(:)); % for debug
    % momentum
    updateW = updateW + mu*momentum(i).W;
    updateB = updateB + mu*momentum(i).b;
    % add in regularization
    updateW = updateW - (a*lmda).*model.layers(i).params.W;
    updateB = updateB - (a*lmda).*model.layers(i).params.b;
    % avoid blowing gradient and vanishing gradient
     normU = norm([updateW(:); updateB(:)]);
    % fprintf('update amount is %f\n', normU);
    model.layers(i).params.W = model.layers(i).params.W + updateW;
    model.layers(i).parmas.b = model.layers(i).params.b + updateB;
    model.momentum(i).W = updateW; model.momentum(i).b = updateB;
end

save('momentum.mat','momentum');

updated_model = model;