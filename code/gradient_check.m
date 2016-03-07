function gradient_check(model, layer)
    % given a model, check whether the implementation of layer layer is correct
    % init a model, should be passed in instead
    addpath layers; data_size = 10;
    
    ro_conv_net = [
        init_layer('ro_conv',struct('num_angle',3,'filter_height',3,'filter_width',3,...
            'filter_depth',1, 'num_channels',1,'num_filters',3,'circular',false))
        %init_layer('relu', []) % 26*26*3*3, 8*8*3*3
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        init_layer('ro_conv',struct('num_angle',3,'filter_height',3,'filter_width',3,...
            'filter_depth',3, 'num_channels',3,'num_filters',10,'circular',true))
        %init_layer('relu', []) % 24*24*3*10, 6*6*3*10
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        init_layer('3dAvgPool',struct('filter_size',2)) % 12*12*3*10, 3*3*3*10 filter_size is also the stride size
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ro_conv version of linear layer
        init_layer('ro_linear',struct('filter_height',3,'filter_width',3,...
            'filter_depth',3, 'num_channels',10,'num_filters',10,'dropout',true))
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         init_layer('flatten',struct('num_dims',5))
%         init_layer('linear',struct('num_in',12*12*3*3,'num_out',10)) % 10
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        init_layer('softmax',[])
    ];
    debug_net = [ % for debug: passed
        init_layer('flatten',struct('num_dims',4)),
        init_layer('linear',struct('num_in',28*28,'num_out',30)),
        init_layer('sigmoid',[]),
        init_layer('linear',struct('num_in',30,'num_out',10)),
        init_layer('softmax',[])
    ];
    conv_net = [
        init_layer('conv',struct('filter_size',3,'filter_depth',1,'num_filters',3)) % 26*26*3

        init_layer('pool',struct('filter_size',2,'stride',2))
%         
%         init_layer('addDim',[])
%         init_layer('3dpool',struct('filter_size',2)) % 13*13*1*3 filter size is stride
%         init_layer('minusDim',[])
%         
        init_layer('flatten',struct('num_dims',4))
        init_layer('linear',struct('num_in',13*13*3,'num_out',10))
        init_layer('softmax',[])];

    toy_ro_conv_net_3 = [
        init_layer('ro_conv', struct('num_angle',8,'filter_height',28,'filter_width',28,...
                   'filter_depth',1,'num_channels',1,'num_filters',1,'circular',false)) % 1*1*8*1
        %init_layer('relu', []) % 1*1*8*1
        init_layer('flatten',struct('num_dims', 5)) % 8
        init_layer('linear',struct('num_in',8,'num_out',1))
        init_layer('sigmoid',[])
    ];

    % layers, input_size (h*w*prev_num_angle*depth), output_size, visualize_each_layer
    model = init_model(ro_conv_net,[data_size data_size 1],10,true);

    % model = init_model(ro_conv_net,[28 28 1],10,true);
    layer = 2;
    fprintf('checking layer %d: %s\n', layer, model.layers(layer).type);
    % get some input and data
%    load_MNIST_data; % get train_data, train_label, test_data, test_label
    load small_one_vs_ten.mat
    fn_loss = @loss_crossentropy;
    train_data = data.train_data; train_label = data.train_label;    
%     % another dataset
%     load 'mnist_ones.mat'; fn_loss = @loss_sigmoid;
%     train_data = data.train_data; train_label = data.train_label;
    % get a batch of input
    el = randsample(size(train_data,4),1);
    data = train_data(:,:,:,el);
    label = train_label(el);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % perform gradient check
    epsilon = 1e-4;
    n_check = 1;
    for i=1:n_check
        if (~(strcmp(model.layers(layer).type,'relu') || strcmp(model.layers(layer).type,'softmax') || strcmp(model.layers(layer).type,'sigmoid') || ...
            strcmp(model.layers(layer).type,'pool') || strcmp(model.layers(layer).type,'3dpool') || strcmp(model.layers(layer).type,'addDim') || ...
            strcmp(model.layers(layer).type,'minusDim') || strcmp(model.layers(layer).type,'flatten') ||...
            strcmp(model.layers(layer).type,'3dAvgPool')))
            % for W
            W = model.layers(layer).params.W;
            real_gradW = zeros([numel(W) 1]);
            %j = randsample(numel(W),1);
            parfor j=1:numel(W)
                T0model = model; T0model.layers(layer).params.W(j) = W(j)-epsilon;
                T1model = model; T1model.layers(layer).params.W(j) = W(j)+epsilon;
                real_gradW(j) = (fn_loss(inference(T1model,data),label,[],true) - ...
                      fn_loss(inference(T0model,data),label,[],true)) / (2*epsilon);
            end  
            [output, activations] = inference(model, data); % run inference
            [loss, dv_output] = fn_loss(output,label,[],true); % calculate loss
            [grad] = calc_gradient(model,data,activations,dv_output); % get gradient
            %gradW = grad{layer}.W(j);
            gradW = grad{layer}.W(:);
            toc;
            
            % for b
            b = model.layers(layer).params.b;
            real_gradB = zeros([numel(b) 1]);
            %j = randsample(numel(b),1);
            parfor j=1:numel(b)
                T0 = b; T1 = b;
                T0(j) = b(j)-epsilon; T1(j) = b(j)+epsilon;
                T0model = model; T0model.layers(layer).params.b = T0;
                T1model = model; T1model.layers(layer).params.b = T1;
                real_gradB(j) = (fn_loss(inference(T1model,data),label,[],true) - ...
                      fn_loss(inference(T0model,data),label,[],true)) / (2*epsilon);
            end
    
            [output, activations] = inference(model, data); % run inference
            [loss, dv_output] = fn_loss(output,label,[],true); % calculate loss
            [grad] = calc_gradient(model,data,activations,dv_output); % get gradient
            % gradB = grad{layer}.b(j);
            gradB = grad{layer}.b(:);
            
            % calculate the relative error
%             fprintf('gradW: %d, real_gradW: %d\n',gradW,real_gradW);
%             fprintf('relative error for W is %d\n',abs(gradW-real_gradW)/(abs(gradW)+abs(real_gradW)));
%             fprintf('gradB: %d, real_gradB: %d\n',gradB,real_gradB);
%             fprintf('relative error for b is %d\n',abs(gradB-real_gradB)/(abs(gradB)+abs(real_gradB)));
            fprintf('relative error for W is %d\n', sum(abs(real_gradW-gradW)./(abs(real_gradW)+abs(gradW)+1e-27)));            
            fprintf('relative error for B is %d\n', sum(abs(real_gradB-gradB)./(abs(real_gradB)+abs(gradB)+1e-27)));                        
        end
             
        % for input
        [output, activations] = inference(model, data); % run inference
        if layer==1 input=data; else input = activations{layer-1}; end
        real_gradI = zeros([numel(input), 1]);
        parfor j=1:numel(input)
        %j = randsample(numel(input),1);
            T0 = input; T1 = input;
            T0(j) = input(j)-epsilon; T1(j) = input(j)+epsilon;
            real_gradI(j) = (fn_loss(fwd(model,T1,layer),label,[],true) - ...
                      fn_loss(fwd(model,T0,layer),label,[],true)) / (2*epsilon);
        end
    
        [output, activations] = inference(model, data); % run inference
        [loss, dv_output] = fn_loss(output,label,[],true); % calculate loss
        [dv_input] = bwd(model,data,activations,dv_output,layer);
        % gradI = dv_input(j); 
        gradI = dv_input(:); 
          
%         fprintf('gradI: %d, real_gradI: %d\n',gradI,real_gradI);
%         fprintf('relative error for input is %d\n\n',abs(gradI-real_gradI)/(abs(gradI)+abs(real_gradI)));
        fprintf('relative error for input is %d\n\n',sum(abs(gradI-real_gradI)./(abs(gradI)+abs(real_gradI)+1e-27)));

    end  

end

% input: batch_size * num_in_nodes 
% output: output of last layer
% activations: outputs of all layers
function [output,activations] = fwd(model, activation, level)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

if num_layers == 0 || level > num_layers
    fprintf('num layer is 0 or layer is larger than num_layer');
    return;
end
% FORWARD PROPAGATION CODE
if level == 1
    [output,activations] = inference(model,activation);
    return;
end

activations{level-1} = activation;
for i=level:num_layers
    layer = model.layers(i);
    [activations{i},~,~] = layer.fwd_fn(activations{i-1}, layer.params, layer.hyper_params, false, []);
end

output = activations{end};

end

function [dv_input] = bwd(model, input, activations, dv_output, level)
% Calculate the gradient at each layer, to do this you need dv_output
% determined by your loss function and the activations of each layer.
% The loop of this function will look very similar to the code from
% inference, just going in reverse this time.

num_layers = numel(model.layers);
grad = cell(num_layers,1);

if num_layers == 0
    return;
end
% Determine the gradient at each layer with weights to be updated
for i=num_layers:-1:2
    layer = model.layers(i);
    [~,dv_output,grad{i}] = layer.fwd_fn(activations{i-1}, layer.params, layer.hyper_params, true, dv_output);
    if level == i
        dv_input = dv_output;
        return
    end
end
layer = model.layers(1);
[~,dv_input,~] = layer.fwd_fn(input, layer.params, layer.hyper_params, true, dv_output);

end
