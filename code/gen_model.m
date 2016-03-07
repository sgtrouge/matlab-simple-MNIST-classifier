function model=gen_model()
    addpath layers;
    tic
    data_size = 28;
    % networks

    conv_net = [
        init_layer('conv', struct('filter_depth',1,'filter_height',data_size, 'filter_width', data_size,'num_filters',3,'filter_size',2));
        init_layer('relu',[])
        init_layer('pool', struct('filter_size', 2));
        init_layer('flatten',struct('num_dims',4));
        init_layer('linear', struct('num_in', (data_size-2)*(data_size-2)*3, 'num_out', 10));
        init_layer('softmax',[])
    ];

    model = init_model(conv_net,[data_size data_size 1],10,true);
toc
end
