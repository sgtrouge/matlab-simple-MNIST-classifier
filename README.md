# matlab-simple-MNIST-solver
This repo is heavily based on Nathan Wang's work @github.com/nathanwang000/ROCNN.

#Tasks: 
- Train a classifier from 60k training data and test on 10k data, each input is a 28x28x1 image.

#How to use:
- Call load_image_data
- Generate the CNN model using gen_model.m 
(Feel free to configure the model's layer. The current one is able to achieve > 96% in 40 mins accuracy w/ intel i5 and Iris).
More on CNN layers: @cs231n.github.io/convolutional-networks/
- Train and test using train.m

# Work-to-do:
- Try deepen the convo layers, and stacks mulitple convo-relu-pool layers. Will requires heavy resource, with best error accuracy about < 0.5%.
