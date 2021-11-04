In Boltzmann machines, we created a Recommender System that predicted binary ratings "Like" or "Not Like". In this part i.e, autoencoders we will take it to the next level and create a Recommender System that predicts ratings from 1 to 5.

Directional neural networks
 takes input process them through hidden layer gives output and AIIMS output to be similar to the inputs

 not purely unsupervised we can say these are self-supervised deep learning models.

in the previous deep learning models like the som  and Boltzmann machines, there were  no output layers
 
used for:

 feature detection (hidden notes used as feature detectors)
 powerful recommender systems
 for encoding data

 the output  goes through softmax the maximum values are taken as 1 and all others as zero
 bias- used in the activation function

 overcomplete hidden layers
 if more or equal hidden notes are used than the input notes then the autoencoder can cheat so no proper encoding happens and some hidden notes are also unused.



 Sparse Autoencoder

If we want to extract more feature at an unrestricted  amount.

Here hidden layers’s nodes are more than input node’s layers by using a regularization technique which reduces overfitting.

we don't use all the hidden neurons in a single pass. In each single pass we use part of the hidden nodes/ neurons.

Denoising autoencoder

Another regularising technique to combat overcomplete hidden layers.

Take input layer left and create another hidden layer same as input but some of the inputs taken as zeroes in each pass while training.


Contractive autoencoders

Another regularization technique for overcomplete autoencoders. 

Adds penalty to the loss function when the function propagates backward in the network.

It specifically doesn't allow autoencoder to simply copy the input  values across output.

Stacked autoencoder

 Here we have another hidden layer in the the autoencoder.

 Two stages of encoding and one stage of decoding.

 A very very powerful algorithm can superseed Deep believe models (DBMs) which are undirected in nature where was stacked autoencoders are directional .

 Deep autoencoders

Stacked autoencoders and deep autoencoders are different.

Deep autoencoders are created when restricted Boltzmann machines (RBMs) are BM stacked over one another.

