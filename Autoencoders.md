In Boltzmann machines, we created a Recommender System that predicted binary ratings "Like" or "Not Like". In this part i.e, autoencoders we will take it to the next level and create a Recommender System that predicts ratings from 1 to 5.

### Autoencoders

* Directional Deep learning neural network model i.e, takes inputs, processes them through hidden layers and gives output and aims output to be similar to the inputs.

* Not purely unsupervised we can say these are self-supervised deep learning models.

In the previous deep learning models like the som and Boltzmann machines there were no output layers.

<img align="Center" height="250px"  src="https://user-images.githubusercontent.com/85345738/140345675-77a5d6d8-687f-43b2-8d5b-59353d9c6c09.png" />

**used for:**

 * feature detection (hidden notes used as feature detectors)
 * powerful recommender systems
 * for encoding data

 The output of the model goes through softmax. So the maximum values are taken as 1 and all others as zero.
 
Bias- used in the activation function

**Overcomplete hidden layers** 
----
If more or equal hidden notes are used than the input notes then the autoencoder can cheat so no proper encoding happens and some hidden notes are also unused.

<img align="Center" height="200px"  src="https://user-images.githubusercontent.com/85345738/140350419-10c85135-abd4-4632-a284-e5480e9fad43.jpeg" />


**Sparse Autoencoder**

* If we want to extract more feature at an unrestricted amount.
* Here, hidden layers’s nodes are more than input node’s layers by using a regularization technique which reduces overfitting.
* we don't use all the hidden neurons in a single pass. In each single pass we use part of the hidden nodes/ neurons.

<img align="Center" height="250px" width="600px"  src="https://user-images.githubusercontent.com/85345738/140350619-09c2849f-359b-4cf7-9594-3ddee5dbd428.png" />

**Denoising autoencoder**

* Another regularising technique to combat overcomplete hidden layers.
* Takes the input layer left and create another hidden layer same as input but some of the inputs taken as zeroes in each pass while training.

<img align="Center" height="250px" width="600px"  src="https://user-images.githubusercontent.com/85345738/140347448-31c942c9-c391-45d2-95b4-28660adb4019.png" />


**Contractive autoencoders**

* Another regularization technique for overcomplete autoencoders. 
* Adds penalty to the loss function when the function propagates backward in the network.
* It specifically doesn't allow autoencoder to simply copy the input  values across output.


**Stacked autoencoder**

* Here, we have another hidden layer in the the autoencoder.
* Two stages of encoding and one stage of decoding.
* A very very powerful algorithm can superseed Deep believe models (DBMs) which are undirected in nature where was stacked autoencoders are directional .


**Deep autoencoders**

* Stacked autoencoders and deep autoencoders are different.
* Deep autoencoders are created when restricted Boltzmann machines (RBMs) are BM stacked over one another.

