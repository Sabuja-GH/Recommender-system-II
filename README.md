# Recommender-system-II
An Stacked autoencoder (SAE) movie recommender system.

The importing of datasets and preprocessing part of the model is same as Boltzmann machines. But without converting the ratings into binary ratings.

## Creating the Architecture of autoencoder

We will make an autoencoder class then create an object of the class.
we will also use inheritance when we use nn.module class (parent class) to use all its methods 

SAE-Stacked autoencoder 

We will define two functions one is __init__ and other one is forward

1. __ init __ function:

    * super(SAE, self).__init__()- After self,  there is a comma because it will consider all the variables in the parent class also.

        * Super- To inherit methods and classes of the nn module.
        * Self- refer to the autoencoder itself.

    * self.fc1- It is the first fully connecting layer related to our autoencoder object which is represented by self. 

        *This represents the full connection of the first input vector of features (input layer) and the first encoded vector i.e, first hidden layer.*

        * nn.Linear- nn is the module linear is the class.
        * (nb_movies,20)- 20, nodes in the first hidden layer. can be adjusted for better result.
        * nb_movies- as we will give ratings of all the movies for a single user as input.
  
        So finally, our first encoded vector is a full vector of 20 elements, acts as features detectors of the movies.

    * self.fc2- Second  full connection layer

        * (20,10)- 20,  Number of neurons between the first hidden layers.
        * 10- neurones in the second hidden layer to detect more features.

    * self.fc3- third full connection layer, here we do decoding so we need to make a symmetry so we will use (10,20).
    * self.fc4- output layer, number of neurons or nodes in the layer will  the equal to to input layer neurons.

    * specify an activation function which activates the neurons when observations go into the network.
      Sigmoid activation function gives better results rather than other functions in this specific problem.

2. forward function:
    It returns the predicted rating vector which is compared to the input vector to measure the loss function and update the weights.
 
    X- initially it is the input vector.

    we will encode and decode it twice:
    * The first Self.activation creates the first encoded vector which is used as the new X for the next Full connection layer.
    * The 2nd Self.activation is also an encoded vector. 
    * The 3rd Self.activation is a decoded vector. 
    * The 4th Self.activation is the output layer vector (no activation needed in the output layer).
  
* We create an object of the SAE class
* Criterion-  object of the nn.MSELoss class, used for loss function by measuring the mean square error.

* Optimizer - Object of the nn module and RMSprop class (optim.RMSprop)

    Optimizer contains three inputs:
    * sae.parameters()-  all the parameters of our autoencoder.
    * Lr- learning rate
    * Decay-  helps reduce the learning rate after a few epochs to regulate the conversion.


## Training the autoencoder

* First, we will look over all the epochs.
* Now, we will look over all the users/observations.

* The input here is a vector but the tensor will not take a single vector as input. so, we need to add an additional dimension Variable. 
    * (training_set[id_user]).unsqueeze(0) that creates a batch of the single input vector.

* Target- copy of the input which will be later compared with The input which is encoded and decoded.

* if torch.sum(target.data > 0) > 0:

  Will loop over the users weho atleast rated one movie-  saves computation memory. 
  * target.data- all the ratings for the user in the loop.

* Output- vector of predicted ratings, Calculated by using the forward method of the SAE class.

* target.require_grad = False

  This makes sure that we don't compute the gradient wrt the target twitch saves a lot of computation. 
* output[target == 0] = 0

    * we only include the non zero values in the computation so we don't deal with the movies that the  user didn't rate. That is only for the output vector.

    * These values would be counted in the computation and the error so they won't have impact on the update of different weights right after measuring the loss.

* Loss- calculates the loss(loss function)
* mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)

    * represents average of the error but by only considering the movies that were rated. Because we only consider the movies that had non zero ratings Previously in the code i.e, in the if statement. 
    * (+1e-10)- Added to make sure The denominator is not zero
    * float(torch.sum(target.data > 0)- Number of movies that have positive ratings

* loss.backward()- Backward Method, it will tell in which direction we need to update the weights do we need to increase or decrease the weight.

* train_loss += np.sqrt(loss.data*mean_corrector)

    * We take that part of of the Loss object that contains the error i.e, loss.data[0]/loss.data.
    * Now we will use mean corrector (adjustment factor). So, we are adjusting training loss by using mean corrector. 
    * we will use 1° loss so we will take square root of (loss.data*mean_corrector) by using np.sqrt method.

* S- increment counter i.e, users that rated at least one movie.
    * 1.- Dot is there to ensure s remains floating integer.

* Optimizer- will be used to update the weights 

   * .step()- method of the RMS prop class. 

*Backward method decides the direction in which the weights will be updated that is if they will be increased or decreased whereas Optimizer decides the intensity of the weight update.*

* At last we print the epoch and subsequent losses.

## Testing the SAE

**We will just use the training code and edit it** 

*We do not need 200 epochs but just one epoch.*

* Input- training set is kept as it is
    * The input vector is fed  into the network and we get the final output and then we compare it to the target which contains ratings from the test set.

*Training_set-  Ratings of all the movies including the movies which the user hasn’t watched.*

*Test_set-  ratings of the movies including the movies user watches over time. so we can compare the real test set ratings with the the Predicted ratings in the end.*

* loss.backward()  is removed as it was related to back propagation and backpropagation is related to training of a model.
    * In measuring test_set losses we are not updating any weights. so, we have removed it. 

* Train loss---test loss

* Optimizer- also removed as it was related to back propagation and backpropagation is related to training.

Rest of the code is kept same and finally we print the test_loss for he single epoch.
