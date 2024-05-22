import numpy as np
from pandas import read_csv
import tensorflow as tf
#from tensorflow.keras import Sequential
import tensorflow_probability as tfp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
tfd = tfp.distributions
tfpl = tfp.layers
import tf_keras 

# This function defines the prior function for the probabilistic deep network. 
# The "kernel_size" controls the number of parameters. 
# The prior distribution has no trainable variables, and has a zero covariance. 
def get_prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf_keras.Sequential([
       # tfpl.DistributionLambda(lambda t: 
       # tfp.distributions.Uniform(low=0.0, high=1, validate_args=False, allow_nan_stats=True, name='Uniform'))
                               
        tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag( loc=tf.zeros(n),scale_diag=tf.ones(n)))
    ])
    return prior_model


# This function defines the posterior function for the probabilistic deep network, 
# assuming it is of Gaussian functional form. 
# Covariance of the Gaussian is not zero, and is trainable. 
def get_posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf_keras.Sequential([
        tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfpl.MultivariateNormalTriL(n)
    ])
    return posterior_model

# This is used to renormalise the data for the tidal number 
def renormalize(x, k2min, k2max):
    return (x + np.abs(k2min))/(k2max + np.abs(k2min))

def produce_stats(data):
    # data: Input of sample for which to cmompute mean and std. Must be an array.
    # Returns: The mean and std. of the sample.  
    data_mean = data.mean()
    data_std = data.std() 
    return data_mean, data_std

# This function produces a statistical sample from the deep network 
def produce_sample_Lambda(X_test, model_def, n_samples = 50):
    # n_samples: The number of points in a given sample (i.e statistical realisations of one output)
    # Returns: An array with the sample 
    #X_test = test_data[0]
    #y_test = test_data[1] 
    sample = [] # list to append the sample
    model = model_def
    for _ in range(0, n_samples): # produce n_samples realisations at fixed input i0
        testX1 = np.expand_dims(X_test , axis=0) # The input data for the deep network
        result_testX1 = model(testX1)  # Given input data, make prediction for the features: (mass, c_s)
        Lambda_value = result_testX1[0][0] 
        sample.append(Lambda_value)
        
    sample = np.array(sample)  # Produce array of the sample list         
    return sample
    
# This function produces a statistical sample from the deep network 
def produce_sample_cs(X_test, model_def, n_samples = 50, check_range = False):
    # n_samples: The number of points in a given sample (i.e statistical realisations of one output)
    # check_range: If True, then checks whether predicted masses and radii are [0,1]
    # Returns: An array with the sample 
    #X_test = test_data[0]
    #y_test = test_data[1] 
    sample = [] # list to append the sample
    model = model_def
    for _ in range(0, n_samples): # produce n_samples realisations at fixed input i0
        testX1 = np.expand_dims(X_test , axis=0) # The input data for the deep network
        result_testX1 = model(testX1)  # Given input data, make prediction for the features: (mass, c_s)
        mass_values = result_testX1[0][1:8] # 7 mass values - the "x-axis" 
        cs_values = result_testX1[0][8:15] # 7 sound speed values - the "y-axis" 
        data_X1_zip = zip( np.array(mass_values), np.array(cs_values) )  # create pairs (mass, c_s)
        
        if check_range == True: # check that the predicted (mass, c_s) are within [0,1]
            check_mass = any(element > 1. or element < 0.  for element in mass_values) # tests if 0 < mass < 1 
            check_cs   = any(element > 1. or element < 0. for element in cs_values) # tests if 0 < cs < 1
            if check_mass == False and check_cs == False: 
                sample.append(list(data_X1_zip)) # Append the given realisation of predicted (mass, c_s) to the list
        else:
            sample.append(list(data_X1_zip))
            
    sample = np.array(sample)  # Produce array of the sample list         
    return sample


# This function evaluates the results of the deep network for the sound speed prediction according to the statistical pipeline.
def produce_evaluation(test_data, input_list, model, n_samples = 100, α = 2, check_range = True): 
      # input_list: list of integers denoting the input data to consider for evaluation
      # n_samples: number of samples for each datapoint
      # α: the sigma level to consider as accepted range for predicted data 
      # Returns: A list with the percentage of "accepted" predictions, i.e predicted points which lie in the allowed interval.
    eval_data_x_cs = [] # list to append succesful evaluation of the mass and cs predicion (success = 1)
    X_test = test_data[0]
    y_test = test_data[1]
    for i in input_list:
        sample = produce_sample_cs(X_test[i], model, n_samples = n_samples, check_range = check_range)
        sample = np.array(sample) 
        col_i = i 
        real_Lambda, real_mass, real_cs = y_test[i][0], y_test[i][1:8], y_test[i][8:15] 
        data_temp_x = [] 
        data_temp_cs = [] 
        data_temp_x_cs = [] 
        #mean_Lambda, std_Lambda = np.mean(sample[:,0]), np.std(sample[:,0])
        for j in range(0,len(real_mass)): 
            mean_x, std_x = np.mean(sample[:,j][:,0]), np.std(sample[:,j][:,0])
            mean_cs, std_cs  = np.mean(sample[:,j][:,1]), np.std(sample[:,j][:,1])
            if mean_x - α*std_x < real_mass[j] < mean_x + α*std_x and mean_cs - α*std_cs < real_cs[j] < mean_cs + α*std_cs:
                data_temp_x_cs.append(1)           
        eval_data_x_cs.append([len(data_temp_x_cs), len(sample)])
    return eval_data_x_cs

# Produces the plot with the sample output of the deep network for the sound speed, mean values, std's, and real values
def produce_sample_plot(sample):
    # sample: an array with a sample/realisation of pairs of (mass, c_s)
    # Returns: A scatter plot with the sample and mean/std. values.
    stats_x = []
    stats_cs = []
    data_sigma = []
    for j in range(0,7): # range of masses/c_s values
        mean_x, std_x = produce_stats(sample[:,j][:,0]) #np.mean(sample[:,j][:,0]), np.std(sample[:,j][:,0])
        mean_cs, std_cs  = produce_stats(sample[:,j][:,1]) #np.mean(sample[:,j][:,1]), np.std(sample[:,j][:,1])
        stats_x.append([mean_x, std_x])
        stats_cs.append([mean_cs, std_cs])
    stats_x = np.array(stats_x)
    stats_cs = np.array(stats_cs)    
    for i in range(len(sample)):
        plt_i = plt.scatter(sample[i][:,0], sample[i][:,1], alpha=i/len(sample))
    plt.errorbar(stats_x[:,0], stats_cs[:,0], xerr = 2*stats_x[:,1], yerr = 2*stats_cs[:,1], fmt ='o', color = 'black')
    plt.xlabel("Mass (normalised)")
    plt.ylabel("Sound speed")
    #plt_i = plt.scatter(stats_x[:,0], stats_cs[:,1],color = 'black')


# This function is helpful to bin data in case needed. 
def produce_binning_cs(i0, sample, grid_points = 100):
    data_binned = []
    data_mean = []
    x_range = np.linspace(0,1, grid_points)
    dx = 1./grid_points
    for i in range(0,len(x_range)):
        select_i = filter(lambda x:  x_range[i] <= x[0] <= x_range[i]+dx, sample2)
        elem_i = np.array(list(select_i))
        if len(elem_i) > 0:
            data_binned.append([np.sum(elem_i [:,0])/(len(elem_i [:,0])), elem_i[:,1]])
    for i in range(0,len(data_binned)):
        x_i = data_binned[i][0]
        cs_mean_i = data_binned[i][1].mean()
        cs_std_i = data_binned[i][1].std()
        data_mean.append([x_i, cs_mean_i, cs_std_i])
    data_mean = np.array(data_mean)
    return np.array(data_mean)


# This function evaluates the results of the deep network for the cosmological constant according to the statistical pipeline.
def produce_evaluation_Lambda(test_data, input_list, model, n_samples = 100, α = 2): 
      # input_list: list of integers denoting the input data to consider for evaluation
      # n_samples: number of samples for each datapoint
      # Returns: A list with the mean and std of the cosmological constant values
    X_test = test_data[0]
    y_test = test_data[1]
    eval_data_Lambda = []
    for i in input_list:
        sample = produce_sample_Lambda(X_test[i], model, n_samples = n_samples)
        sample = np.array(sample) # the sample of the cosm. constant predictions at gtiven input
        real_Lambda = y_test[i][0] # the 'real' cosmological constant/feature, as predicted by the theory.
        mean_Lambda, std_Lambda = np.mean(sample), np.std(sample)
        if mean_Lambda - α*std_Lambda < real_Lambda < mean_Lambda + α*std_Lambda:
            eval_data_Lambda.append(1.) # append 1 if the evaluation criterion is met. Final evaluation will be the 
                                        # length of this list divided by the length of the inputs tried
    
    # returns the accuracy = (number of sucesfull evaluations)/(total inputs)          
    return len(eval_data_Lambda)/len(input_list)

def plot_pred_vs_real(real_mass, real_cs, predicted_data): 
    # predicted_data are the output of ''produce_binning_cs()''. Must be an array.
    # real_mass sets the x-grid. Must be an array
    # real_cs sets the ''real'' c_s. Must be an array
    x = np.linspace(0, 1, 50)
    c_s_interp = np.interp(x, predicted_data[:,0], predicted_data[:,1])
    sigma_interp = np.interp(x, predicted_data[:,0], predicted_data[:,2])
    plt.errorbar(x,c_s_interp, sigma_interp, color = 'black')
    #plt.scatter(real_mass, real_cs, color = 'orange')

# Useful callback 



# The definition of the deep model 
def produce_model(train_data, units, activation, prior, posterior, activation_last_layer=False ):
    
    if activation_last_layer == True:
        final_activation = 'sigmoid'
    else:
        final_activation = None 

    n_features = train_data[0].shape[1]
    norm = train_data[0].shape[0]
    
    model = tf_keras.Sequential([
    
    tfpl.DenseVariational(input_shape=(n_features,), units=units,
                              make_prior_fn = get_prior,
                              make_posterior_fn=get_posterior,
                              kl_weight=1/norm,
                              kl_use_exact=False,
                              activation=activation), 
    
    tfpl.DenseVariational(input_shape=(n_features,), units=units,
                              make_prior_fn = get_prior,
                              make_posterior_fn=get_posterior,
                              kl_weight=1/norm,
                              kl_use_exact=False,
                              activation=activation), 
    
    tfpl.DenseVariational(input_shape=(n_features,), units=units,
                              make_prior_fn = get_prior,
                              make_posterior_fn=get_posterior,
                              kl_weight=1/norm,
                              kl_use_exact=False,
                              activation=activation), 
    
    tfpl.DenseVariational(input_shape=(n_features,), units=units,
                              make_prior_fn = get_prior,
                              make_posterior_fn=get_posterior,
                              kl_weight=1/norm,
                              kl_use_exact=False,
                              activation=activation), 

    # Output layer with activation function.
    tfpl.DenseVariational(units=15, 
                              make_prior_fn = get_prior,
                              make_posterior_fn=get_posterior,
                              kl_weight=1/norm,
                              kl_use_exact=False,
                              activation = final_activation)

                   ])
    return model

# Custom callback 
def training_stop(terminate_at_accuracy = 0.75):
    
    class CustomCallback(tf_keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') >= terminate_at_accuracy:
                self.model.stop_training = True
    callback = CustomCallback()
    return callback 

# Initiates the training of the model 
def train_model(model, train_X, train_y, epochs, callbacks = None):
    
    history = model.fit(train_X, train_y, epochs, verbose=1, callbacks=callbacks)

    return history 
    
    
    
    
