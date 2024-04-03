# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:08:11 2022

@author: Manu
"""

import numpy as np
import random as rnd
import json 

from model.neuralnetwork import NeuralNetwork

def generate_public_key(private_key, size_public_key=None, seed=None, key_generator_nn=None, 
                        save=True, **optionalArgs):
    
    # Convert the private_key in int    
    private_key_int = np.array(list(private_key), dtype=int)
    
    meta_kwargs = {}
    # Generate network (if needed)
    if key_generator_nn is None:
        # Size of input and output of the network
        size_private_key = len(private_key)    
        if size_public_key is None:
            size_public_key = size_private_key
        # Create key_generator using NeuralNetwork
        meta_kwargs.update(dict(inputSize=size_private_key,
                                outputSize=optionalArgs.get('outputSize', size_public_key),
                                hiddenSize=optionalArgs.get('hiddenSize', size_public_key),
                                nbHiddenLayer=optionalArgs.get('nbHiddenLayer', 100))
                      )
        key_generator_nn = NeuralNetwork(**meta_kwargs)
        meta_kwargs.update(dict({'key_generator':key_generator_nn.__version__}))
    else:
        # If the key_generator is provided, we strongly recommend storing the architecture parameters.
        # Store it in `custom` inside the optionalArgs dic, to be able to regenerate the public_key from the private_key
        customArchitecture = optionalArgs.get('custom', dict(custom="WARNING: No information provided"))
        meta_kwargs.update(customArchitecture)
        
    # Generate weights
    # Seed: for security it's CRITICAL that the seed cannot be found.
    # This is why it is preferable to use the private key as the seed.
    if seed is None:
        np.random.seed(private_key_int) 
        # Notice the format of the private key provided in the seed.
        # private_key_int is an array of int.
    else:
        np.random.seed(seed)
    mu = optionalArgs.get('mu', -0.5)
    sigma = optionalArgs.get('sigma', 100000000) # The higher sigma the better the security
    random_weights = np.random.normal(mu, sigma, key_generator_nn.numberSynapses)
    key_generator_nn.setWeights(random_weights)
    
    # Format public key
    public_key_int = key_generator_nn.run(private_key_int, option='step')
    public_key_int = np.array(public_key_int, dtype=int)   
    public_key_str = np.array(public_key_int, dtype=str); public_key_str="".join(public_key_str)
    
    # Save public_key and metadata
    meta_kwargs.update(dict(mu=mu, sigma=sigma))
    dic_public_key = {'public_key(str)':public_key_str,
                      'metadata':meta_kwargs,}
    if save:
        with open("public_key.txt", "w") as fp:
            json.dump(dic_public_key , fp) 
    
    return public_key_int, dic_public_key


def fake_private_key(n, seed=None, periodic=False):
    
    if not periodic:
        if seed is not None:
            np.random.seed(seed)
        private_key = np.random.randint(0, 2, n)
    else:
        private_key = np.zeros(n)
        step = int(n/10)
        private_key[1:n:step] = 1
    return np.array(list(private_key), dtype=int)

def bin_flip(array_bin, nb_flip):
    array_bin = array_bin.copy()
    indexes = np.arange(len(array_bin))
    indexes_flip = rnd.sample(list(indexes), nb_flip)    
    array_bin[indexes_flip] = 1 - array_bin[indexes_flip]
    return array_bin

