# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:19:00 2023

@author: Manu
"""

# Standard library
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import time 
import hashlib

# The model and key generator
from model.neuralnetwork import NeuralNetwork
from model.public_key_generator import generate_public_key, fake_private_key, bin_flip

# Evaluation of the bientropy
'''
Check this: https://pypi.org/project/BiEntropy/
'''
from bientropy import bien, tbien
from bitstring import Bits

from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 8
rcParams['font.size'] = 12


# Make folder to save results
script_dir = os.path.dirname(__file__) # Get script directory
folder = 'result\\' 
results_dir = os.path.join(script_dir, folder) # Concatenate path inteligently
if not os.path.isdir(results_dir): # Check directory is not existing
    os.makedirs(results_dir) 


def hamming_distance(word1, word2, norm=True):
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(word1, word2))
    return hamming_distance/len(word1) if norm else hamming_distance


def hexa_to_bin(ini_string):
    n = int(ini_string, 16) 
    bStr = ''
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1    
    return [int(i) for i in bStr]


def sha256_key(input_key_str):
    m = hashlib.sha256(b""+input_key_str)
    hex_key = m.hexdigest()
    key_int = hexa_to_bin(hex_key)
    key_str = np.array(key_int, dtype=str); key_str="".join(key_str)
    return key_int, key_str
    

def array_to_str(array_int):
    key_str = np.array(array_int, dtype=str); 
    return b"".join(key_str)


def display_phase_transition(nb_samples, n, mu, sigmas):
    print("Display phase transition of the neural network with BiEntropy")
    filename = f"phase_transition_entropy_public_key_generator"
    
    entropies_vs_sigmas = {'mean':[], 'var':[]}
    for sigma in tqdm(sigmas):        
        # Weights parameters
        option_weights = dict(mu=mu, sigma=sigma)
        
        hamming_infos = []
        entropies = []
        # Repeat the experiment to get solid statistics
        for i in range(nb_samples):
            # Generate fake private key
            priv_key_int = fake_private_key(n, seed=i)
        
            # Generate the public key            
            public_key_int, dic_public_key = generate_public_key(priv_key_int, 
                                                                 save=False, 
                                                                 **option_weights)
            public_key_str = dic_public_key['public_key(str)']
    
            # Compute entropy of public key
            entropies.append(tbien(Bits('0b'+public_key_str)))

        # Update dic to plot results        
        entropies_vs_sigmas['mean'].append(np.mean(entropies))
        entropies_vs_sigmas['var'].append(np.var(entropies))
        
        # Save results
        np.save(results_dir+filename, entropies_vs_sigmas)
        
    # Plot results
    fig = plt.figure()
    fig.suptitle('Phase transition, exhibited by the BiEntropy of ANN outputs')
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('$<BiH>$')
    ax1.set_title('Average')
    ax1.plot(sigmas, entropies_vs_sigmas['mean'])
    ax2 = fig.add_subplot(212)
    ax2.plot(sigmas, entropies_vs_sigmas['var'])
    ax2.set_xlabel('$\sigma(W)$')
    ax2.set_ylabel('$Var(BiH)$')
    ax2.set_title('Variance')
    fig.tight_layout()
    
    # Save fig
    fig.savefig(results_dir+filename+'.png')
    fig.savefig(results_dir+filename+'.pdf')


def display_non_linearity(nb_samples, n, mu, sigmas,    
                          nb_flip = 1):
    print("Show the non-lineatity of the network with hamming distance")
    filename = f"phase_transition_hamming-distance_public_key_generator"
    
    distance_vs_sigmas = {'input':[], 'output':[]}
    for sigma in tqdm(sigmas):
        option_weights = dict(mu=-0.5, sigma=sigma)

        mutual_infos_input = []
        mutual_infos_output = []
        for i in range(nb_samples):
            
            # Compute output for key and key flipped
            priv_key = fake_private_key(n, periodic=True)
            key_int_1, key1 = generate_public_key(priv_key, 
                                                  save=False, 
                                                  **option_weights)
            key_str_1 = key1['public_key(str)']
            priv_key_flip = bin_flip(priv_key, nb_flip)
            key_int_2, key2 = generate_public_key(priv_key_flip, 
                                                  save=False, 
                                                  **option_weights)
            key_str_2 = key2['public_key(str)']
                        
            # Compute mutual information between private and public key 
            D = hamming_distance(priv_key, key_int_1)
            mutual_infos_input.append(D)
            D = hamming_distance(key_int_1, key_int_2)
            mutual_infos_output.append(D)
        
        # Update dic
        distance_vs_sigmas['input'].append(np.mean(mutual_infos_input))
        distance_vs_sigmas['output'].append(np.mean(mutual_infos_output))
    
    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_xlabel('$\sigma(W)$')
    ax.set_ylabel('$<D(input, output)>$')
    ax.plot(sigmas, distance_vs_sigmas['input'])
    ax.set_title(f'Hamming distance between input and output')
    
    ax1 = fig.add_subplot(212)
    ax1.set_xlabel('$\sigma(W)$')
    ax1.set_ylabel('$<D(output1, output2)>$')
    ax1.plot(sigmas, distance_vs_sigmas['output'])
    ax1.set_title(f'Hamming distance between two outputs obtained with {nb_flip} bit flip in input')
    fig.tight_layout()
    
    # Save fig
    fig.savefig(results_dir+filename+'.png')
    fig.savefig(results_dir+filename+'.pdf')    


def find_best_ann(nb_samples, n, sigmas, mu=-0.5):
    print("Find the best parameter to maximize encryption")
    filename = f"best_ann_bientropy_hamming_distance"
    
    dic_result = {'BiH':[], 'D':[]}
    for sigma in tqdm(sigmas):
        
        option_weights = dict(mu=mu, sigma=sigma,
                              nbHiddenLayer=100)
        
        D = []
        B = []
        for i in range(nb_samples):
            
            # Generate fake private key
            priv_key = fake_private_key(n, periodic=True)
            priv_key_str = np.array(priv_key, dtype=str); priv_key_str=b"".join(priv_key_str)
    
            # Generate key with ANN
            key_int_1, dic_key = generate_public_key(priv_key, 
                                                     save=False, 
                                                     **option_weights)    
            key_str_1 = dic_key['public_key(str)']
            
            # Compute hamming distance
            D.append(hamming_distance(priv_key, key_int_1))
            
            # Compute entropy of public key
            B.append(tbien(Bits('0b'+key_str_1)))
            
            
        # Update dic
        dic_result['BiH'].append(np.mean(B))    
        dic_result['D'].append(np.mean(D))

    # Limit case mu=0.0
    option_weights = dict(mu=0, sigma=1)
    key_int, dic_key = generate_public_key(priv_key, 
                                             save=False, 
                                     **option_weights)  
    ## Compute hamming distance
    D_lim = hamming_distance(priv_key, key_int_1)
    
    ## Compute entropy of public key
    B_lim = tbien(Bits('0b'+key_str_1))
    
    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_xlabel('$\sigma(W)$')
    ax.set_ylabel('$<BiH(output)>$')
    ax.plot(sigmas, dic_result['BiH'], '.-')
    ax.axhline(y=B_lim, ls='--', linewidth=3, color='green')
    ax.set_xscale('log')
    ax.set_title(f'BiEntropy of the output')
    
    ax1 = fig.add_subplot(212)
    ax1.set_xlabel('$\sigma(W)$')
    ax1.set_ylabel('$<D(input, output)>$')
    ax1.plot(sigmas, dic_result['D'], '.-')
    ax1.axhline(y=D_lim, ls='--', linewidth=3, color='green')
    ax1.set_title(f'Hamming distance between input and output')
    ax1.set_xscale('log')
    fig.tight_layout()
    
    # Save fig
    fig.savefig(results_dir+filename+'.png')
    fig.savefig(results_dir+filename+'.pdf')    
  
def comparison_sha256_ann(nb_samples, n, mu=-0.5, sigma=1000):
    print("Compare the performances of SHA256 and ANN with BiEntropy and hamming distance")
    
    option_weights = dict(mu=mu, sigma=sigma,
                          nbHiddenLayer=100)
    
    B_dic = {'ann':[], 'sha256':[]}
    D_dic = {'ann':[], 'sha256':[]}
    timings = {'ann':[], 'sha256':[]}
    for i in tqdm(range(nb_samples)):
        
        # Generate fake private key
        priv_key = fake_private_key(n, periodic=True)
        priv_key_str = np.array(priv_key, dtype=str); priv_key_str=b"".join(priv_key_str)

        # Generate key with ANN
        time_start = time.perf_counter()
        key_int_1, dic_key = generate_public_key(priv_key, 
                                                 save=False, 
                                                 **option_weights)
        time_elapsed = (time.perf_counter() - time_start)
        timings['ann'].append(time_elapsed)

        key_str_1 = dic_key['public_key(str)']
        
        # Generate key with SHA256
        time_start = time.perf_counter()
        key_int_2, key_str_2 = sha256_key(priv_key_str)
        time_elapsed = (time.perf_counter() - time_start)
        timings['sha256'].append(time_elapsed)
        
        # Compute hamming distance
        D1 = hamming_distance(priv_key, key_int_1)
        D2 = hamming_distance(priv_key, key_int_2)
        
        # Compute entropy of public key
        B1 = tbien(Bits('0b'+key_str_1))
        B2 = tbien(Bits('0b'+key_str_2))
        
        # Update dic
        B_dic['ann'].append(B1)
        B_dic['sha256'].append(B2)

        D_dic['ann'].append(D1)
        D_dic['sha256'].append(D2)
    
    print(f'Comparison of the two methods for {nb_samples} samples')
    print('----> BiEntropy')
    print('sha256 -> BiH(output):', np.mean(B_dic['sha256']))
    print('ann -> BiH(output):', np.mean(B_dic['ann']))
    print('----> Hamming distance')
    print('sha256 -> D(input, output):', np.mean(D_dic['sha256']))
    print('ann -> D(input, output):', np.mean(D_dic['ann']))
    print('----> Timings')
    print('sha256 -> average time (s):', np.mean(timings['sha256']))
    print('ann -> average time (s):', np.mean(timings['ann']))

    
def non_linearity_comparison(nb_samples, n, mu=-0.5, sigma=1000):
    print("Compare the non-linearity of SHA256 and ANN with hamming distance")
    filename = f'non-linearity_comparison_sha256_ann_public_key_generator'
    
    nb_flips = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50]
    option_weights = dict(mu=mu, sigma=sigma)
    
    D_dic = {'ann':{'mean':[], 'var':[]}, 
             'sha256':{'mean':[], 'var':[]}}
    for nb_flip in tqdm(nb_flips):
        
        D1 = []
        D2 = []
        for i in range(nb_samples):
            # Generate fake private key
            priv_key = fake_private_key(n, periodic=True)
            priv_key_flip = bin_flip(priv_key, nb_flip)
            priv_key_str = array_to_str(priv_key)
            priv_key_flip_str = array_to_str(priv_key_flip)
    
            # Generate key with ANN
            key_int_ann_1, dic_key = generate_public_key(priv_key, 
                                                         save=False, 
                                                         **option_weights)
            key_str_ann_1 = dic_key['public_key(str)']
            
            key_int_ann_2, key2 = generate_public_key(priv_key_flip, 
                                                      save=False, 
                                                      **option_weights)
            key_str_ann_2 = key2['public_key(str)']
            
            # Generate key with SHA256
            key_int_sha_1, key_str_sha_1 = sha256_key(priv_key_str)
            key_int_sha_2, key_str_sha_2 = sha256_key(priv_key_flip_str)
            
            # Compute hamming distance
            D1.append(hamming_distance(key_int_ann_1, key_int_ann_2))
            D2.append(hamming_distance(key_int_sha_1, key_int_sha_2))
        
        # Update dic
        D_dic['ann']['mean'].append(np.mean(D1))
        D_dic['sha256']['mean'].append(np.mean(D2))
        D_dic['ann']['var'].append(np.var(D1))
        D_dic['sha256']['var'].append(np.var(D2))
        
    # Plot results
    fig = plt.figure()
    fig.suptitle(f'Hamming distance between outputs, versus the number of bit flip in input')
    ax = fig.add_subplot(211)
    ax.set_xlabel('nb flips')
    ax.set_ylabel('$<D(output1, output2)>$')
    ax.plot(nb_flips, D_dic['ann']['mean'], label=f'ann $\sigma$={sigma}')
    ax.plot(nb_flips, D_dic['sha256']['mean'], label='sha256')
    ax.set_title('Average')
    ax.legend()
    ax1 = fig.add_subplot(212)
    ax1.set_xlabel('nb flips')
    ax1.set_ylabel('$Var(D)$')
    ax1.plot(nb_flips, D_dic['ann']['var'], label='ann')
    ax1.plot(nb_flips, D_dic['sha256']['var'], label='sha256')
    ax1.set_title('Variance')
    fig.tight_layout()
    
    # Save fig
    fig.savefig(results_dir+filename+'.png')
    fig.savefig(results_dir+filename+'.pdf') 

def main(option):
    
    print('==============================')
    print('         '+option)
    print('==============================')
    
    # The network
    n = 256 # Size of the keys
    nb_samples = 200 # Number of samples for random generation
    
    # Parameter of the weight distribution
    mu = -0.5 # Mean of the distribution 
    sigmas = np.linspace(0.01, 10, 1000) # Standard deviation
    
    if option == 'entropy':
        display_phase_transition(nb_samples, n, mu, sigmas)
    elif option == 'hamming':
        display_non_linearity(nb_samples, n, mu, sigmas, nb_flip = 1)
    elif option == 'best':
        exponents = np.arange(1, 10, 0.5)
        sigmas = np.power(10, exponents)
        find_best_ann(nb_samples, n, sigmas, mu)
    elif option == 'comparison':
        comparison_sha256_ann(nb_samples, n, mu=0, sigma=10)
    elif option == 'nl_comparison':
        non_linearity_comparison(nb_samples, n, mu=0, sigma=10)


if __name__ == '__main__':
    
    option = 'best'
    
    # Run main
    time_start = time.perf_counter()
    main(option)
    time_elapsed = (time.perf_counter() - time_start)
    print('All done, in '+str(time_elapsed)+'s')