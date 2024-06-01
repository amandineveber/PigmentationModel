# -*- coding: utf-8 -*-
"""
Created on Thu May 16 02:23:26 2024

@author: lfreoa
"""

import numpy as np
import pandas

def sigmoid(x,lamb) :
    return(1/(1+np.exp(-lamb*x)))

def model_dim2_2locus(family, C_delta_delta, time, h, delta_1, delta_2, alpha, beta, m, periode, lamb, a, b, c, constante_vieillesse, nb_competition):
    # Initialize variables
    T = 0
    bar_alpha = alpha + 0.01 * abs(alpha)
    bar_beta = beta - 0.01 * abs(beta)
    NT = time / h
    p = 1
    total = 0
    vec_AA_XX = np.ones(int(time / h))
    vec_AA_XY = np.ones(int(time / h))
    vec_Aa_XX = np.ones(int(time / h))
    vec_Aa_XY = np.ones(int(time / h))
    vec_aa_XX = np.ones(int(time / h))
    vec_aa_XY = np.ones(int(time / h))
    vec_mean = np.ones(int(time / h))
    vec_sd = np.ones(int(time / h))
    
    # Simulation loop
    for k in range(1, int(NT), 1):
        # Extract living individuals
        I_living = family[family['Death'] == -1]
        I_living_AA = I_living[(I_living['locus_1'] == 'AA')]
        I_living_Aa = I_living[(I_living['locus_1'] == 'Aa')]
        I_living_aa = I_living[(I_living['locus_1'] == 'aa')]
        
        # Calculate population counts for different genotypes
        vec_AA_XX[k] = I_living_AA[I_living_AA['locus_2'] == 'XX'].shape[0]
        vec_AA_XY[k] = I_living_AA[I_living_AA['locus_2'] == 'XY'].shape[0]
        vec_Aa_XX[k] = I_living_Aa[I_living_Aa['locus_2'] == 'XX'].shape[0]
        vec_Aa_XY[k] = I_living_Aa[I_living_Aa['locus_2'] == 'XY'].shape[0]
        vec_aa_XX[k] = I_living_aa[I_living_aa['locus_2'] == 'XX'].shape[0]
        vec_aa_XY[k] = I_living_aa[I_living_aa['locus_2'] == 'XY'].shape[0]
        
        # Calculate mean and standard deviation of pigmentation for individuals with a certain genotype
        vec_mean[k] = np.mean(I_living[I_living['locus_2'] == 'XX']['Pigmentation'])
        vec_sd[k] = np.std(I_living[I_living['locus_2'] == 'XX']['Pigmentation'])
        
        n = I_living.shape[0]
        
        # Check for population extinction
        if family[family['Death'] == -1].shape[0] == 1:
            data_graph = pandas.DataFrame({'N_AA_XX': vec_AA_XX, 'N_AA_XY': vec_AA_XY, 'N_aa_XX': vec_aa_XX, 'N_aa_XY': vec_aa_XY, 'N_Aa_XX': vec_Aa_XX, 'N_Aa_XY': vec_Aa_XY, 'Pigmentation_moyenne': vec_mean, 'Pigmentation_sd': vec_sd})
            return data_graph
        
        # Update exponential clock
        if k * h >= T and p == 1:
            T_old = T
            T = T_old + np.random.exponential(1 / (C_delta_delta * n * (n + 1)), 1)
            total = total + 1
            p = 0
        
        # Population movements
        U_old = np.sqrt((k * h)) * np.random.randn(n)
        U = np.sqrt((k * (h + 1))) * np.random.randn(n)
        V = np.random.exponential(1 / (2 * k * (h + 1)), n)
        Y_1 = 0.5 * (-np.sqrt(2 * m) + np.sqrt((( -np.sqrt(2 * m)) ** 2) * V + (-np.sqrt(2 * m) * U) ** 2))
        Y_2 = 0.5 * (np.sqrt(2 * m) + np.sqrt(((np.sqrt(2 * m)) ** 2) * V + (np.sqrt(2 * m) * U) ** 2))
        B_old = U_old
        B = U
        S_1 = Y_1 + np.sqrt(2 * m) * U
        S_2 = Y_2 - np.sqrt(2 * m) * U
        family.loc[I_living.index, 'X'] = np.maximum(alpha, np.minimum(beta, I_living.X + np.sqrt(2 * m) * (B - B_old) + (I_living.X < bar_alpha) * np.maximum(0, S_1 - (I_living.X - alpha)) - (I_living.X > bar_beta) * np.maximum(0, S_2 + (I_living.X - beta))))
        
        U_old = np.sqrt((k * h)) * np.random.randn(n)
        U = np.sqrt((k * (h + 1))) * np.random.randn(n)
        V = np.random.exponential(1 / (2 * k * (h + 1)), n)
        Y_1 = 0.5 * (-np.sqrt(2 * m) + np.sqrt((( -np.sqrt(2 * m)) ** 2) * V + (-np.sqrt(2 * m) * U) ** 2))
        Y_2 = 0.5 * (np.sqrt(2 * m) + np.sqrt(((np.sqrt(2 * m)) ** 2) * V + (np.sqrt(2 * m) * U) ** 2))
        B_old = U_old
        B = U
        S_1 = Y_1 + np.sqrt(2 * m) * U
        S_2 = Y_2 - np.sqrt(2 * m) * U
        family.loc[I_living.index, 'Y'] = np.maximum(alpha,np.minimum(beta, I_living.Y + np.sqrt(2*m)*(B-B_old) + (I_living.Y <bar_alpha)*np.maximum(0,S_1-(I_living.Y-alpha))- (I_living.Y>bar_beta)*np.maximum(0,S_2+ (I_living.Y -beta))))
         
        # Birth and death events
        if k*h>=T and p==0 :
            I_living = family[family['Death'] == -1]
            N = I_living.shape[0]
            individu = I_living.sample()
            # Seasonal temperature variation
            A = 11
            temp = A*(np.sin((2*np.pi*k*h)/periode)+1)+10
            temp_max = 32
            temp_min =10
            claire = (a*sigmoid(-(temp-11),0.6)+b)*(temp<=21) +(a*sigmoid(-(temp-29),0.6))*(temp>21)
            sombre = a*sigmoid((temp-11),0.6)*(temp<=21)+(b*sigmoid((temp-29),0.6)+a)*(temp>21)
            
            # Fate of a male individual: no change or death 
            if (individu.locus_2 == 'XY').bool() : 
                morts_potentiels = I_living[((I_living['X'] <= individu.X.tolist()[0] + delta_1) & (I_living['X'] >= individu.X.tolist()[0] - delta_1) & (I_living['Y'] <= individu.Y.tolist()[0] + delta_1) & (I_living['Y'] >= individu.Y.tolist()[0] - delta_1))]
                I_delta1 = morts_potentiels.shape[0] # si dim=2   I_delta1=(1/(np.pi*delta_1*delta_1))*I_delta1
                r_death = nb_competition*(I_delta1-1)
                c_death= (individu.Pigmentation.tolist()[0]/10)*sombre + (1-individu.Pigmentation.tolist()[0]/10)*claire
                mu = c_death*(constante_vieillesse +r_death)*(temp_max-temp)/((N+1)*C_delta_delta) 
                theta_old = 0
                theta = np.random.uniform(0,1)
                if theta < mu : # Death of a male individual
                    family.loc[individu.index[0], 'Death'] = k*h
                    p = 1
                    theta = theta_old + mu 
                if theta> theta_old  : #no change
                        p = 1
            else : 
                    # Fate of a female individual: no change, death or reproduction
                    morts_potentiels = I_living[((I_living['X'] <= individu.X.tolist()[0] + delta_1) & (I_living['X'] >= individu.X.tolist()[0] - delta_1) & (I_living['Y'] <= individu.Y.tolist()[0] + delta_1) & (I_living['Y'] >= individu.Y.tolist()[0] - delta_1))]
                    parents_potentiels = I_living[(I_living['X'] <= individu.X.tolist()[0] + delta_2) & (I_living['X'] >= individu.X.tolist()[0] - delta_2) & (I_living['Y'] <= individu.Y.tolist()[0] + delta_2) & (I_living['Y'] >= individu.Y.tolist()[0] - delta_2) & (I_living['locus_2'] == 'XY')]

                    I_delta1 = morts_potentiels.shape[0] 

                    I_delta2_AA = parents_potentiels[parents_potentiels['locus_1'] == 'AA'].shape[0]  # si dim=2   I_delta2=(1/(np.pi*delta_2*delta_2))*I_delta2
                    I_delta2_Aa = parents_potentiels[parents_potentiels['locus_1'] == 'Aa'].shape[0]
                    I_delta2_aa = parents_potentiels[parents_potentiels['locus_1'] == 'aa'].shape[0]
                    r_death = nb_competition*(I_delta1-1)
                    c_death= (individu.Pigmentation.tolist()[0]/10)*sombre + (1-individu.Pigmentation.tolist()[0]/10)*claire
                    constante_naissance = c 
                    mu = c_death*(constante_vieillesse +r_death)*(temp_max-temp)/((N+1)*C_delta_delta) 
                    theta_old = 0
                    theta = np.random.uniform(0,1)
                    
                    if theta < mu : #Death
                        family.loc[ individu.index[0], 'Death'] = k*h
                        p = 1

                    theta_old = mu
                    # Sexual reproduction
                    
                    # A female individual reproduces with a male of genotype AA
                    if theta_old <= theta <= theta_old + constante_naissance*(sigmoid(I_delta2_AA,0.18) -0.5)*(temp-temp_min)/((N+1)*C_delta_delta)  :#AA
                        
                        # The probability vector of the children's genotypes as a function of the mother's genotype knowing that the father's genotype is AA
                        proba = [0.5,0.5,0,0,0,0]*(individu.locus_1.tolist()[0] == 'AA') + [0.25,0.25,0.25,0.25,0,0]*(individu.locus_1.tolist()[0] == 'Aa') + [0,0,0.5,0.5,0,0]*(individu.locus_1.tolist()[0] == 'aa')
                        
                        # Total number of children given by a Poisson law of prameter lambda. Here Lambda is chosen as the average number of children of a fruit fly.
                        n_birth = np.random.poisson(lamb)
                       
                        # The vector of the number of children according to their sex and genotype,this vector is the realization of a multinomial law with parameters n_birth and proba 
                        [n_AA_XX,n_AA_XY,n_Aa_XX,n_Aa_XY,n_aa_XX,n_aa_XY] = np.random.multinomial(n_birth,proba)
                        
                        # Integration of the children's genotype in the dataframe
                        locus_1 = np.repeat(['AA','Aa','aa'],[n_AA_XX+n_AA_XY,n_Aa_XX+n_Aa_XY,n_aa_XX+n_aa_XY])
                        locus_2_AA = np.repeat(['XX','XY'],[n_AA_XX,n_AA_XY])
                        locus_2_Aa = np.repeat(['XX','XY'],[n_Aa_XX,n_Aa_XY])
                        locus_2_aa = np.repeat(['XX','XY'],[n_aa_XX,n_aa_XY])
                        locus_2 = np.concatenate((locus_2_AA, locus_2_Aa,locus_2_aa))
                        
                        # Integration of children's pigmentation into the dataframe
                        pigmentation = np.ones(n_birth)
                       
                        # Males always have the darkest pigmentation possible, here 10
                        pigmentation[locus_2 == 'XY'] = 10
                       
                        # Females have pigmentation that depends on their genotype and the temperature at birth; it will be fixed throughout their life.
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'AA')] = 3.5*sigmoid(-(temp-26),1) + 6.5
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'Aa')] = 10*sigmoid(-(temp-24.8),0.8)
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'aa')] = 6.5*sigmoid(-(temp-20),0.6)
                        nouveaux_nes = pandas.DataFrame({'X': individu.X.tolist()[0]*np.ones(n_birth) , 'Y': individu.Y.tolist()[0]*np.ones(n_birth), 'locus_1': locus_1, 'locus_2': locus_2 , 'Pigmentation': pigmentation, 'Birth': np.zeros(n_birth) + k*h, 'Death': np.zeros(n_birth) - 1})                     
                        
                        # Updating the family dataframe
                        family = pandas.concat((family,nouveaux_nes), axis = 0, ignore_index = True)
                        p = 1

                    theta_old= theta_old + constante_naissance*(sigmoid(I_delta2_AA,0.18) -0.5)*(temp-temp_min)/((N+1)*C_delta_delta)
                    
                    # A female individual reproduces with a male of genotype Aa
                    if theta_old <= theta <= theta_old + constante_naissance*(sigmoid(I_delta2_Aa,0.18) -0.5)*(temp-temp_min)/((N+1)*C_delta_delta)  : #Aa
                        
                        # The probability vector of the children's genotypes as a function of the mother's genotype knowing that the father's genotype is Aa
                        proba = [0.25,0.25,0.25,0.25,0,0]*(individu.locus_1.tolist()[0] == 'AA') + [0.125,0.125,0.25,0.25,0.125,0.125]*(individu.locus_1.tolist()[0] == 'Aa') + [0,0,0.25,0.25,0.25,0.25]*(individu.locus_1.tolist()[0] == 'aa')
                        
                        # Total number of children given by a Poisson law of prameter lambda. Here Lambda is chosen as the average number of children of a fruit fly.
                        n_birth = np.random.poisson(lamb)
                        
                        # The vector of the number of children according to their sex and genotype,this vector is the realization of a multinomial law with parameters n_birth and proba
                        [n_AA_XX,n_AA_XY,n_Aa_XX,n_Aa_XY,n_aa_XX,n_aa_XY] = np.random.multinomial(n_birth,proba)
                        
                        # Integration of the children's genotype in the dataframe
                        locus_1 = np.repeat(['AA','Aa','aa'],[n_AA_XX+n_AA_XY,n_Aa_XX+n_Aa_XY,n_aa_XX+n_aa_XY])
                        locus_2_AA = np.repeat(['XX','XY'],[n_AA_XX,n_AA_XY])
                        locus_2_Aa = np.repeat(['XX','XY'],[n_Aa_XX,n_Aa_XY])
                        locus_2_aa = np.repeat(['XX','XY'],[n_aa_XX,n_aa_XY])
                        locus_2 = np.concatenate((locus_2_AA, locus_2_Aa,locus_2_aa))
                        
                        # Integration of children's pigmentation into the dataframe
                        pigmentation = np.ones(n_birth)
                        
                        # Males always have the darkest pigmentation possible, here 10
                        pigmentation[locus_2 == 'XY'] = 10
                        
                        # Females have pigmentation that depends on their genotype and the temperature at birth; it will be fixed throughout their life.
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'AA')] = 3.5*sigmoid(-(temp-26),1) + 6.5
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'Aa')] = 10*sigmoid(-(temp-24.8),0.8)
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'aa')] = 6.5*sigmoid(-(temp-20),0.6)
                        nouveaux_nes = pandas.DataFrame({'X': individu.X.tolist()[0]*np.ones(n_birth) , 'Y': individu.Y.tolist()[0]*np.ones(n_birth), 'locus_1': locus_1, 'locus_2': locus_2 , 'Pigmentation': pigmentation, 'Birth': np.zeros(n_birth) + k*h, 'Death': np.zeros(n_birth) - 1})                     
                        # Updating the family dataframe
                        family = pandas.concat((family,nouveaux_nes), axis = 0, ignore_index = True)
                        p = 1

                    theta_old= theta_old + constante_naissance*(sigmoid(I_delta2_Aa,0.18) -0.5)*(temp-temp_min)/((N+1)*C_delta_delta)
                    
                    # A female individual reproduces with a male of genotype aa
                    if theta_old <= theta <= theta_old + constante_naissance*(sigmoid(I_delta2_aa,0.18) -0.5)*(temp-temp_min)/((N+1)*C_delta_delta)  : #aa
                        
                        # The probability vector of the children's genotypes as a function of the mother's genotype knowing that the father's genotype is aa
                        proba = [0,0,0.5,0.5,0,0]*(individu.locus_1.tolist()[0] == 'AA')+ [0,0,0.25,0.25,0.25,0.25]*(individu.locus_1.tolist()[0] == 'Aa')+[0,0,0,0,0.5,0.5]*(individu.locus_1.tolist()[0] == 'aa')
                        
                        # Total number of children given by a Poisson law of prameter lambda. Here Lambda is chosen as the average number of children of a fruit fly.
                        n_birth = np.random.poisson(lamb)
                        
                        # The vector of the number of children according to their sex and genotype,this vector is the realization of a multinomial law with parameters n_birth and proba
                        [n_AA_XX,n_AA_XY,n_Aa_XX,n_Aa_XY,n_aa_XX,n_aa_XY] = np.random.multinomial(n_birth,proba)
                        
                        # Integration of the children's genotype in the dataframe
                        locus_1 = np.repeat(['AA','Aa','aa'],[n_AA_XX+n_AA_XY,n_Aa_XX+n_Aa_XY,n_aa_XX+n_aa_XY])
                        locus_2_AA = np.repeat(['XX','XY'],[n_AA_XX,n_AA_XY])
                        locus_2_Aa = np.repeat(['XX','XY'],[n_Aa_XX,n_Aa_XY])
                        locus_2_aa = np.repeat(['XX','XY'],[n_aa_XX,n_aa_XY])
                        locus_2 = np.concatenate((locus_2_AA, locus_2_Aa,locus_2_aa))
                        
                        # Integration of children's pigmentation into the dataframe
                        pigmentation = np.ones(n_birth)
                        
                        # Males always have the darkest pigmentation possible, here 10
                        pigmentation[locus_2 == 'XY'] = 10
                        
                        # Females have pigmentation that depends on their genotype and the temperature at birth; it will be fixed throughout their life.
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'AA')] = 3.5*sigmoid(-(temp-26),1) + 6.5
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'Aa')] = 10*sigmoid(-(temp-24.8),0.8)
                        pigmentation[(locus_2 == 'XX') & (locus_1 == 'aa')] = 6.5*sigmoid(-(temp-20),0.6)
                        nouveaux_nes = pandas.DataFrame({'X': individu.X.tolist()[0]*np.ones(n_birth) , 'Y': individu.Y.tolist()[0]*np.ones(n_birth), 'locus_1': locus_1, 'locus_2': locus_2 , 'Pigmentation': pigmentation, 'Birth': np.zeros(n_birth) + k*h, 'Death': np.zeros(n_birth) - 1})                     
                        
                        # Updating the family dataframe
                        family = pandas.concat((family,nouveaux_nes), axis = 0, ignore_index = True)
                        p = 1

                    theta_old = theta_old + constante_naissance*(sigmoid(I_delta2_aa,0.18) -0.5)*(temp-temp_min)/((N+1)*C_delta_delta)

                    if theta> theta_old  : #no change
                        p = 1
    # data_graph is a dataframe which collected the necessary data for the graphs at each time iteration.
    # Although this increases the calculation time, it is always faster to do it at the same time as the simulation than to do the simulation then generate the data for the graphs.
    data_graph = pandas.DataFrame({'N_AA_XX': vec_AA_XX, 'N_AA_XY': vec_AA_XY, 'N_aa_XX': vec_aa_XX, 'N_aa_XY': vec_aa_XY , 'N_Aa_XX': vec_Aa_XX, 'N_Aa_XY': vec_Aa_XY, 'Pigmentation_moyenne': vec_mean, 'Pigmentation_sd': vec_sd})
    return family, data_graph 