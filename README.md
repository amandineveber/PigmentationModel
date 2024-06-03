# PigmentationModel
Simulation code for the paper "The impact of environmental fluctuations, sexual dimorphism, dominance reversal and plasticity on the pigmentation-related genetic and phenotypic variation in *D. melanogaster* populations – A modelling study."

Python code to simulate the stochastic population model introduced and used in the article. 
Inputs of the function model_dim2_2locus are the following:

**family=** dataframe with 7 columns, representing the entire population, with each row corresponding to an individual: 

    **Locus_1:** determines the version of the pigmentation gene in the individual (AA, Aa, or aa).
    
    **Locus_2:** sex of the individual (XX for females and XY for males).
    
    **Pigmentation:** pigmentation of the individual.
    
    **X:** first coordinate of the individual in the geographical span of the population.
    
    **Y:** second coordinate of the individual in the geographical span of the population.
    
    **Birth:** date of birth of the individual.
    
    **Death:** date of death of the individual, if the value is -1 then the individual is alive.

**C_delta_delta=** positive constant bounding the individual reproduction rate

**time=** end time of the simulation

**h=** time step

**delta_1=** positive number describing the interaction range for death by competition

**delta_2=** positive number describing the interaction range for the reproduction kernel

**alpha,beta=** the spatial locations of individuals take their values in [-alpha, alpha] x [-beta, beta]

**m=** diffusion coefficient (or variance) of the individual Brownian motion

**periode=** period of the temperature function

**lamb=** average total number of offspring of a female during a single reproduction event

**a=** coefficient regulating the adaptation of dark phenotypes

**b=** coefficient regulating the adaptation of light phenotypes

**c=** coefficient regulating the mortality rate

**constante_vieillesse=**

**nb_competition=** 
