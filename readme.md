## CGP for gym environments
In this project I tried to solve gym environments through CGP. I mainly
focused on diversification of the population since this was the main problem 
I encountered in previous my experiences with CGP.
I tried to solve this using different niching techniques, 
upregulating terms and different computation budget settings.

### How to run
Install libraries requirements.txt and run experiments.py. (some numpy errors in any of the libraries might occur because of np.bool -> bool transition,
you gotta fix this by yourself) 
You can change the parameters of  experiments in the regression function of experiments.py.
Experiments are logged in their respective folder (e.g. you vary ncolumns param -> ncolumns folder
is created in logs folder) and the best solution is rendered and saved after each experiment repetition.

### HAL_CGP disclaimer
I used CGP implementation from HAL_CGP library. I had to modify it quite a bit though. All of my modifications are
denoted with '# xkaras38' comment. Incomplete list of functions I added or modified is:
- Distance calculation between 2 genomes
- All of the fitness sharing functionality 
- Separation of evolve function into evolve_init and evolve_continue, so I can insert my code 
  in between the generations
- Calculating number of active nodes in the genome + activating number of active functions
  (per function calculation e.g. how many times genome actively used saturated add) in the genome 
- Modification of to_numpy function -> This allowed me to vectorize all calculations, which is 
  a huge speedup (hal-cgp has some HIGHLY cumbersome way to do it) 

### Files
- experiments.py: Main file, where you can run experiments
- lunar_lander.py: Lunar lander environment wrapper for CGP
- functions.py: All functions used in CGP
- parse_experiments.py: Script for parsing, plotting and analysing of experiments
- readme.md: File that contains very sneaky recursion easter egg
- requirements.txt: Requirements for the project
- logs: Folder with all logs from experiments
- hal-cgp: Modified hal-cgp library
- plots_and_images: Folder with plots and images from experiments
- videos: Folder with videos from experiments

### Results
I managed to solve LunarLander-v2 environment with high success rate. Most notable modifications that worked for me were:
- Dynamic mutation rate/parent count (if we are stuck in local optima, we increase mutation rate/parent count,
  and vica versa)
- Bigger population size and lower generation count (search is more breath than depth)
- up-regulation of function count (number of active functions in the genome correlated with fitness)