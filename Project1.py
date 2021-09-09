# Group Project 1: Multi-Output Neural Network on Large Data
# Jessica Hamilton
# James McHale
# Andrew Paulonis
# Bailey Perosa


# import statements
import tensorflow as tf
import numpy as np
import gzip
import os

# code given (setup)
os.chdir('OneDrive/Desktop/BZAN 554/Project 1') # change directory

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

i = 0
df = {}
for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    X = np.array(d['title'])
    print('X (title):\n')
    print(X)
    Y = np.array(d['category'])
    print('\nY (category):\n')
    print(Y)
    if i == 10: #change/remove
        break
        
        
# model

#PROJECT HELP
# have nested loop: outside = epochs and inside = instances; set batch size and epoch to 1; when it hits model.fit it will update the parameters
#for i in range(10): # epochs
    #for i in range(len(x)): (X will not be loaded -> this will need to be hard coded) # instances
        # load new instance
        #model.fit(new instance, epochs=?, batch_size=1) #train model
