
from re import T
import numpy as np
import gzip
import os
from collections import Counter

import tensorflow as tf

#os.chdir('/Users/mballin2/Desktop')
#2684888 Records

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def gen_test_set(test_size = 0.2, toy_set = False, seed = 24785):
    np.random.seed(seed)
    if toy_set:
        return np.random.choice(np.arange(1,1000+1),size = int(test_size*1000), replace= False)
    else:
        return np.random.choice(np.arange(1,2684888+1),size = int(.2*2684888), replace= False)

def gen_unique_values(test_index, toy_set = False, export = True):
    #Generate idf_values and y_values
    y_counter = Counter()
    idf_counter = Counter()
    i = 0
    for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
        i += 1

        if i in test_index:
            continue

        X = np.array([d['title']])
        title = [word.split(' ') for word in X][0]
        idf_counter.update(set(title))
  
        Y = np.array(d['category'])
        y_counter.update(Y)

        if i % 10000 == 0:
            print(f'Iteration: {i}')

        if toy_set and (i > 1000): #change/remove
            break

    if export:
        with open('counters/y_training.csv', 'wt') as f:
            for k,v in y_counter.most_common():
                f.write(f'{k}, {v}\n')

        with open('counters/idf_training.csv', 'wt') as f:
            for k,v in idf_counter.most_common():
                f.write(f'{k}, {v}\n')
    
    return y_counter, idf_counter


# with open('counters/x_training.csv', 'wt') as f:
#     for k,v in X_counter.most_common():
#         f.write(f'{k}, {v}\n')

# with open('counters/y_counter.csv', 'wt') as f:
#     for k,v in y_counter.most_common():
#         f.write(f'{k}, {v}\n')

# with open('counters/idf_counter.csv', 'wt') as f:
#     for k,v in idf_counter.most_common():
#         f.write(f'{k}, {v}\n')

def title_to_sparse(X_dict,idf_counter, title):
    X_keys = list(X_dict.keys())
    sparse_index = []
    sparse_value = []
    sparse_dense_shape = [1,len(X_dict.keys())]
    for word in title:
        sparse_index.append([0,X_keys.index(word)])
        sparse_value.append(X_dict[word]/idf_counter[word])
    return tf.SparseTensor(sparse_index, sparse_value, sparse_dense_shape)

def y_to_sparse(y_dict, y):
    y_keys = list(y_dict.keys())
    sparse_index = []
    sparse_value = []
    sparse_dense_shape = [1,len(y_dict.keys())]
    for word in y:
        sparse_index.append([0,y_keys.index(word)])
        sparse_value.append(y_dict[word])
    return tf.SparseTensor(sparse_index, sparse_value, sparse_dense_shape)

def pprint_sparse_tensor(st):
    s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
    for (index, value) in zip(st.indices, st.values):
        s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
    return s + "}>"

if __name__ == '__main__':
    #Saving All
    #test_index = gen_test_set()
    #gen_unique_values(test_index)

    #testing stuff
    test_index = gen_test_set(toy_set=True)
    y_counter, idf_counter = gen_unique_values(test_index, toy_set= True, export= False) #Can import for this stop... eventually

    #Toy Training Loop
    i = 0
    j = 0
    for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
        X_dict = dict.fromkeys(idf_counter.copy(), 0) #Reset values to 0
        y_dict = dict.fromkeys(y_counter.copy(), 0) #Reset values to 0
        i += 1

        if i in test_index:
            j += 1
            continue

        X = np.array([d['title']])
        title = [word.split(' ') for word in X][0]
        for word in title:
            try: X_dict[word] += 1
            except: pass
        Xsparse = title_to_sparse(X_dict, idf_counter, title )

        Y = np.array(d['category'])
        for word in Y:
            try: y_dict[word] += 1
            except: pass
        Ysparse = y_to_sparse(y_dict, Y)

        if i % 100 == 0:
            print(f'index {i}')
            print(f'X Tensor: {pprint_sparse_tensor(Xsparse)}')
            print(f'Y Tensor: {pprint_sparse_tensor(Ysparse)}')
            print('')
            print(f'X: {title[0]}, {X_dict[title[0]]}') 
            print(f'Y: {Y[0]}, {y_dict[Y[0]]}')

        if i > 1000: #change/remove
            break

