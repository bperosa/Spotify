

import numpy as np
import gzip
import os
from collections import Counter
import tensorflow as tf
from time import perf_counter
#print(tf.__version__)

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
    
    return dict(y_counter), dict(idf_counter)


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
    for word in set(title):
        try:
            sparse_index.append([0,X_keys.index(word)])
            sparse_value.append(X_dict[word]/idf_counter[word])
        except:
            pass
    if not sparse_index:
        sparse_index = np.zeros((0, 2), dtype=np.int64)
#tf.SparseTensor(indices=np.empty((0, 2), dtype=np.int64), values=[], dense_shape=(10, 10))
    return tf.sparse.reorder(tf.SparseTensor(sparse_index, sparse_value, sparse_dense_shape))

def y_to_sparse(y_dict, y):
    y_keys = list(y_dict.keys())
    sparse_index = []
    sparse_value = []
    sparse_dense_shape = [1,len(y_dict.keys())]
    for word in set(y):
        try: 
            sparse_index.append([0,y_keys.index(word)])
            sparse_value.append(y_dict[word])
        except:
            pass
    if not sparse_index:
        sparse_index = np.zeros((0, 2), dtype=np.int64)

    return tf.sparse.reorder(tf.SparseTensor(sparse_index, sparse_value, sparse_dense_shape))

def pprint_sparse_tensor(st):
    s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
    for (index, value) in zip(st.indices, st.values):
        s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
    return s + "}>"

#if __name__ == '__main__':
    #Saving All
test_index = np.sort(gen_test_set())
    
    #gen_unique_values(test_index) #Can import for this from ./counters/idf_training.csv (X) and ./counters/y_training.csv (y)

#Import Data
min_ys = 25
with open('counters/idf_training.csv') as f:
    idf_counter = {k: int(v) for k,v in [line.strip().split(', ') for line in f]}

with open('counters/y_training.csv') as f:
    y_tmp = { k: int(v) for k,v in [line.strip().rsplit(', ', maxsplit=1) for line in f]}

y_counter = {}
y_counter['other'] = 0
for k, v in y_tmp.items():
    if k == 'other':
        y_counter['other'] += v
    elif v >= min_ys:
        y_counter[k] = v
    else:
        y_counter['other'] += v

    #testing stuff
#test_index = np.sort(gen_test_set(toy_set=True))
#y_counter, idf_counter = gen_unique_values(test_index, toy_set= True, export= False) 

y_levels = len(y_counter.keys())

#Model Architecture
inputs = tf.keras.layers.Input(shape= (len(idf_counter.keys()),), name='input') 
hidden1 = tf.keras.layers.Dense(units= int(len(idf_counter.keys())*(.25**1)) , activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units= int(len(idf_counter.keys())*(.25**2)), activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units= int(len(idf_counter.keys())*(.25**3)), activation="sigmoid", name= 'hidden3')(hidden2)
hidden4 = tf.keras.layers.Dense(units= int(len(idf_counter.keys())*(.25**4)), activation="sigmoid", name= 'hidden4')(hidden3)
outputs = tf.keras.layers.Dense(units= y_levels, activation = "sigmoid", name= 'output')(hidden4) 

model = tf.keras.Model(inputs = inputs, outputs = outputs)

model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(lr = 0.001))

#Epoch Loop
epochs = 1
test_loss_epochs = []
epoch_timers =[]

for ep in range(epochs):
    time_start = perf_counter()
#Training Loop (record)
    i = 0
    j = 0
    y_test =0
    X_test = 0
    for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
        X_dict = dict.fromkeys(idf_counter.copy(), 0) #Reset values to 0
        y_dict = dict.fromkeys(y_counter.copy(), 0) #Reset values to 0
        i += 1

        X = np.array([d['title']])
        title = [word.split(' ') for word in X][0]
        for word in title:
            try: X_dict[word] += 1
            except: pass
        Xsparse = title_to_sparse(X_dict, idf_counter, title)

        Y = np.array(d['category'])
        for word in Y:
            try: y_dict[word] += 1
            except: pass
        Ysparse = y_to_sparse(y_dict, Y)

        if i in test_index:
            j += 1
            if j == 1 and ep == 0:
                y_test = y_to_sparse(y_dict, Y)
                X_test = title_to_sparse(X_dict, idf_counter, title)
            else:
                y_patterns = [y_test, Ysparse]
                y_test = tf.sparse.concat(axis=0, sp_inputs = y_patterns)

                X_patterns = [X_test, Xsparse]
                X_test = tf.sparse.concat(axis=0, sp_inputs = X_patterns)
        else:
            try:
                model.fit(x=tf.sparse.to_dense(Xsparse), y= tf.sparse.to_dense(Ysparse), batch_size = 1)
            except:
                print(i)
                print(pprint_sparse_tensor(Xsparse))
                print(pprint_sparse_tensor(Ysparse))

        if i % 100 == 0:
            print(f'index {i}')
            print(f'X Tensor: {pprint_sparse_tensor(Xsparse)}')
            print(f'Y Tensor: {pprint_sparse_tensor(Ysparse)}')
            print('')
            # print(f'X: {title[0]}, {X_dict[title[0]]}') 
            # print(f'Y: {Y[0]}, {y_dict[Y[0]]}')

        if i > 1000: #change/remove
            break

    epoch_timers.append(perf_counter() - time_start)
    test_loss = model.evaluate(x= tf.sparse.to_dense(X_test), y= tf.sparse.to_dense(y_test))
    test_loss_epochs.append(test_loss)

print(f'X Tensor: {pprint_sparse_tensor(X_test)}')
print(f'Y Tensor: {pprint_sparse_tensor(y_test)}')

print(pprint_sparse_tensor(tf.sparse.reorder(X_test)))

X_test.indices
X_dense = tf.sparse.to_dense(X_test)
y_dense = tf.sparse.to_dense(y_test)
X_dense
y_dense
#model.fit(x=X_dense, y= y_dense, batch_size = 1)

test_loss = model.evaluate(x= tf.sparse.to_dense(X_test), y= tf.sparse.to_dense(y_test))

for ep in range(1):
    print(ep)