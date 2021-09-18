

import numpy as np
import gzip
import os
from collections import Counter
import tensorflow as tf
from time import perf_counter
import pandas as pd
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

def title_to_sparse_rework(indices, values, idf_levels = 31412):
    sparse_index = [[0, index] for index in indices]
    sparse_value = values
    sparse_dense_shape = [1, idf_levels]

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

def y_to_numpy(y_dict, y):
    y_keys = list(y_dict.keys())
    index = []
    value = []
    array = np.zeros((1,len(y_dict.keys())))
    for word in set(y):
        try: 
            index.append(y_keys.index(word))
            value.append(y_dict[word])
        except:
            pass
    if not index:
        array[index] = np.array(value)
        
    return array

def y_to_numpy_rework(indices, y_levels = 718 ):
    array = np.zeros((1,y_levels))
    indices_array = np.array(indices)
    if indices_array.size > 0:
        array[0, indices_array] = 1
    return array


def pprint_sparse_tensor(st):
    s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
    for (index, value) in zip(st.indices, st.values):
        s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
    return s + "}>"

if __name__ == '__main__':
    print(tf.config.list_physical_devices())
    #Saving All
    test_index = np.sort(gen_test_set())
    #test_index
        
    #gen_unique_values(test_index) #Can import for this from ./counters/idf_training.csv (X) and ./counters/y_training.csv (y)

    #Import Data
    min_ys = 1000
    min_x = 25
    with open('counters/idf_training.csv') as f:
        idf_tmp = {k: int(v) for k,v in [line.strip().split(', ') for line in f]}

    idf_counter = {}
    for k, v in idf_tmp.items():
        if v >= min_x:
            idf_counter[k] = v

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

    y_index = {k: i for i , (k,v) in enumerate(y_counter.items())}
    X_index = {k: i for i , (k,v) in enumerate(idf_counter.items())}
        #testing stuff
    #test_index = np.sort(gen_test_set(toy_set=True))
    #y_counter, idf_counter = gen_unique_values(test_index, toy_set= True, export= False) 

    y_levels = len(y_counter.keys())
    idf_levels = len(idf_counter.keys())

    idf_array = np.array(list(idf_counter.values()))
    X_unique = set(idf_counter.keys())

    #Model Architecture
    #inputs = tf.keras.layers.Input(shape= (idf_levels,), name='input') 
    inputs = tf.keras.layers.Input(shape= (idf_levels,), name='input', sparse = True) 
    hidden1 = tf.keras.layers.Dense(units= int(idf_levels*(.25**1)) , activation="sigmoid", name = 'hidden1')(inputs)
    hidden2 = tf.keras.layers.Dense(units= int(idf_levels*(.25**1)) , activation="sigmoid", name = 'hidden2')(hidden1)
    hidden3 = tf.keras.layers.Dense(units= int(idf_levels*(.25**2)), activation="sigmoid", name= 'hidden3')(hidden2)
    hidden4 = tf.keras.layers.Dense(units= int(idf_levels*(.25**3)), activation="sigmoid", name= 'hidden4')(hidden3)
    # hidden5 = tf.keras.layers.Dense(units= int(idf_levels*(.25**4)), activation="sigmoid", name= 'hidden5')(hidden4)
    outputs = tf.keras.layers.Dense(units= y_levels, activation = "sigmoid", name= 'output')(hidden4) 

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))


    print('generating test set')
    #Gen Test Data
    i = 0
    j = 0
    first_record = 0
    y_test = 0
    X_test = 0

    # dictionary = 0
    X_split = 0
    y_split = 0
    stacking = 0
    
    for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
        i += 1 
        if not i in test_index:
            continue

        x_indices = []
        x_values = []
        
        y_indices = []

        j += 1
        
        # X_split_start = perf_counter()
        X = np.array([d['title']])
        title = [word for word in X[0].split(' ') if word in X_unique]
        for word in set(title):
            try: 
                x_indices.append(X_index[word])
                x_values.append(title.count(word)/ idf_counter[word])
            except: pass
        Xsparse = title_to_sparse_rework(x_indices, x_values, idf_levels)
        # X_split += perf_counter() - X_split_start

        # y_split_start = perf_counter()
        Y = np.array(d['category'])
        for cat in set(Y):
            try: y_indices.append(y_index[cat])
            except: pass
        Ysparse = y_to_numpy_rework(y_indices, y_levels)
        # y_split += perf_counter() - y_split_start

        # stacking_start = perf_counter()
        if first_record == 0:
            first_record += 1
            y_test = y_to_numpy_rework(y_indices, y_levels)
            X_test = title_to_sparse_rework(x_indices, x_values, idf_levels)
        else:
            y_test = np.vstack((y_test, Ysparse))

            X_patterns = [X_test, Xsparse]
            X_test = tf.sparse.concat(axis=0, sp_inputs = X_patterns)
        # stacking += perf_counter() - stacking_start
        
        if j % 1000 == 0 : print(i, X_test.shape, y_test.shape)
        if i >= 40000: break
    print('Test Set built')


    #v1 i = 40000
    # >>> dictionary
    # 50.55356387700931
    # >>> X_split
    # 24.712150256012137
    # >>> y_split
    # 1.166882449997047
    # >>> stacking
    # 53.31768974997976

    #v2 i = 40000
    # >>> X_split
    # 5.0026881949979725
    # >>> y_split
    # 0.4179564459254834
    # >>> stacking
    # 47.28284420797172

    #Epoch Loop
    epochs = 10
    test_loss_epochs = []
    epoch_timers =[]
    train_loss_epochs = []

    for ep in range(epochs):
        print(f'epoch: {ep}')
        time_start = perf_counter()
    #Training Loop (record)
        i = 0
        j = 0

        for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
            # X_dict = dict.fromkeys(idf_counter.copy(), 0) #Reset values to 0
            # y_dict = dict.fromkeys(y_counter.copy(), 0) #Reset values to 0
            i += 1
            if i in test_index:
                continue

            x_indices = []
            x_values = []
            
            y_indices = []
            X = np.array([d['title']])
            title = [word for word in X[0].split(' ') if word in X_unique]
            for word in set(title):
                try: 
                    x_indices.append(X_index[word])
                    x_values.append(title.count(word)/ idf_counter[word])
                except: pass
            Xsparse = title_to_sparse_rework(x_indices, x_values, idf_levels)

            Y = np.array(d['category'])
            for cat in set(Y):
                try: y_indices.append(y_index[cat])
                except: pass
            Ysparse = y_to_numpy_rework(y_indices, y_levels)

            history = model.fit(x=Xsparse, y= Ysparse, batch_size = 1, epochs = 1, verbose = 0)

            if i % 1000 == 0: #change/remove
                print(i)
                print(perf_counter() - time_start, ' seconds in epoch')
            if i % 10000 == 0:
                break
                

        epoch_timers.append(perf_counter() - time_start)
        test_loss = model.evaluate(x= X_test, y= y_test)
        test_loss_epochs.append(test_loss)
        train_loss_epochs.append(history.history['loss'])
        print(f'END of dpoch: {ep}')
        print(f'test_loss: {test_loss_epochs}')
        print(f'train_loss: {train_loss_epochs}')

        model.save('./models/initial_model')

    #mod2 = tf.keras.models.load_model('./models/initial_model')

        df = pd.DataFrame(dict(test_loss = test_loss, test_loss_epochs = test_loss_epochs, train_loss_epochs = train_loss_epochs, epoch_timers=epoch_timers ))

        df.to_csv('training_stats.csv', index = False)
