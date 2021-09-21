import numpy as np
import gzip
import json
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

def parse_subset(path, indices):
    #g = gzip.open(path, 'rb')
    max_index = np.max(indices)
    with gzip.open(path, 'rb') as g:
        for i, l in enumerate(g):
            if i > max_index: break
            if i in indices: 
                yield {k: v for k, v in json.loads(l).items() if k in ['title', 'category'] }

def gen_test_set(test_size = 0.2, toy_set = False, seed = 24785):
    np.random.seed(seed)
    if toy_set:
        return np.random.choice(np.arange(0,1000),size = int(test_size*1000), replace= False)
    else:
        return np.random.choice(np.arange(0,2684888),size = int(test_size*2684888), replace= False)

def gen_training_sample(n_records, test_index, sample_size = 0.05):
    return np.sort(np.random.choice( np.delete(np.arange(1, n_records), test_index), int(sample_size*2684888), replace = False ))

def gen_unique_values(test_index, toy_set = False, export = True):
    #Generate idf_values and y_values
    y_counter = Counter()
    idf_counter = Counter()

    for j, d in enumerate(parse_subset('meta_Clothing_Shoes_and_Jewelry.json.gz', test_index)):
        if j % 10000 == 0:
            print(f'Making Inverse doc Frequency Dictionary, Iteration: {i}')

        X = [word.lower() for word in d['title'].split(' ')]
        # title = [word.split(' ') for word in X][0]
        idf_counter.update(set(X))
  
        Y = np.array(d['category'])
        y_counter.update(Y)

        if toy_set and (i > 1000): #change/remove
            break

    if export:
        with open('counters/y_training_10percent.csv', 'wt') as f:
            for k,v in y_counter.most_common():
                f.write(f'{k}, {v}\n')

        with open('counters/idf_training_10percent.csv', 'wt') as f:
            for k,v in idf_counter.most_common():
                f.write(f'{k}, {v}\n')
    
    return dict(y_counter), dict(idf_counter)

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

def get_test_set(test_index, idf_counter, y_index, X_index, import_data = False, save_data = True  ):
    """import_data = True if you have the following files:
        ./testdata/x_test_dense_shape.csv
        ./testdata/x_test_indices.csv
        ./testdata/x_test_values.csv
        ./testdata/y_test.csv
    """
    first_record = 0
    y_test = 0
    X_test = 0

    # dictionary = 0 #Reworked and removed this portion
    # X_split = 0
    # y_split = 0
    # stacking = 0
    if import_data:
        sparse_index = np.genfromtxt('testdata/x_test_indices.csv',  delimiter = ',')
        sparse_value = np.genfromtxt('testdata/x_test_values.csv',  delimiter = ',')
        sparse_dense_shape = np.genfromtxt('testdata/x_test_dense_shape.csv',  delimiter = ',')

        X_test = tf.sparse.reorder(tf.SparseTensor(sparse_index, sparse_value, sparse_dense_shape))
        y_test = np.genfromtxt('testdata/y_test.csv', delimiter = ',') # Don't use... too much memory to load (probably)

    else:
        all_x_indices = []
        all_x_values = []
        all_y_indices = []

        x_indices = []
        x_values = []
        y_indices = []
        for j, d in enumerate(parse_subset('meta_Clothing_Shoes_and_Jewelry.json.gz', test_index)):

            # X_split_start = perf_counter()
            title = [word.lower() for word in d['title'].split(' ') if word in X_unique]
            for word in set(title):
                try: 
                    x_indices.append([j,X_index[word]])
                    x_values.append(title.count(word)/ idf_counter[word])
                except: pass
            #Xsparse = title_to_sparse_rework(x_indices, x_values, idf_levels)
            # if len(x_indices) > 0:
            #     all_x_indices.append([idx for idx in x_indices])
            #     all_x_values.append(x_values)
            # X_split += perf_counter() - X_split_start

            # y_split_start = perf_counter()
            Y = np.array(d['category'])
            for cat in set(Y):
                try: y_indices.append([j, y_index[cat]])
                except: pass
            
            if j % 1000 == 0 : print(test_index[j])#, X_test.shape, y_test.shape)

        y_test = np.zeros((len(test_index), y_levels))

        for i, j in np.array(y_indices):
            y_test[i,j] = 1

        X_test = tf.sparse.reorder(tf.SparseTensor(x_indices,x_values, [len(test_index), idf_levels]))

    if save_data:
        np.savetxt('testdata/x_test_indices.csv', np.array(X_test.indices), delimiter = ',')
        np.savetxt('testdata/x_test_values.csv', np.array(X_test.values), delimiter = ',')
        np.savetxt('testdata/x_test_dense_shape.csv', np.array(X_test.dense_shape), delimiter = ',')

        np.savetxt('testdata/y_test.csv', y_test, delimiter = ',')

    return X_test, y_test

def fit_model(training_sample):
    for j, d in enumerate(parse_subset('meta_Clothing_Shoes_and_Jewelry.json.gz', training_sample)):

        if j % 10000 == 0:
            print(training_sample[j])
            print(perf_counter() - time_start, ' seconds in epoch')

        x_indices = []
        x_values = []
        
        y_indices = []

        # x_split_start = perf_counter()
        title = [word.lower() for word in d['title'].split(' ') if word in X_unique]
        for word in set(title):
            try: 
                x_indices.append(X_index[word])
                x_values.append(title.count(word)/ idf_counter[word])
            except: pass
        Xsparse = title_to_sparse_rework(x_indices, x_values, idf_levels)
        # X_split += perf_counter() - x_split_start

        # y_split_start = perf_counter()
        Y = np.array(d['category'])
        for cat in set(Y):
            try: y_indices.append(y_index[cat])
            except: pass
        Ysparse = y_to_numpy_rework(y_indices, y_levels)
        # y_split += perf_counter() - y_split_start

        # fitting_start = perf_counter()
        history = model.fit(x=Xsparse, y= Ysparse, batch_size = 1, epochs = 1, verbose = 0)
        # fitting += perf_counter() - fitting_start

    return history
    #Timing breakdown for 10k records:
    #I made some changes to the parse function that should speed this up, but the proportions should be about the same
        # ~305 seconds
        # >>> X_split
        # 17.50673349999873
        # >>> y_split
        # 0.41373170000035486
        # >>> fitting
        # 278.9441726000015
if __name__ == '__main__':
    print(tf.config.list_physical_devices())

    test_index = np.sort(gen_test_set(.1))

    #Import Data
    min_ys = 1000
    min_x = 25

    import_data = True

    if import_data:
        with open('counters/idf_training_10percent.csv') as f:
            idf_tmp = {k: int(v) for k,v in [line.strip().rsplit(', ', maxsplit=1) for line in f]}
        with open('counters/y_training_10percent.csv') as f:
            y_tmp = { k: int(v) for k,v in [line.strip().rsplit(', ', maxsplit=1) for line in f]}
    else:
        y_tmp, idf_tmp = gen_unique_values(test_index) 

    idf_counter = {}
    for k, v in idf_tmp.items():
        if v >= min_x:
            idf_counter[k] = v

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

    y_levels = len(y_counter.keys())
    idf_levels = len(idf_counter.keys())

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
    X_test, y_test = get_test_set(test_index, idf_counter, y_index, X_index, import_data = False, save_data = False  )
    print('Test Set built/loaded')

    #Epoch Loop
    epochs = 15
    test_loss_epochs = []
    epoch_timers =[]
    train_loss_epochs = []

    # X_split = 0
    # y_split = 0 
    # fitting = 0
    for ep in range(epochs):
        print(f'epoch: {ep}')
        time_start = perf_counter()
    
        training_sample = gen_training_sample(2684888, test_index, 0.05)
        
        #Training Loop
        history = fit_model(training_sample)

        epoch_timers.append(perf_counter() - time_start)
        test_loss = model.evaluate(x= X_test, y= y_test)
        test_loss_epochs.append(test_loss)
        train_loss_epochs.append(history.history['loss'])
        print(f'END of epoch: {ep}')
        print(f'test_loss: {test_loss_epochs}')
        print(f'train_loss: {train_loss_epochs}')

        model.save('./models/initial_model')

        

        df = pd.DataFrame(dict(test_loss = test_loss, test_loss_epochs = test_loss_epochs, train_loss_epochs = train_loss_epochs, epoch_timers=epoch_timers ))
        df.to_csv('training_stats.csv', index = False)


    #Evaluating Stuff:
    #######################################
    test_index = np.sort(gen_test_set(.1))

    #Import Data
    min_ys = 1000
    min_x = 25

    import_data = True

    if import_data:
        with open('counters/idf_training_10percent.csv') as f:
            idf_tmp = {k: int(v) for k,v in [line.strip().rsplit(', ', maxsplit=1) for line in f]}
        with open('counters/y_training_10percent.csv') as f:
            y_tmp = { k: int(v) for k,v in [line.strip().rsplit(', ', maxsplit=1) for line in f]}
    else:
        y_tmp, idf_tmp = gen_unique_values(test_index) 

    idf_counter = {}
    for k, v in idf_tmp.items():
        if v >= min_x:
            idf_counter[k] = v

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

    y_levels = len(y_counter.keys())
    idf_levels = len(idf_counter.keys())

    X_unique = set(idf_counter.keys())

    print('generating test set')
    #Gen Test Data
    X_test, y_test = get_test_set(test_index, idf_counter, y_index, X_index, import_data = False, save_data = False  )
    print('Test Set built/loaded')

    model2 = tf.keras.models.load_model('./models/initial_model')
    model2.evaluate(X_test, y_test)

    preds = model2.predict(X_test)

    #Making Accuracy Metric
    accuracy = []
    for j in range(y_test.shape[0]):
        predicted_labels = [list(y_counter.keys())[i] for i in np.argpartition(preds[j,:],-5)[-5:]]
        actual_labels = [list(y_counter.keys())[i] for i in np.nonzero(y_test[j,:])[0]]

        accuracy.append(np.sum([1 for i in predicted_labels if i in actual_labels])/ min(len(actual_labels),5))

    np.mean(np.array(accuracy))

    y_test.shape[1]
    X_test.shape

    model2.summary()
    
    
    df = pd.read_csv('training_stats.csv')

    train_loss = [float(el.replace('[', '', el.count('[')).replace(']', '', el.count(']'))) for el in df['train_loss_epochs']]
    
    df['train_loss_epochs'] = train_loss

    df
    import matplotlib.pyplot as plt

    plt.plot(np.array(list(range(15)))+1,  df['train_loss_epochs'], label = 'Training Loss')
    plt.plot(np.array(list(range(15)))+1,  df['test_loss_epochs'], label = 'Test Loss')
    plt.legend()
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

    df.tail(1)


    #Variable Importance
    vi_index = np.sort(gen_test_set(test_size = 0.001, toy_set = False, seed = 24785))
    vi_index.shape
    X_test_vi, y_test_vi = get_test_set(vi_index, idf_counter, y_index, X_index, import_data = False, save_data = False  )

    loss_before = 0.024762

    x_indices_array = np.array(X_test_vi.indices)
    x_values_array = np.array(X_test_vi.values)

    x_indices_importance = x_indices_array.copy()
    #x_values_importance = x_values_array.copy()

    for i in range(10):
        print(x_indices_array[i])
        print(x_values_array[i])

    importance = []
    column_dictionary = {}
    for i in range(X_test_vi.shape[1]):
        column_dictionary[str(i)] = []
        x_indices_importance = x_indices_array.copy()

        for j in range(x_indices_array.shape[0]):
            if x_indices_array[j][1] == i:
                column_dictionary[str(i)].append(j)

        new_index = np.random.choice(np.arange(0, X_test_vi.shape[0]), size = len(column_dictionary['0']) , replace=False)

        for k in range(new_index.shape[0]):
            x_indices_importance[column_dictionary[str(i)][k]][0] = new_index[k]

        X_test_importance = tf.sparse.reorder(tf.SparseTensor(x_indices_importance, x_values_array, X_test_vi.dense_shape))
        
        loss_new = model2.evaluate(X_test_importance, y_test_vi)

        importance.append(loss_before - loss_new)

        if i % 1000:
            print(i)
    #~30,000 minutes to build variable importance.



    column_dictionary['0'][0]

    new_col = np.zeros(X_test.shape[0])
    new_col[np.random.choice(np.arange(0, X_test.shape[0]), size = len(column_dictionary['0']) , replace=False)] = x_values_array[np.array(column_dictionary['0'])]
    new_col

    new_index = np.random.choice(np.arange(0, X_test.shape[0]), size = len(column_dictionary['0']) , replace=False)


    x_indices_array.shape
    x_values_array.shape
            
    

    x_indices_array[0]
    x_values_array[0]

    #Scramble row indices for each column index
    importance = list()
    for ind in [1,2]:
        #3.1
        cte_X_cp = np.copy(cte_X)
        variable = np.random.permutation(np.copy(cte_X[:,ind]))
        #note: np.random.pemutation already makes a copy, but including
        #np.copy for explicitness
        cte_X_cp[:,ind] = variable
        #3.2
        yhat = model.predict(cte_X_cp)
        performance_after = np.corrcoef(y,yhat)[0,1]
        #3.3
        importance.append(performance_before - performance_after)
    
    importance



