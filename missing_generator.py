import numpy as np
import pandas as pd
import math
import random

def random_sampling(dataframe, no_of_samples):
    no, dim = dataframe.shape

    if no < no_of_samples:
        no_of_samples = no

    data_x = dataframe.values.astype(np.float32)
    sample_idx = sample_batch_index(no, no_of_samples)
    data_x_i = data_x[sample_idx, :]

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )
    return actual_dataframe

def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
  binary_random_matrix = 1 * (unif_random_matrix < p)
  return binary_random_matrix.astype('float32')


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx


def mcar_sampling(dataframe, miss_rate, no_of_samples):
    '''introduce miss_rate percentage of missing data in a dataset in completely randomly
    Args:
    - data: original data
    - missing_rate: percentage of data missing
    - no_of_samples: no of rows to be samples
    Returns:
    - miss_data_x: dataset with missing data
    '''
    if no_of_samples != None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values.astype(np.float32)

        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)

    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no_i, dim_i)
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan


    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:,0:],
        index=[i for i in range(miss_data_x.shape[0])],
        columns=dataframe.columns
        )

    return actual_dataframe, missing_dataframe

def mar_sampling(dataframe, miss_rate, no_of_samples):
    '''introduce miss_rate percentage of missing data in a dataset in randomly
    Args:
    - data: original data
    - missing_rate: percentage of data missing (50% should be sent as .5)
    - no_of_samples: no of rows to be samples
    Returns:
    - miss_data_x: dataset with missing data
    '''

    if no_of_samples != None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values.astype(np.float32)

        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)
    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )

    missing=0
    j_size = len(data_x_i)
    max_missing = j_size * len(data_x_i[0]) * miss_rate
    
    if dim_i < 5:
        raise ValueError("There should be more than five features")
    if miss_rate>.85:
        raise ValueError("Miss rate can not be more than 85 percent")

    quantile_low = miss_rate / 2
    quantile_high = 1 - miss_rate / 2

    for i in range(0, dim_i):
        np.random.seed(i)
        sc1 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i]])
        sc2 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1]])
        sc3 = np.random.choice([x for x in range(0,dim_i-1) if x not in [i,sc1,sc2]])
        df_1 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_high)]
        df_2 = actual_dataframe[actual_dataframe[dataframe.columns[sc1]] >= actual_dataframe[dataframe.columns[sc1]].quantile(quantile_low)]
        df_3 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_high)]
        df_4 = actual_dataframe[actual_dataframe[dataframe.columns[sc2]] >= actual_dataframe[dataframe.columns[sc2]].quantile(quantile_low)]
        df_5 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_high)]
        df_6 = actual_dataframe[actual_dataframe[dataframe.columns[sc3]] >= actual_dataframe[dataframe.columns[sc3]].quantile(quantile_low)]
        result_indexes = list(set(df_1.index)|set(df_2.index)|set(df_3.index)|set(df_4.index)|set(df_5.index)|set(df_6.index))
        random.shuffle(result_indexes)
        
        data_m_bin = binary_sampler(1, no_i, 1)
        column_limit = math.ceil(no_i*miss_rate)
        column_missing = 0
        
        for j in result_indexes:
            if missing<max_missing and column_missing<column_limit:
                data_m_bin[j] = 0
                column_missing+=1
                missing+=1
            
        if 'data_m' in vars():
            data_m = np.append(data_m, data_m_bin, 1)
        else:
            data_m = data_m_bin


    
    print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan


    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:,0:],
        index=[i for i in range(miss_data_x.shape[0])],
        columns=dataframe.columns
        )

    return actual_dataframe, missing_dataframe


def mnar_sampling(dataframe, miss_rate, no_of_samples):
    '''introduce miss_rate percentage of missing data in a dataset in randomly
    Args:
    - data: original data
    - missing_rate: percentage of data missing (50% should be sent as .5)
    - no_of_samples: no of rows to be samples
    Returns:
    - miss_data_x: dataset with missing data
    '''

    if no_of_samples != None:
        no, dim = dataframe.shape

        if no < no_of_samples:
            no_of_samples = no

        data_x = dataframe.values.astype(np.float32)

        sample_idx = sample_batch_index(no, no_of_samples)
        data_x_i = data_x[sample_idx, :]
    else:
        data_x_i = dataframe.values.astype(np.float32)

    no_i, dim_i = data_x_i.shape

    actual_dataframe = pd.DataFrame(
        data=data_x_i[0:,0:],
        index=[i for i in range(data_x_i.shape[0])],
        columns=dataframe.columns
        )

    missing=0
    j_size = len(data_x_i)
    max_missing = j_size * len(data_x_i[0]) * miss_rate
    maxReached = False

    high = True
    low = True

    if dim_i < 2:
        raise ValueError("There should be more than one feature")
    if miss_rate>.85:
        raise ValueError("Miss rate can not be more than 85 percent")

    quantile_low = miss_rate / 2
    quantile_high = 1 - (miss_rate / 2)

    column_limit = math.ceil(no_i*miss_rate)
    for i in range(0, dim_i):
        column_missing = 0
        percentile_high = actual_dataframe[dataframe.columns[i]].quantile(quantile_high)
        percentile_low = actual_dataframe[dataframe.columns[i]].quantile(quantile_low)
        data_m_bin = binary_sampler(1, no_i, 1)
        for j in range (0, no_i):
            if high and percentile_high <= data_x_i[j][i] and not maxReached and column_missing<column_limit:
                data_m_bin[j] = 0
                missing+=1
                column_missing+=1
                if missing >= max_missing:
                    maxReached = True
            elif low and percentile_low >= data_x_i[j][i] and not maxReached and column_missing<column_limit:
                data_m_bin[j] = 0
                missing+=1
                column_missing+=1
                if missing >= max_missing:
                    maxReached = True
            
        if 'data_m' in vars():
            data_m = np.append(data_m, data_m_bin, 1)
        else:
            data_m = data_m_bin

    
    print("max missing: "+str(max_missing)+":::   total removed:"+str(missing))
    miss_data_x = data_x_i.copy()
    miss_data_x[data_m == 0] = np.nan


    missing_dataframe = pd.DataFrame(
        data=miss_data_x[0:,0:],
        index=[i for i in range(miss_data_x.shape[0])],
        columns=dataframe.columns
        )

    return actual_dataframe, missing_dataframe