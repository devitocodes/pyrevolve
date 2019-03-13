import pickle
import numpy as np


def dumps(data):
    if isinstance(data, np.ndarray):
        data = {'data': data.tobytes(), 'shape': data.shape,
                'dtype': data.dtype, 'creator': 'custom_pickle'}
    return pickle.dumps(data)


def loads(data):
    outdata = pickle.loads(data)
    if isinstance(outdata, dict) \
       and 'creator' in outdata  \
       and outdata['creator'] == 'custom_pickle':
        outdata = np.frombuffer(outdata['data'], dtype=outdata['dtype'])
        outdata = outdata.reshape(outdata['shape'])
    return outdata
