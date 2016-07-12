import os
import sys
import numpy as np
import pandas as pd
from glob import glob


def load_lsa(fname):
    """
    Load a single lsa file as a numpy record array. Columns are: t,x,y,p for the time, (x,y) coordinates,
    and polarity (1 for pressure increase, 0 for pressure decrease).
    """
    with open(fname, 'rb') as f:
        dat = f.read()

    # Don't copy or try to unpack yet
    a = np.frombuffer(dat, dtype=np.uint8).reshape(-1, 5)

    # Remove timesync events
    timesync = (a[:, 0] & 0x08) > 0
    a = a[~timesync]

    # Remove intensity events
    intensity = (a[:, 0] & 0x20) > 0
    a = a[~intensity]

    # timestamp from part of the first byte and bytes 2 and 3
    t = (((((a[:, 0] & 0x3).astype(np.uint32) << 7) | a[:, 1]) << 7) | a[:, 2]).cumsum(dtype=np.uint32)

    # x,y coords are in [0,63] with type uint8
    x = a[:, 3]
    y = a[:, 4]

    # polarity from part of the first byte
    p = ((a[:, 0] & 0x40) > 0).view(np.uint8)

    return np.core.records.fromarrays([t, x, y, p],
                                      names='t,x,y,p',
                                      formats='u4,u1,u1,u1')


def load_tactile(data_dir):
    """
    Load a directory of .lsa files into a dataframe indexed by:
     template: the user who's signature is written
     query: the user who actually writes the signature
     session: the session number

    Genuine samples are such that the template matches the query, impostor samples are the others.
    """
    dfs = []
    fnames = sorted(glob(data_dir + '*.lsa'))
    for fname in fnames:
        query, template, session_ext = os.path.split(fname)[1].split('_')
        session = session_ext[:-4]

        print('Loading:', fname)
        sample = load_lsa(fname)

        sample['t'] -= sample['t'][0]

        dfs.append(pd.DataFrame(sample, index=pd.MultiIndex.from_tuples([(template, query, session)] * len(sample),
                                                                        names=['template', 'query', 'session'])))

    return pd.concat(dfs).sort_index()


if __name__ == '__main__':
    from IPython import embed

    data_dir = sys.argv[1]
    df = load_tactile(data_dir)
    embed()
