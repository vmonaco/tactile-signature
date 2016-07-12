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

    return np.core.records.fromarrays([t - t[0], x, y, p],
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

        dfs.append(pd.DataFrame(sample, index=pd.MultiIndex.from_tuples([(template, query, session)] * len(sample),
                                                                        names=['template', 'query', 'session'])))

    return pd.concat(dfs).sort_index()


def animate_sample(sample, update_step=50, update_interval=1, out=None):
    """
    Animate a tactile sample using matplotlib.

    :param sample: a tactile sample loaded with the load_lsa function
    :param update_step: size of the update window in milliseconds
    :param update_interval: time between screen updates in milliseconds
    """
    if out is not None:
        import matplotlib
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import gridspec

    T = int(np.ceil((sample['t'].max() / 10) / update_step))

    fig = plt.figure(figsize=(6, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1])
    ax_spikes = plt.subplot(gs[0])
    ax_time = plt.subplot(gs[1])

    ax_spikes.set_xlim(0, 64)
    ax_spikes.set_ylim(0, 64)

    ax_time.set_xlim(sample['t'][0] / 10, sample['t'][-1] / 10)
    ax_time.set_ylim(-1, 1)
    ax_time.set_xlabel('Time (ms)')
    ax_time.get_yaxis().set_ticks([])

    on_spikes = ax_spikes.scatter([], [], c='r', alpha=0.5, label='Pressure increase')
    off_spikes = ax_spikes.scatter([], [], c='b', alpha=0.5, label='Pressure decrease')
    timeline = ax_time.scatter(0, 0, c='k')

    ax_spikes.legend(loc='upper right')

    def update_frame(t_step):
        if t_step == 0:
            return

        begin_t = ((t_step * update_step) - update_step) % (T * update_step)
        end_t = (t_step * update_step) % (T * update_step)

        s = sample[(sample['t'] >= begin_t * 10) & (sample['t'] < end_t * 10)]
        on_spikes.set_offsets(s[s['p'].view(np.bool)][['x', 'y']].view(np.uint8))
        off_spikes.set_offsets(s[~s['p'].view(np.bool)][['x', 'y']].view(np.uint8))
        timeline.set_offsets([(begin_t + end_t) / 2, 0])

        return on_spikes, off_spikes, timeline

    if out is None:
        ani = animation.FuncAnimation(fig, update_frame, frames=T, interval=update_interval, blit=False, repeat=True)
        plt.show()
    else:
        ani = animation.FuncAnimation(fig, update_frame, frames=T, interval=update_interval, blit=False, repeat=False)
        if 'gif' == out[-3:]:
            ani.save(out, writer='imagemagick')
        elif 'mp4' == out[-3:]:
            writer = animation.writers['ffmpeg'](fps=15, bitrate=None)
            ani.save(out, writer=writer)
        else:
            raise Exception('Unknown output type:', out)


if __name__ == '__main__':
    from IPython import embed

    data_dir = sys.argv[1]

    # Animate a window
    animate_sample(load_lsa(data_dir + '/DS_DS_01.lsa'))

    # Create a gif with the out parameter
    animate_sample(load_lsa(data_dir + '/DS_DS_01.lsa'), out='signature.gif')

    df = load_tactile(data_dir)
    embed()
