import os
import sys
import numpy as np
import pandas as pd
from itertools import chain
from operator import itemgetter
from sklearn.metrics import roc_curve
from pohmm import Pohmm

from data import load_tactile


def preprocess(df):
    """
    Extract features from a tactile sample.
    """
    # Throw away samples samples with 0 time delta
    df = df.drop_duplicates('t')

    df = pd.DataFrame({
        'event': df['p'],
        'dt': df['t'].astype(np.int32).diff(),
        'dx': df['x'].astype(np.int32).diff(),
        'dy': df['y'].astype(np.int32).diff(),
    })

    df = df.dropna()
    return df[['event', 'dt', 'dx', 'dy']]


def pohmm_factory():
    """
    Factory function to create the classifier objects
    """
    return Pohmm(n_hidden_states=2,
                 init_spread=1,
                 emissions=[('dt', 'lognormal'), ('dx', 'normal'), ('dy', 'normal')],
                 smoothing='freq',
                 init_method='obs',
                 thresh=1)


def identification_results(df, num_train=1):
    """
    Perform a 1 out of N user classification by choosing the maximum likelihood model under each sample.
    """
    # Keep only genuine samples
    df = df[df.index.get_level_values('template') == df.index.get_level_values('query')]

    # Train a model for each user
    models = {}
    for l in df.index.get_level_values('query').unique():
        print('Fitting', l)
        models[l] = pohmm_factory()
        train_idx = sorted(df.loc[l, l].index.get_level_values('session').unique())[:num_train]
        models[l].fit_df([preprocess(df.loc[l, l, session]) for session in train_idx])

    # Classify the remaining samples (split by the :num_train slice for each user)
    predict_labels, ground_truth = [], []
    for l in df.index.get_level_values('query').unique():
        test_idx = sorted(df.loc[l, l].index.get_level_values('session').unique())[num_train:]

        for session in test_idx:
            logliks = [(i, p.score_df(preprocess(df.loc[l, l, session]))) for i, p in models.items()]
            predict_labels.append(max(logliks, key=itemgetter(1))[0])
            ground_truth.append(l)

    predict_labels, ground_truth = np.array(predict_labels), np.array(ground_truth)
    print('Identification ACC:', (predict_labels == ground_truth).sum() / len(ground_truth))


def verification_results(df, num_train=1):
    """
    Compute the verification equal error rate (EER) for each user.
    """
    scores = []
    for l in df.index.get_level_values('template').unique():
        print('Fitting', l)
        cl = pohmm_factory()

        session_idx = sorted(df.loc[l, l].index.get_level_values('session').unique())
        genuine_train_idx = session_idx[:num_train]
        genuine_test_idx = session_idx[num_train:]

        impostor_test = df[(df.index.get_level_values('template') == l) &
                           (df.index.get_level_values('query') != l)]

        # Skip templates with no impostor samples
        if len(impostor_test) == 0:
            continue

        cl.fit_df([preprocess(df.loc[l, l, session]) for session in genuine_train_idx])

        # Score the remaining genuine samples
        for session in genuine_test_idx:
            score = cl.score_df(preprocess(df.loc[l, l, session]))/len(df.loc[l, l, session])
            scores.append((l, l, session, score))

        # Score all of the impostor samples
        for (template, query, session), test_sample in impostor_test.groupby(level=[0, 1, 2]):
            score = cl.score_df(preprocess(test_sample))/len(test_sample)
            scores.append((template, query, session, score))

    scores = pd.DataFrame(scores, columns=['template', 'query', 'session', 'score'])

    # for l in df.index.get_level_values('template').unique():


    def eer_from_scores(s):
        far, tpr, thresholds = roc_curve((s['template'] == s['query']).values, s['score'])
        frr = (1 - tpr)
        idx = np.argmin(np.abs(far - frr))
        return np.mean([far[idx], frr[idx]])

    eers = scores.groupby(['template']).apply(eer_from_scores)
    print('Verification EER: %.2f +/- %.2f' % (eers.mean(), eers.std()))


def zeroshot_verification_results(df):
    """
    Classify unknown samples as being either genuine or impostor without having previously seen the signature.
    This is a zero-shot learning scenario in terms of verification since the genuine user's template is unknown.
    Classification is performed by choosing the model (either genuine or impostor) with higher loglikelihood.
    """
    # Index the genuine samples
    genuine = df.index.get_level_values('template') == df.index.get_level_values('query')

    df_genuine = df[genuine]
    df_impostor = df[~genuine]

    predict_labels, ground_truth, scores = [], [], []
    cl_genuine, cl_impostor = pohmm_factory(), pohmm_factory()
    for l in df.index.get_level_values('template').unique():
        print('Fitting', l)

        test_genuine_idx = (df_genuine.index.get_level_values('template') == l) & (
            df_genuine.index.get_level_values('query') == l)

        test_impostor_idx = (df_impostor.index.get_level_values('template') == l) & (
            df_impostor.index.get_level_values('query') != l)

        train_genuine = df_genuine[~test_genuine_idx]
        train_impostor = df_impostor[~test_impostor_idx]
        test_genuine = df_genuine[test_genuine_idx]
        test_impostor = df_impostor[test_impostor_idx]

        cl_genuine.fit_df([preprocess(s) for _, s in train_genuine.groupby(level=[0, 1, 2])])
        cl_impostor.fit_df([preprocess(s) for _, s in train_impostor.groupby(level=[0, 1, 2])])

        for _, test_sample in chain(test_genuine.groupby(level=[0, 1, 2]), test_impostor.groupby(level=[0, 1, 2])):
            ground_truth.append(
                (test_sample.index.get_level_values('template') == test_sample.index.get_level_values('query')).all())

            genuine_loglik = cl_genuine.score_df(preprocess(test_sample))
            impostor_loglik = cl_impostor.score_df(preprocess(test_sample))

            predict_labels.append(genuine_loglik > impostor_loglik)
            scores.append(2 * (impostor_loglik - genuine_loglik))

    predict_labels, ground_truth, scores = np.array(predict_labels), np.array(ground_truth), np.array(scores)
    print('Zero-shot verification ACC:', (predict_labels == ground_truth).sum() / len(ground_truth))


if __name__ == '__main__':
    np.random.seed(1234)
    if len(sys.argv) < 2:
        print('Usage: $ python classify.py data_dir')
        sys.exit(1)

    data_dir = sys.argv[1]
    df = load_tactile(data_dir)

    identification_results(df, num_train=1)
    identification_results(df, num_train=5)
    verification_results(df, num_train=1)
    zeroshot_verification_results(df)
