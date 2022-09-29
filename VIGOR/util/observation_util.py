from typing import Union, List, Tuple

import numpy as np


def append_contexts(samples: Union[List[np.array], np.array], contexts: np.array,
                    flatten: bool = False) -> Union[List[np.array], np.array]:
    """
    We create contextualized samples by simply appending the corresponding context to a given sample. The resulting
    concatenated sample can be further processed by appropriate functions
    Args:
        samples: Either an array of samples per context, a list of samples per context, or an array of samples for
        a single context
        contexts: One or more contexts
        flatten: Whether to return the original shape or flatten to a 2d np array with shape
            [#num_samples, #num_sample_dims+#num_context_dims]

    Returns:

    """
    assert contexts.ndim <= 3, "Need 1- to 3-dimensional contexts"
    if isinstance(samples, list):
        assert len(samples) == len(contexts)
        contextualized_samples = []
        for sample_batch, context in zip(samples, contexts):
            assert isinstance(samples, np.ndarray), "Must have samples provided as numpy array"
            if sample_batch.ndim == 2:
                context = np.broadcast_to(context[None, :], shape=(sample_batch.shape[0], len(context)))
                contextualized_batch = np.concatenate((sample_batch, context), axis=-1)
                contextualized_samples.append(contextualized_batch)
            else:
                raise NotImplementedError("append_contexts for lists w/ more than 2d currently not supported")
        if flatten:
            contextualized_samples = np.concatenate(contextualized_samples, axis=0)
    else:
        if samples.ndim == 4:  # something like multiple contexts and time series
            assert len(samples) == len(contexts), "Samples and contexts must be aligned. " \
                                                  "Got shapes {} and {}".format(samples.shape, contexts.shape)
            if contexts.ndim == 2:
                contexts = np.broadcast_to(contexts[:, None, None, :],
                                           shape=(
                                               samples.shape[0], samples.shape[1], samples.shape[2], contexts.shape[1]))
            elif contexts.ndim == 3:
                assert contexts.shape[:1] == samples.shape[:1]
                contexts = np.broadcast_to(contexts[:, :, None, :],
                                           shape=(
                                               samples.shape[0], samples.shape[1], samples.shape[2],
                                               contexts.shape[-1]))
            else:
                raise ValueError("Bad context dimension for contexts of shape {}".format(contexts.shape))

            contextualized_samples = np.concatenate((samples, contexts), axis=-1)
            if flatten:
                contextualized_samples = np.concatenate(contextualized_samples, axis=0)
            return contextualized_samples
        elif samples.ndim == 3:  # samples for multiple contexts
            assert len(samples) == len(contexts), "Samples and contexts must be aligned. " \
                                                  "Got shapes {} and {}".format(samples.shape, contexts.shape)
            contexts = np.broadcast_to(contexts[:, None, :],
                                       shape=(samples.shape[0], samples.shape[1], contexts.shape[1]))
            contextualized_samples = np.concatenate((samples, contexts), axis=-1)
            if flatten:
                contextualized_samples = np.concatenate(contextualized_samples, axis=0)
            return contextualized_samples
        elif samples.ndim == 2:  # single context
            assert contexts.ndim == 1
            contexts = np.broadcast_to(contexts[None, :],
                                       shape=(samples.shape[0], contexts.shape[0]))
            contextualized_samples = np.concatenate((samples, contexts), axis=-1)
        else:
            raise ValueError("Bad type/shape for samples", samples.shape)

    return contextualized_samples
