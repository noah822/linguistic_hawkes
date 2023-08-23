import numpy as np
from utils.vocab import Reservoir

from hawkes.fit import DiscreteHawkes

from dask.distributed import Client

import numpy as np
import json
import time


def load_hawkes_from_file(path: str):
    with open(path, 'r') as handler:
        param_list = json.load(handler)
    mu0 = param_list['mu0'];  A = param_list['A']
    mu_t = np.array(param_list['mu_t'])
    g_t = np.array(param_list['g_t'])
    print(f'Successfully load model from {path}')
    return mu0, A, mu_t, g_t

import argparse

def _generate_argparser(
    init_mu0: float=0.01,
    init_A: float=0.01,
    init_exp_lambda: float=0.02,
    mle_inner_loop: int=5,
    X_chunk_size: int=8000,
    kernel_chunk_size: int=2000,
    lag_chunk_size: int=4000,
    g_window_size: int=2000,
    num_worker: int=36
):
    parser = argparse.ArgumentParser(prog='Discrete Hawkes for Linguistics')
    parser.add_argument('--word', type=str, help='target word to fit the model on', required=True)
    parser.add_argument('--corpus-path', type=str,
                       help="""Path to the json file that stores corpus information.
                               Json file should adhere to the format of
                               [key]: `word`
                               [value]: `list of occurence position` (List[int])
                            """,
                        required=True
                       )
    parser.add_argument('--total-word-num', type=int, help='total word count', required=True)
    parser.add_argument('--target-word-num', type=int, help='number of target word to look at',
                        default=None)
    parser.add_argument('--ckp-save-path', type=str, help='path to save the checkpoint of fitted model', required=True)
    parser.add_argument('--X-bandwidth', type=int,
                        help='size of kernel bandwidth for estimating background', required=True)
    parser.add_argument('--lag-bandwidth', type=int,
                        help='size of kernel bandwidth for estimating g', required=True)
    parser.add_argument('--gif-save-path', type=str,
                        help='directory to save the animation that records the training progress', required=True)
    parser.add_argument('--epoch', type=int, help='number of training epoch')
    parser.add_argument('--mle-inner-loop', type=int, default=mle_inner_loop,
                        help='number of inner iterative mle loop')
    parser.add_argument('--num-worker', type=int, default=num_worker,
                        help='number of dask worker, i.e threads spawned from main process')
    parser.add_argument('--X-chunk-size', type=int, default=X_chunk_size,
                        help='chunk size to split across episode of whole process')
    parser.add_argument('--kernel-chunk-size', type=int, default=kernel_chunk_size,
                        help="""chunk size to split across episode of whole process,
                                array of size ($kernel-chunk-size, $total-word-num)
                                should be able to fit into RAM by all workers simultanously
                        """)
    parser.add_argument('--g-window-size', type=int, default=g_window_size,
                        help="""number of preceding event to consider. Event occurring beyond this
                        threshold will be ignored""")
    parser.add_argument('--lag-chunk-size', type=int, default=lag_chunk_size,
                        help="""chunk size to split across episode of whole process,
                                array of size ($lag-chunk-size, $g-window-size) should be 
                                able to fit into RAM by all workers simultanously
                        """)
    parser.add_argument('--init-mu0', type=float, default=init_mu0,
                        help='init value for mu0')
    parser.add_argument('--init-A', type=float, default=init_A,
                        help='init value for A')
    parser.add_argument('--init-exp-lambda', type=float, default=init_exp_lambda,
                        help='parameter $lambda for exponential distribution to initilize g')
    return parser


vocab = Reservoir(
    from_json='./asset/corpus.json',
)


if __name__ == '__main__':
    client = Client(processes=False)

    config = _generate_argparser(
        mle_inner_loop=5,
        X_chunk_size=8000,
        kernel_chunk_size=200,
        lag_chunk_size=4000
    )

    process = DiscreteHawkes(
        bandwidth={
            'X' : config.X_bandwidth,
            'lag' : config.lag_bandwidth
        },
        init_mu0=config.init_mu0,
        init_A=config.init_A,
        init_exp_lambda=config.init_exp_lambda,
        mle_iter_round=config.mle_inner_loop,
        save_path=config.ckp_save_path,
        save_as_gif_path=config.gif_save_path,
        chunk_config={
            'kernel' : config.kernel_chunk_size,
            'X' : config.X_chunk_size,
            'occur_lag' : config.lag_chunk_size
        },
        g_truncate_bound=config.g_window_size
    )
    TOTAL_TOKEN_NUM = config.total_word_num
    # TOTAL_TOKEN_NUM = 62_100

    with open(config.corpus_path, 'r') as handler:
        corpus = json.load(handler)

    pos = corpus[config.word]

    if config.target_word_num is not None:
        pos = pos[:config.target_word_num]

    X = np.zeros(TOTAL_TOKEN_NUM, dtype=np.int32)
    X[pos] = 1
    num_sample = X.shape[0]

    print(f'total length: {TOTAL_TOKEN_NUM}\t occurance: {len(pos)}')


    s = time.time()
    mu0, A, mu_t, g_t = process.fit(
        X, epoch=config.epoch, verbose=True,
    )
    e = time.time()
    print('time consumed: {:.2f}'.format(e-s))



