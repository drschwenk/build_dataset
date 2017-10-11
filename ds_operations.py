import os
import multiprocessing
from fsds_utils.tracking import draw_video_tracking
from amt_utils.mturk import pickle_this, unpickle_this


procs = 8


def multimap(method, iterable, *args):
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['THEANO_FLAGS'] = 'device=cpu'

    pool = multiprocessing.Pool(procs)
    results = pool.map(method, iterable)
    pool.close()
    pool.join()

    return results


def perform_ds_operation (ds, operation):
    pass


if __name__ == '__main__':
    # prod_dataset = unpickle_this( 'prod_dataset_10_6.pkl')
    random_complete_sample = unpickle_this('rsample_1_update.pkl')
    # draw_video_tracking(random_complete_sample[0])
    _ = multimap(draw_video_tracking, random_complete_sample)
    # pickle_this(all_hits, 'latest_hit_group_8_19.pkl')
