import os
import multiprocessing
from functools import partial
from anigen_tools.mturk import unpickle_this
from anigen_tools.interpolation import interpolate_all_video_entites
from anigen_tools.tracking import track_all_video_entites
from anigen_tools.frame_extraction import video_to_npy
from anigen_tools.interpolation import draw_video_interps
procs = os.cpu_count()




def multimap(method, iterable, *args):
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['THEANO_FLAGS'] = 'device=cpu'

    pool = multiprocessing.Pool(procs)
    results = pool.map(method, iterable)
    pool.close()
    pool.join()

    return results


def perform_interpolation(videos):
    complete_vids_all = unpickle_this('complete_vids_all.pkl')
    complete_prod_dataset = unpickle_this('complete_prod_dataset.pkl')
    # interp_func = partial(interpolate_all_video_entites, complete_prod_dataset)    
    interp_func = partial(draw_video_interps, complete_prod_dataset)
    _ = multimap(interp_func, videos)


def perform_tracking(videos):
    _ = multimap(track_all_video_entites, videos)


def perform_frame_extraction(videos):
    _ = multimap(video_to_npy, videos)



if __name__ == '__main__':
    complete_vids_all = unpickle_this('complete_ds_checkpoint_v2p8_cleaned.pkl')    
    with open('./overwritten') as f:
        retrack_ids = f.readlines()
        retrack_ids = [l.strip() for l in retrack_ids]
        retrack_ids = set(['_'.join(l.split('_')[:7]) for l in retrack_ids])
    vids_to_do = [v for v in complete_vids_all if v.gid() in retrack_ids]
    # perform_frame_extraction(complete_vids_all[:10])
    # perform_interpolation(complete_vids_all[:10])
    perform_tracking(vids_to_do)
