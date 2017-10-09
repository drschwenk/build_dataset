from skimage.feature import match_template
import skimage.draw as skdraw
import os
import numpy as np
from skimage.feature import match_template
import skimage.draw as skdraw
from amt_utils.mturk import pickle_this, unpickle_this
import os
import numpy as np
import PIL.Image as pil
import cv2


new_dim = 128
owidth = 640
oheight = 480

scale_down = new_dim / owidth
asp_ratio = owidth / oheight

prod_dataset = unpickle_this('prod_dataset_10_6.pkl')


def track(frames,boxes,anno_frame_ids,search_region_scale_factor,img_size):
    boxes = boxes.astype(np.float32)

    # Template box
    tx1,ty1,tx2,ty2 = boxes[anno_frame_ids[0],:].astype(np.int32)
    template = frames[anno_frame_ids[0],ty1:ty2,tx1:tx2,:]
    for i in range(anno_frame_ids[0]):
        boxes[i,:] = track_inner(
            frames[i],
            boxes[i,:],
            template,
            search_region_scale_factor,
            img_size)

    for j in range(len(anno_frame_ids)-1):
        template_choices = [None]*2
        tx1,ty1,tx2,ty2 = boxes[anno_frame_ids[j],:].astype(np.int32)
        template_choices[0] = frames[anno_frame_ids[j],ty1:ty2,tx1:tx2,:]
        tx1,ty1,tx2,ty2 = boxes[anno_frame_ids[j+1],:].astype(np.int32)
        template_choices[1] = frames[anno_frame_ids[j+1],ty1:ty2,tx1:tx2,:]
        for i in range(anno_frame_ids[j]+1,anno_frame_ids[j+1]):
            if i-anno_frame_ids[j] < anno_frame_ids[j+1]-i:
                template = template_choices[0]
            else:
                template = template_choices[1]

            boxes[i,:] = track_inner(
                frames[i],
                boxes[i,:],
                template,
                search_region_scale_factor,
                img_size)

    tx1,ty1,tx2,ty2 = boxes[anno_frame_ids[-1],:].astype(np.int32)
    template = frames[anno_frame_ids[-1],ty1:ty2,tx1:tx2,:]
    for i in range(anno_frame_ids[-1],frames.shape[0]):
        boxes[i,:] = track_inner(
            frames[i],
            boxes[i,:],
            template,
            search_region_scale_factor,
            img_size)

    return boxes


def track_inner(frame,box,template,search_region_scale_factor,img_size):
    im_h,im_w = img_size
    x1,y1,x2,y2 = box
    w = search_region_scale_factor*(x2-x1)
    h = search_region_scale_factor*(y2-y1)
    x1_ = int(max(0,0.5*(x1+x2-w)))
    x2_ = int(min(im_w,0.5*(x1+x2+w)))
    y1_ = int(max(0,0.5*(y1+y2-h)))
    y2_ = int(min(im_h,0.5*(y1+y2+h)))
    frame = frame[y1_:y2_,x1_:x2_,:]
    if any([v==0 for v in template.shape]) or any([v==0 for v in frame.shape]):
        return box

#     print(template.shape, frame.shape)
    result = match_template(frame,template,pad_input=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    cy,cx = ij[0:2]
    del_x = cx + x1_ - 0.5*(x1+x2)
    del_y = cy + y1_ - 0.5*(y1+y2)
    x1,x2 = [x+del_x for x in [x1,x2]]
    y1,y2 = [y+del_y for y in [y1,y2]]
    box = [x1,y1,x2,y2]

    return box


def scale_boxes(boxes,in_frame_size,out_frame_size):
    hi,wi = in_frame_size
    ho,wo = out_frame_size
    fh = ho/float(hi)
    fw = wo/float(wi)
    boxes = np.array(boxes).astype(np.float32)
    for i in [0,2]:
        boxes[:,i] *= fw
        boxes[:,i+1] *= fh

    boxes = boxes.astype(np.int32)

    return boxes


def draw_boxes(frames,boxes):
    for i in range(boxes.shape[0]):
        x1,y1,x2,y2 = boxes[i,:]
        c = [x1,x2,x2,x1]
        r = [y1,y1,y2,y2]
        rr,cc = skdraw.polygon_perimeter(r,c,shape=frames[0].shape[0:2])
        frames[i,rr,cc,:] = 0

    return 

def track_all_video_entites(video, frame_arr_data):
    all_eids = [ent.gid() for ent in video.data()['characters'] +  video.data()['objects'] if ent.data()['entityLabel'] != 'None']
    return [generate_tracking(prod_dataset, eid, frame_arr_data) for eid in all_eids]

def track_rects(entity_key_rects, frame_arr_data):
    boxes = scale_boxes(entity_key_rects,(480,640),(128,128))
    boxes = track(frame_arr_data,boxes,[9,39,69],1.1,(128,128))
    rescaled_boxes = scale_boxes(boxes,(128,128), (480,640))
    return rescaled_boxes

def generate_tracking(dataset, eid, frame_arr_data, interp_dir = 'tracked_boxes'):
    boxes_npy = os.path.join('interpolated_boxes', eid + '.npy')
    boxes = np.load(boxes_npy)
    entity_rects = np.nan_to_num(track_rects(boxes, frame_arr_data))
    outfile = os.path.join(interp_dir, eid + '.npy')
    np.save(outfile, entity_rects)
    return entity_rects

def draw_all_bboxes(frame_arr_square, raw_bboxes, entity_type = 'character'):
    color_assignments = {
        'character': (0, 255, 255),
        'object': (0, 255, 0),
    }
    frame_arr = cv2.resize(frame_arr_square, None, fx = asp_ratio, fy=1)
    bboxes = [bb.reshape(2, 2) for bb in raw_bboxes]
    for bb in bboxes:
        bb[:, 0] = bb[:, 0] * scale_down * asp_ratio
        bb[:, 1] = bb[:, 1] * scale_down * asp_ratio
    bboxes = [bb.astype(int) for bb in bboxes]
    _ = [cv2.rectangle(frame_arr, tuple(bb[0]), tuple(bb[1]), color_assignments[entity_type] , thickness=1) for bb in bboxes]
    return pil.fromarray(frame_arr)

def draw_video_tracking(video, play=False, out_dir='tracking_viz', frame_dir='./object_tracking/vid_arr_data/'):
    outfile = os.path.join(out_dir, video.gid() + '_tracking.gif')
    frame_arr_data = np.load(frame_dir + video.gid() + '.npy')
    try:
        entity_interps = track_all_video_entites(video, frame_arr_data)
        interp_img_seq = [draw_all_bboxes(frame_arr_data[frame_n], [entity_rect[frame_n] for entity_rect in entity_interps], 
            'object') for frame_n in range(frame_arr_data.shape[0])]
        interp_img_seq[0].save(outfile, save_all=True, optimize=True, duration=42, append_images=interp_img_seq[1:])
        if play:
            return Image(filename=outfile)
        return interp_img_seq
    except ValueError as e:
        print(video.gid(),'\n', e)
