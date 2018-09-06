import numpy as np
import cv2
import imageio
import tensorflow as tf
import json

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import action_detection.action_detector as act

def test_on_local_segment():
    actors = [0,1,3]
    size = [400,400]
    timesteps = 32
    batch_np = np.zeros([len(actors), timesteps] + size + [3])
    rois_np = np.zeros([len(actors), 4])
    batch_indices_np = np.array(range(len(actors)))

    for bb, actor_id in enumerate(actors):
        vid_path = 'person_%i.mp4' % actor_id
        reader = imageio.get_reader(vid_path, 'ffmpeg')
        for tt, frame in enumerate(reader):
            batch_np[bb,tt,:] = frame
        
        roi_path = "person_%i_roi.json" % actor_id
        with open(roi_path) as fp:
            rois_np[bb] = json.load(fp)
    
    act_detector = act.Action_Detector('i3d_tail')
    ckpt_name = 'model_ckpt_RGB_i3d_pooled_tail-4'

    main_folder = sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)

    act_detector.restore_model(ckpt_path)

    input_seq, rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders()

    feed_dict = {input_seq:batch_np, rois:rois_np, roi_batch_indices:batch_indices_np}
    probs = act_detector.session.run(pred_probs, feed_dict=feed_dict)

    highest_conf_actions = np.argmax(probs, axis=1)
    for ii in range(len(actors)):
        print("Person %i with action %s" % (actors[ii], act.ACTION_STRINGS[highest_conf_actions[ii]]))



if __name__ == '__main__':
    test_on_local_segment()