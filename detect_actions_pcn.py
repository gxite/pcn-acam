import numpy as np
import cv2
import imageio
import tensorflow as tf
import json

import os
import sys
import argparse

import object_detection.object_detector_pcn as obj
import action_detection.action_detector_pcn as act

import time

DISPLAY = False
SHOW_CAMS = False

##Debug##
disable_writer = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', type=str, required=False, default="")
    parser.add_argument('-d', '--display', type=str, required=False, default="True")
    args = parser.parse_args()
    display = (args.display == "True" or args.display == "true")

    video_path = args.video_path
    basename = os.path.basename(video_path).split('.')[0]
    out_vid_path = "./output_videos/%s_output.mp4" % (basename if not SHOW_CAMS else basename+'_cams_actor_%.2d' % actor_to_display)
    out_img_path = "./output_videos/%s_output.jpg" % basename

    main_folder = './'

    #actor_to_display = 6 # for cams

    # NAS

    #Sets the name of the object detection model
    #obj_detection_model =  'ssd_mobilenet_v2_coco_2018_03_29' 
    obj_detection_model =  'faster_rcnn_nas_coco_2018_01_28'

    obj_detection_graph = os.path.join("object_detection", "weights", obj_detection_model, "frozen_inference_graph.pb")
    print("Loading object detection model at %s" % obj_detection_graph)
    obj_detector = obj.Object_Detector(obj_detection_graph)

    #Sets the tracker object
    tracker = obj.Tracker()

    #Frequency of action detection
    action_freq = 8

    print("Reading video file %s" % video_path)
    print('Running actions every %i frame' % action_freq)

    #Configures the opencv reader
    reader = cv2.VideoCapture(video_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    W = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = tracker.timesteps
    #Sets the path for output video and initializes VideoWriter
    if not display:
        if not disable_writer:
            writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
            print("Writing output to %s" % out_vid_path)

    #Sets the action detector model & checkpoint
    act_detector = act.Action_Detector('soft_attn')
    ckpt_name = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'

    memory_size = act_detector.timesteps - action_freq

    #Sets variables for action detectors
    updated_frames, temporal_rois, temporal_roi_batch_indices, cropped_frames = act_detector.crop_tubes_in_tf_with_memory([T,H,W,3], memory_size)
    rois, roi_batch_indices, pred_probs = act_detector.define_inference_with_placeholders_noinput(cropped_frames)
    
    ckpt_path = os.path.join(main_folder, 'action_detection', 'weights', ckpt_name)
    act_detector.restore_model(ckpt_path)

    #Loops through every frame of the video
    prob_dict = {}
    frame_count = 0
    while True:
        
        return_value, current_video_frame = reader.read()
        if return_value == False:
            break

        frame_count += 1
        print("frame_count: %i" %frame_count)

        #--Object Detection--#

        current_video_frame_expanded = np.expand_dims(current_video_frame, axis=0)

        t1 = time.time()

        #calling of the object detection functions.
        detection_list = obj_detector.detect_objects_in_np(current_video_frame_expanded)
        detection_info = [info[0] for info in detection_list]

        t2 = time.time(); print('obj det %.2f seconds' % (t2-t1))

        #Tracker
        tracker.update_tracker(detection_info, current_video_frame)
        
        t3 = time.time(); print('tracker %.2f seconds' % (t3-t2))
        num_actors = len(tracker.active_actors)

        #--Action detection--#

        if tracker.active_actors and frame_count % action_freq == 0:
            probs = []

            cur_input_sequence = np.expand_dims(np.stack(tracker.frame_history[-action_freq:], axis=0), axis=0)

            rois_np, temporal_rois_np = tracker.generate_all_rois()

            #Clips the number of actors to the maximum number of actors
            MAX_NUM_ACTORS = 14    
            if num_actors > MAX_NUM_ACTORS:
                num_actors = MAX_NUM_ACTORS
                rois_np = rois_np[:MAX_NUM_ACTORS]
                temporal_rois_np = temporal_rois_np[:MAX_NUM_ACTORS]

            feed_dict = {updated_frames:cur_input_sequence, # only update last #action_freq frames
                         temporal_rois: temporal_rois_np,
                         temporal_roi_batch_indices: np.zeros(num_actors),
                         rois:rois_np, 
                         roi_batch_indices:np.arange(num_actors)}

            run_dict = {'pred_probs': pred_probs}

            if SHOW_CAMS:
                run_dict['cropped_frames'] = cropped_frames
                run_dict['final_i3d_feats'] =  act_detector.act_graph.get_collection('final_i3d_feats')[0]
                run_dict['cls_weights'] = act_detector.act_graph.get_collection('variables')[-2] # this is the kernel

            out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)

            probs = out_dict['pred_probs'] #Stores the confidence score (probaility) of all actions for EACH Actor in an numpy array

            # associate probs with actor ids
            TOP_NUMBER_OF_ITEMS_TO_RETURN_IN_LIST = 5

            for bbox in range(num_actors):

                #all action probabilities for an actor in the indexed sequence in ACTION_DESCRIPTION dictionary
                action_probs = probs[bbox] #numpy array

                #returns a numpy array of the same shape
                #Sorts the indexed action according to probability in decending order
                ordered_action_probs = np.argsort(action_probs)[::-1]

                cur_actor_id = tracker.active_actors[bbox]['actor_id']
                print("Person %i" % cur_actor_id)

                cur_results = []

                TOP_ACTIONS_DICT = {}
                for i in range(TOP_NUMBER_OF_ITEMS_TO_RETURN_IN_LIST):
                    action_key = ordered_action_probs[i]
                    ''' int : [string, float]'''
                    TOP_ACTIONS_DICT[action_key] = [act.ACTION_DESCRIPTION[action_key], action_probs[action_key]]
                
                #appends description and action probablity to current results.
                #prints output in the terminal
                for action_key in TOP_ACTIONS_DICT: 
                    top_action_prob = TOP_ACTIONS_DICT[action_key][1]
                    top_action_description = TOP_ACTIONS_DICT[action_key][0] 

                    print('\t {}: {:.3f}'.format(top_action_description, top_action_prob)) # run/jog: 0.140
                    cur_results.append((top_action_description, top_action_prob))

                #to determine what is the likely overall action based on the top action list and their probabilities
                overall_action_description, overall_action_prob = get_overall_action(TOP_ACTIONS_DICT)

                #Append overall_action_description to cur_results
                print('\t Overall: {}, {:.3f}'.format(overall_action_description, overall_action_prob))
                cur_results.append((overall_action_description, overall_action_prob))

                prob_dict[cur_actor_id] = cur_results 

            t4 = time.time(); print('action %.2f seconds' % (t4-t3))
        
        if frame_count > 16:

            out_img = visualize_detection_results(tracker.frame_history[-16], tracker.active_actors, prob_dict)

            if SHOW_CAMS:
                if tracker.active_actors:
                    actor_indices = [ii for ii in range(num_actors) if tracker.active_actors[ii]['actor_id'] == actor_to_display]
                    if actor_indices:
                        out_img = visualize_cams(out_img, cur_input_sequence, out_dict, actor_indices[0])
                    else:
                        continue
                else:
                    continue
            if display: 
                cv2.imshow('results', out_img[:,:,::-1])
                cv2.waitKey(10)
            else:
                if not disable_writer:
                    writer.write(out_img)
                cv2.imwrite(out_img_path,out_img)
        
    if not display:
        if not disable_writer:
            writer.release()   

#evaluates overall action in a given frame
def get_overall_action(action_dict):
    top_action = ["NULL", 0]
    run_jog_present = False
    carry_present = False
    walk_present = False
    for action_key, action_description_prob in action_dict.items():
        if action_description_prob[1] > top_action[1]:
            top_action = action_description_prob
        if action_key == 8 and action_description_prob[1] > 0.1: #8: run/jog
            run_jog_present = True
        if action_key == 14 and action_description_prob[1] > 0.2: #14: carry/hold
            carry_present = True
        if action_key == 12 and action_description_prob[1] > 0.5: #12: walk
            walk_present = True
    
    if walk_present and run_jog_present and carry_present:
        top_action = ["run/jog", -99]

    return top_action


np.random.seed(10)
COLORS = np.random.randint(0, 255, [1000, 3])
def visualize_detection_results(img_np, active_actors, prob_dict):
    score_th = 0.30
    action_th = 0

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape

    for ii in range(len(active_actors)):
        cur_actor = active_actors[ii]
        actor_id = cur_actor['actor_id']
        cur_act_results = prob_dict[actor_id] if actor_id in prob_dict else []
        object_class = 1   # 'person' has a id of 1 in OBJECT_STRINGS

        try:
            cur_box, cur_score, cur_class = cur_actor['all_boxes'][-16], cur_actor['all_scores'][0], object_class
        except IndexError:
            continue
        
        if cur_score < score_th: 
            continue

        top, left, bottom, right = cur_box

        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = cur_score
        label = obj.OBJECT_STRINGS[cur_class]['name']
        message = '%s_%i: %% %.2f' % (label, actor_id,conf)
        action_message_list = ["%s:%.3f" % (actres[0][0:7], actres[1]) for actres in cur_act_results if actres[1]>action_th]
        action_summary = cur_act_results[-1][0]

        color = COLORS[actor_id]
        cv2.rectangle(disp_img, (left,top), (right,bottom), color.tolist(), 3)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))

        #position and writes the overall action message.
        cv2.rectangle(disp_img, (left, top-int(font_size*40)-60), (right,top-int(font_size*40)), color.tolist(), -1)
        cv2.putText(disp_img, action_summary, (left, top-40), 0, font_size*2, (np.array([255,255,255])-color).tolist(), 1)

        #position and writes the label, actor id and confidence 
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color.tolist(), -1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, (np.array([255,255,255])-color).tolist(), 1)

        #position and writes the action messages.
        cv2.rectangle(disp_img, (left, top), (right,top+10*len(action_message_list)), color.tolist(), -1)
        for aa, action_message in enumerate(action_message_list):
            offset = aa*10
            cv2.putText(disp_img, action_message, (left, top+5+offset), 0, font_size, (np.array([255,255,255])-color).tolist(), 1)


    return disp_img


def visualize_cams(image, input_frames, out_dict, actor_idx):
    #classes = ["walk", "bend", "carry"]
    #classes = ["sit", "ride"]
    classes = ["talk to", "watch (a", "listen to"]
    action_classes = [cc for cc in range(60) if any([cname in act.ACTION_DESCRIPTION[cc] for cname in classes])]

    feature_activations = out_dict['final_i3d_feats']
    cls_weights = out_dict['cls_weights']
    input_frames = out_dict['cropped_frames'].astype(np.uint8)
    probs = out_dict["pred_probs"]

    class_maps = np.matmul(feature_activations, cls_weights)
    min_val = np.min(class_maps[:,:, :, :, :])
    max_val = np.max(class_maps[:,:, :, :, :]) - min_val

    normalized_cmaps = np.uint8((class_maps-min_val)/max_val * 255.)

    t_feats = feature_activations.shape[1]
    t_input = input_frames.shape[1]
    index_diff = (t_input) // (t_feats+1)

    img_new_height = 400
    img_new_width = int(image.shape[1] / float(image.shape[0]) * img_new_height)
    img_to_show = cv2.resize(image.copy(), (img_new_width,img_new_height))[:,:,::-1]
    #img_to_concat = np.zeros((400, 800, 3), np.uint8)
    img_to_concat = np.zeros((400, 400, 3), np.uint8)
    for cc in range(len(action_classes)):
        cur_cls_idx = action_classes[cc]
        act_str = act.ACTION_DESCRIPTION[action_classes[cc]]
        message = "%s:%%%.2d" % (act_str[:20], 100*probs[actor_idx, cur_cls_idx])
        for tt in range(t_feats):
            cur_cam = normalized_cmaps[actor_idx, tt,:,:, cur_cls_idx]
            cur_frame = input_frames[actor_idx, (tt+1) * index_diff, :,:,::-1]

            resized_cam = cv2.resize(cur_cam, (100,100))
            colored_cam = cv2.applyColorMap(resized_cam, cv2.COLORMAP_JET)

            overlay = cv2.resize(cur_frame.copy(), (100,100))
            overlay = cv2.addWeighted(overlay, 0.5, colored_cam, 0.5, 0)

            img_to_concat[cc*125:cc*125+100, tt*100:(tt+1)*100, :] = overlay
        cv2.putText(img_to_concat, message, (20, 13+100+125*cc), 0, 0.5, (255,255,255), 1)

    final_image = np.concatenate([img_to_show, img_to_concat], axis=1)
    return final_image[:,:,::-1]


if __name__ == '__main__':
    main()
