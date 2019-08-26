import numpy as np
import cv2
import imageio
import tensorflow as tf
import json

import os
import sys
import argparse
import time

#PCN Project related 
import object_detection.object_detector_pcn as obj
import action_detection.action_detector_pcn as act

DEBUG = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', type=str, required=False, default="")
    parser.add_argument('-d', '--display', type=str, required=False, default="True")
    args = parser.parse_args()
    display = (args.display == "True" or args.display == "true")

    video_path = args.video_path
    basename = os.path.basename(video_path).split('.')[0]
    out_vid_path = "./output_videos/%s_output.mp4" % basename
    out_img_path = "./output_videos/%s_output.jpg" % basename
    log_output_path = "./output_videos/{}_log_file.txt".format(basename)

    main_folder = './'

    #obj_detection_model =  'ssd_mobilenet_v2_coco_2018_03_29' 
    obj_detection_model =  'faster_rcnn_nas_coco_2018_01_28'

    obj_detection_graph = os.path.join("object_detection", "weights", obj_detection_model, "frozen_inference_graph.pb")
    
    print("Loading object detection model at %s" % obj_detection_graph)
    obj_detector = obj.Object_Detector(obj_detection_graph)

    tracker = obj.Tracker()

    #Frequency of action detection
    action_freq = 8

    print("Reading video file %s" % video_path)
    print('Running actions every %i frame' % action_freq)

    #Configures the opencv cap
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = tracker.timesteps
    frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #Sets the path for output video and initializes VideoWriter
    if not display:
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
    frames_count = 0
    while frames_count < frames_len:

        if frames_count == 0:
            #Creates a txt log file
            if DEBUG:
                log_file = open(log_output_path, "w+")
        else:
            if DEBUG:
                log_file = open(log_output_path, "a")
        
        return_value, current_video_frame = cap.read() 
        
        if return_value:
            frames_count += 1
            print("frames_count: %i" %frames_count)
        else:
            continue
        
        if DEBUG:
            log_file.write("frames_count: %i \n" %frames_count)

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

        if tracker.active_actors and frames_count % action_freq == 0:
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

            out_dict = act_detector.session.run(run_dict, feed_dict=feed_dict)

            probs = out_dict['pred_probs'] #Stores the confidence score (probaility) of all actions for EACH Actor in an numpy array

            #associate probs with actor ids
            TOP_NUMBER_OF_ACTIONS = 5
            for bbox in range(num_actors):

                #all action probabilities for an actor in the indexed sequence in ACTION_DESCRIPTION dictionary
                action_probs = probs[bbox] #numpy array

                #Sorts the indexed action according to probability in descending order
                ordered_action_probs = np.argsort(action_probs)[::-1] # returns a numpy array of the same shape

                cur_actor_id = tracker.active_actors[bbox]['actor_id']

                print("Person %i" % cur_actor_id)
                if DEBUG:
                    log_file.write("Person %i \n" % cur_actor_id)

                cur_results = []

                TOP_ACTIONS_DECTECTED = {}
                for i in range(TOP_NUMBER_OF_ACTIONS):
                    action_key = ordered_action_probs[i]
                    ''' int : [string, float]'''
                    TOP_ACTIONS_DECTECTED[action_key] = [act.ACTION_DESCRIPTION[action_key], action_probs[action_key]]
                
                #appends description and action probablity to current results.
                for action_key in TOP_ACTIONS_DECTECTED: 
                    action_detection_prob = TOP_ACTIONS_DECTECTED[action_key][1]
                    action_detection_description = TOP_ACTIONS_DECTECTED[action_key][0] 

                    print('\t {}: {:.3f}'.format(action_detection_description, action_detection_prob))
                    if DEBUG:
                        log_file.write('\t {}: {:.3f} \n'.format(action_detection_description, action_detection_prob))

                    cur_results.append((action_detection_description, action_detection_prob))

                #determines and updates action of actor in frame
                actor_action_in_frame_description, actor_action_in_frame_prob = eval_actor_action_in_frame(TOP_ACTIONS_DECTECTED)

                #updates and store the ovserall action history 
                tracker.update_all_actor_action_history(cur_actor_id, actor_action_in_frame_description)

                #check overall_action_history for any previous run/jog
                #if yes, override the actor_action_in_frame_description with run/jog
                '''if "run/jog" in tracker.all_actor_action_history[cur_actor_id] and actor_action_in_frame_description != 'run/jog':
                    print("OVERIDE >>>> run/jog")
                    if DEBUG:
                        log_file.write("OVERIDE >>>> run/jog \n")
                    actor_action_in_frame_description, actor_action_in_frame_prob = "run/jog", -99.0'''

                #Append actor_action_in_frame_description to cur_results
                if DEBUG:
                    log_file.write("\t Person's action is <{}> \n".format(actor_action_in_frame_description))
                cur_results.append((actor_action_in_frame_description, actor_action_in_frame_prob))

                prob_dict[cur_actor_id] = cur_results 
            
            if DEBUG:
                log_file.write("All current Person IDs: " + str(tracker.get_all_current_actor_id()) + " \n")

            if DEBUG:
                log_file.write("Pre-Cleanup >>  \n")
                for entry in tracker.all_actor_action_history.items():
                    log_file.write("\t Person No." + str(entry)[1:-1]  + " \n")


            tracker.cleanup_all_actor_action_history()

            if DEBUG:
                log_file.write("After-Cleanup >>  \n")
                for entry in tracker.all_actor_action_history.items():
                    log_file.write("\t Person No." + str(entry)[1:-1]  + " \n")

            action_tally, current_frame, action_untracked = tracker.get_all_action_tally_at_frame()
            if DEBUG:
                log_file.write("Action Tally {}, at frame {}. ".format(action_tally, current_frame) + " \n")
                log_file.write("Untracked Actions {}, at frame {}. ".format(action_untracked, current_frame) + " \n")
                log_file.write("\n")

            t4 = time.time(); print('action %.2f seconds' % (t4-t3))
            
        
        if frames_count > 16:
            out_img = visualize_detection_results(tracker.frame_history[-16], tracker.active_actors, prob_dict,frames_count)
            if display: 
                cv2.imshow('results', out_img[:,:,::-1])
                cv2.waitKey(10)
            else:
                writer.write(out_img)
                cv2.imwrite(out_img_path,out_img)
        
        if DEBUG:
            log_file.close()

    if not display:
        cap.release()
        writer.release()   

#returns a summary of detected actions within the current frame
def get_all_actions_in_frame():
    all_actions_out = []
    return all_actions_out

#evaluates overall action in a given frame
def eval_actor_action_in_frame(action_dict):
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
        top_action = ["run/jog", -99.0] #dummy variable -99.0 for probability float

    return top_action


np.random.seed(10)
COLORS = np.random.randint(0, 255, [1000, 3])
def visualize_detection_results(img_np, active_actors, prob_dict,frames_count):
    score_th = 0.30
    action_th = 0

    # copy the original image first
    disp_img = np.copy(img_np)
    H, W, C = img_np.shape

    #draw frame label
    draw_label(disp_img, 'Frame ' + str(frames_count), (0, 12), font=cv2.FONT_HERSHEY_SIMPLEX, font_size=0.5, font_weight=1, highlight=(0, 0, 0))

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
        #15/08/2019 MOD
        #action_summary = cur_act_results[-1][0]
        action_summary = cur_act_results[-1][0] if cur_act_results else "NO_ACTION"

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

def draw_label(image, text, top_left_coor, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=0.5, font_weight=1, highlight=(0, 0, 0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, font_weight)
    cv2.rectangle(image, top_left_coor, (top_left_coor[0] + text_width, top_left_coor[1] - text_height), highlight, cv2.FILLED)
    cv2.putText(image, text, (top_left_coor[0], top_left_coor[1]), font, font_size, (255, 255, 255), font_weight)
  

if __name__ == '__main__':
    main()