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
import library.util as util

LOG_FILE = True
MAX_NUM_ACTORS = 30
TIMESTEPS = 18
#Frequency of action detection
ACTION_FREQ = 8 

def main(video_src_path=""):
    parser = argparse.ArgumentParser(description="Takes in a GoPro Video in .MP4 format and outputs the detection in .pkl, along with an augmented video and a log file.")
    parser.add_argument('-v', '--video_path', type=str, required=False, default="")
    parser.add_argument('-f', '--folder_path', type=str, required=False, default="")
    args = parser.parse_args()

    if video_src_path != "":
        args.folder_path = video_src_path

    if args.video_path and not args.folder_path:
        video_path = args.video_path
        if is_video_valid(video_path):
            action_detection(video_path)
    if args.folder_path and not args.video_path: 
        folder_path = args.folder_path
        if os.path.isdir(folder_path):
            batch_action_detection(folder_path)
        else:
            print("#Error#'{}' is not a valid directory.".format(folder_path))

def batch_action_detection(folder_path):
    invalid_files = []
    for f in os.listdir(folder_path):
        video_path = os.path.join(folder_path,f)
        if is_video_valid(video_path): 
            action_detection(video_path)
        else:
            invalid_files.append(f)
    
    if invalid_files:
        print("--Invalid Files--")
        for f in invalid_files:
            print(f)
        print("----")

def is_video_valid(video_path) :
    if os.path.splitext(video_path)[1] == '.MP4' or os.path.splitext(video_path)[1] == '.mp4':
        return True  
    else:
        print("#Error#'{}' is not a valid video path. Video file must be of extension .MP4".format(video_path))
        return False
    

def action_detection(video_path):
    basename = os.path.basename(video_path).split('.')[0]
    out_vid_path = "{}/video_vis/{}.VIS.MP4".format(os.path.split(os.path.dirname(video_path))[0],basename)
    out_pkl_path = "{}/pkl/{}.ACT.pkl".format(os.path.split(os.path.dirname(video_path))[0],basename)
    out_log_path = "{}/log/{}.LOG.txt".format(os.path.split(os.path.dirname(video_path))[0],basename)
    
    main_folder = './'

    obj_detection_model =  'faster_rcnn_nas_coco_2018_01_28'
    obj_detection_graph = os.path.join("object_detection", "weights", obj_detection_model, "frozen_inference_graph.pb")
    
    print("Loading object detection model at %s" % obj_detection_graph)
    obj_detector = obj.Object_Detector(obj_detection_graph)

    tracker = obj.Tracker(timesteps=TIMESTEPS)
    action_out = util.Output()

    print("Reading video file %s" % video_path)
    print('Running actions every %i frame' % ACTION_FREQ)

    #Configures the opencv cap
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = tracker.timesteps
    frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #Sets the path for output video and initializes VideoWriter
    writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
    print("Writing output to %s" % out_vid_path)

    #Sets the action detector model & checkpoint
    act_detector = act.Action_Detector('soft_attn',timesteps=TIMESTEPS)
    ckpt_name = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'

    memory_size = act_detector.timesteps - ACTION_FREQ

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
            if LOG_FILE:
                log_file = open(out_log_path, "w+")
        else:
            if LOG_FILE:
                log_file = open(out_log_path, "a")
        
        return_value, current_video_frame = cap.read() 
        
        if return_value:
            frames_count += 1
            print("frames_count: %i" %frames_count)
        else:
            continue
        
        if LOG_FILE:
            log_file.write("frames_count: %i     | " %frames_count)

        #--Object Detection--#

        current_video_frame_expanded = np.expand_dims(current_video_frame, axis=0)
        t1 = time.time()

        #calling of the object detection functions.
        detection_list = obj_detector.detect_objects_in_np(current_video_frame_expanded)
        detection_info = [info[0] for info in detection_list]

        t2 = time.time(); print('obj det %.2f s' % (t2-t1))

        if LOG_FILE:
            log_file.write('obj det %.2f s  | ' % (t2-t1))

        #Tracker
        tracker.update_tracker(detection_info, current_video_frame)     
        t3 = time.time(); print('tracker %.2f s' % (t3-t2))
        num_actors = len(tracker.active_actors)

        if LOG_FILE:
            log_file.write('tracker %.2f s \n' % (t3-t2))

        #--Action detection--#
        if tracker.active_actors and frames_count % ACTION_FREQ == 0:
            probs = []

            cur_input_sequence = np.expand_dims(np.stack(tracker.frame_history[-ACTION_FREQ:], axis=0), axis=0)

            rois_np, temporal_rois_np = tracker.generate_all_rois()

            #Clips the number of actors to the maximum number of actors
            #MAX_NUM_ACTORS = 30

            if num_actors > MAX_NUM_ACTORS:
                num_actors = MAX_NUM_ACTORS
                rois_np = rois_np[:MAX_NUM_ACTORS]
                temporal_rois_np = temporal_rois_np[:MAX_NUM_ACTORS]

            feed_dict = {updated_frames:cur_input_sequence, # only update last #ACTION_FREQ frames
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
                if LOG_FILE:
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
                    if LOG_FILE:
                        log_file.write('\t {}: {:.3f} \n'.format(action_detection_description, action_detection_prob))

                    cur_results.append((action_detection_description, action_detection_prob))

                #determines and updates action of actor in frame
                actor_action_in_frame_description, actor_action_in_frame_prob = eval_actor_action_in_frame(TOP_ACTIONS_DECTECTED)

                #updates and store the ovserall action history 
                tracker.update_all_actor_action_history(cur_actor_id, actor_action_in_frame_description)

                #Append actor_action_in_frame_description to cur_results
                if LOG_FILE:
                    log_file.write("\t Person's action is <{}> \n".format(actor_action_in_frame_description))
                cur_results.append((actor_action_in_frame_description, actor_action_in_frame_prob))

                #stores all detection result of an actor
                prob_dict[cur_actor_id] = cur_results 
            
            if LOG_FILE:
                log_file.write("All current Person IDs: " + str(tracker.get_all_current_actor_id()) + " \n")
                log_file.write("Pre-Cleanup >>  \n")
                for entry in tracker.all_actor_action_history.items():
                    log_file.write("\t Person No." + str(entry)[1:-1]  + " \n")

            tracker.cleanup_all_actor_action_history()

            if LOG_FILE:
                log_file.write("After-Cleanup >>  \n")
                for entry in tracker.all_actor_action_history.items():
                    log_file.write("\t Person No." + str(entry)[1:-1]  + " \n")

            action_tally, current_frame, action_untracked = tracker.get_all_action_tally_at_frame()
            if LOG_FILE:
                log_file.write("Action Tally {}, at frame {}. ".format(action_tally, current_frame) + " \n")
                log_file.write("Untracked Actions {}, at frame {}. ".format(action_untracked, current_frame) + " \n")

            t4 = time.time(); print('action det %.2f s' % (t4-t3))
            if LOG_FILE:
                log_file.write('action det %.2f s \n' % (t4-t3))
                log_file.write("\n")

        elif frames_count > ACTION_FREQ*2 and frames_count % ACTION_FREQ  == 0: #ie no active actors detected in frame
            action_tally, current_frame, action_untracked = tracker.get_all_action_tally_at_frame()
            if LOG_FILE:
                log_file.write("Action Tally {}, at frame {}. ".format(action_tally, current_frame) + " \n")
                log_file.write("Untracked Actions {}, at frame {}. ".format(action_untracked, current_frame) + " \n")

        if frames_count == ACTION_FREQ:
            if LOG_FILE:
                log_file.write("**********Note!!!The first action detection will not be written to the pkl file.*********"+ " \n")
                log_file.write("\n")

        #appends a result every x frame. x = ACTION_FREQ
        if frames_count > ACTION_FREQ*2 and frames_count % ACTION_FREQ == 0:
            action_tally, current_frame, action_untracked = tracker.get_all_action_tally_at_frame()
            action_out.add(action_tally)
        else:
            action_out.add({})

        #argument for frame history is changed from -16 to -ACTION_FREQ*2
        if frames_count > ACTION_FREQ*2: #Visualize every frame after (ACTION_FREQ*2) 
            out_img = visualize_detection_results(tracker.frame_history[-ACTION_FREQ*2], tracker.active_actors, prob_dict,frames_count)
            writer.write(out_img)  

        if LOG_FILE:
            log_file.close()

    action_out.dump_pickle(out_pkl_path)
    cap.release()
    writer.release()

#returns a summary of detected actions within the current frame
def get_all_actions_in_frame():
    all_actions_out = []
    return all_actions_out

#evaluates overall action in a given frame
def eval_actor_action_in_frame(action_dict):
    top_action = ["NULL", 0]
    for action_key, action_description_prob in action_dict.items():
        if action_description_prob[1] > top_action[1]:
            top_action = action_description_prob
    return top_action

##-----------visualization------------##
np.random.seed(10)
COLOR_SIZE = 50
COLORS = np.random.randint(0, 255, [COLOR_SIZE, 3])

#visualizer
def visualize_detection_results(img_np, active_actors, prob_dict,frames_count):
    score_th =  0 #0.30
    
    COLOR_COUNTER = 0

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

        COLOR_COUNTER += 1

        try:
            cur_box, cur_score, cur_class = cur_actor['all_boxes'][-ACTION_FREQ*2], cur_actor['all_scores'][0], object_class
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

        label = "P"
        message = '%s_%i:%.2f' % (label, actor_id,conf)

        action_summary = cur_act_results[-1][0] if cur_act_results else "NO_ACTION"

        if COLOR_COUNTER < COLOR_SIZE:
            color = COLORS[COLOR_COUNTER]
        else:
            COLOR_COUNTER = 0
            color = COLORS[COLOR_COUNTER]


        cv2.rectangle(disp_img, (left,top), (right,bottom), color.tolist(), 3)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))

        #position and writes the overall action message.
        cv2.rectangle(disp_img, (left, top-int(font_size*40)-60), (right,top-int(font_size*40)), color.tolist(), -1)
        cv2.putText(disp_img, action_summary, (left, top-40), 0, font_size*2, (np.array([255,255,255])-color).tolist(), 1)

        #position and writes the label, actor id and confidence 
        cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), color.tolist(), -1)
        cv2.putText(disp_img, message, (left, top-12), 0, font_size, (np.array([255,255,255])-color).tolist(), 1)

    return disp_img

def draw_label(image, text, top_left_coor, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=0.5, font_weight=1, highlight=(0, 0, 0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, font_weight)
    cv2.rectangle(image, top_left_coor, (top_left_coor[0] + text_width, top_left_coor[1] - text_height), highlight, cv2.FILLED)
    cv2.putText(image, text, (top_left_coor[0], top_left_coor[1]), font, font_size, (255, 255, 255), font_weight)
  
if __name__ == '__main__':
    main()