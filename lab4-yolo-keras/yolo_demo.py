'''
Program that reads in a video stream and outputs an object detection stream

Note : use your python virtual env : 
 'source  ~/.virtualenvs/coursera_env/bin/activate' (home)
 'source  /root/python3_env/bin/activate' (nimbix)
'''

# add imports here ..
import argparse as ap
import os
import numpy as np
import cv2
import sys
import re
import json
import shutil

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


from yolo_utils import read_classes, read_anchors
import yad2k.models.keras_yolo as yad2k
import random
import colorsys

# Globals
MODEL_PATH = "./model_data/yolo.h5"


def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))


class yolo_demo(BaseException):
    """The fully-connected neural network model."""

    def __init__(self, sess, _input_stream=None, _class_names=None, _anchors=None,
            image_shape=(720.,1280.), 
            batch_size=64,
            frame_stride=1):
        
        self.sess = sess
        self.batch_size = batch_size
        self.frame_stride = frame_stride
        self.image_shape = image_shape # in future, auto set this ...
        self.yolo_model_image_shape = (608.0,608.0)
        self.yolo_model = None
        self.yolo_outputs = None
        self.scores = None
        self.boxes_xy = None
        self.classes = None
        self._output_dir = None
        self._audit_mode = False
        self._input_stream = _input_stream
        self._class_names = _class_names
        self._anchors = _anchors
        self._retrain_file = None
        self.rotation = self.get_rotation()

    # Setters/Getters
    def input_stream(self,val=None) :
        if(val == None) : # get
            if(self._input_stream == None) :
                nprint("Error self._input_stream not set")
                return None
            return self._input_stream
        else : # set
            self._input_stream = val

    def class_names(self,val=None) :
        if(val == None) : # get
            if(self._class_names == None) :
                nprint("Error self._class_names not set")
                return None
            return self._class_names
        else : # set
            self._class_names = val
 
    def anchors(self,val=None) :
        if(not(isinstance(val, np.ndarray))) : # get
            if(not(isinstance(self._anchors,np.ndarray))) :
                nprint("Error self._anchors not set")
                return None
            return self._anchors
        else : # set
            self._anchors = val

    def retrain_file(self,val=None) :
        if(val == None) : # get
            if(self._retrain_file == None) :
                nprint("Error self._retrain_file not set")
                return None
            return self._retrain_file
        else : # set
            self._retrain_file = val

    def output_dir(self,val=None, overwrite=False) :

        if(overwrite == True and val != None) :
            nprint("Removing old output dir")
            if os.path.exists(val) :
                #os.remove(input)
                shutil.rmtree(val)

        if(val == None) :
            if(self._output_dir == None) :
                nprint("Error self._output_dir not set")
                return None
            return self._output_dir
        else :
            if not os.path.exists(val):
                os.makedirs(val)
            self._output_dir = val




    def audit_mode(self,input=None) :
        if(input == None) :
            if(self._audit_mode == None) :
                nprint("Error self._audit_mode not set")
                return None
            return self._audit_mode
        else :
            self._audit_mode = input

    # end Setters / Getters

    # Filename manipulations
    def append_output_path(self,filein) :
        return self.output_dir()+"/"+filein




    # Image Manipulations
    def get_rotation(self):
        rv=0
        if(self.input_stream() != None) :
            if( re.search("\.mov", self.input_stream())) :
                rv = 270
            elif( re.search("\.mp4", self.input_stream())) :
                rv = 0
            else :
                nprint("warning, unhandled file extenstion.  Currently support *.mov/*.mp4")
                rv = 0
            nprint("Rotation set to : " + str(rv) + " degrees")
        else :
            nprint("Input Stream not set, returning None for rotations")
            rv = None
        return rv
    
    def load_and_build_graph(self) :
        nprint("Loading Model")
        self.yolo_model = load_model(MODEL_PATH)
        nprint("Instantiating graph (yolo_head)")
        self.yolo_outputs = yad2k.yolo_head(self.yolo_model.output, self.anchors(), len(self.class_names()))
        nprint("Loading yolo eval.  Final part of yolo that perform non max suppression and scoring")
        self.scores, self.boxes_xy, self.classes = self.yolo_eval() # sets self.scores, self.boxes, self.classes structures

    def print_model_summary(self):
        print(self.yolo_model.summary())



    def shade_image_19_19(self) :
        '''
        stub for image mod ....
        '''
        a=1

    def retrain(self):
        '''
        stub for training step ...
        '''
        

        # 1. create and modify original model
        model_full, model_final_wloss = self.create_model()

        #3. add a new layer per your data set requirements
        #4. prepare X/Y
        image_box_dict = self.parse_train_data() # return np array of processed images // boxes

        # Convert image_box_class dictionary to final data structures, fully formed np.arrays! X , Y
        (image_X, labels_Y) = self.create_labels(image_box_dict,self.anchors())

        ##5. retrain
        '''
        retrain/fine-tune the model
        logs training with tensorboard
        saves training weights in current directory
        best weights according to val_loss is saved as trained_stage_3_best.h5
        '''
        model_final_wloss.compile(
            optimizer='adam', loss={
               'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.


        logging = TensorBoard()
        checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                    save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

        nprint("Fitting Model ")
        model_final_wloss.fit(x=[image_X, labels_Y],
                  y=np.zeros(len(image_X)),
                  validation_split=0.1,
                  batch_size=32,
                  epochs=30,
                  callbacks=[logging])
        nprint("Training Complete ")

        model_final_wloss.save_weights('trained_stage_2.h5')

        nprint("Fitting Model Phase 2")
        model_final_wloss.fit(x=[image_X, labels_Y],
                              y=np.zeros(len(image_X)),
                              validation_split=0.1,
                              batch_size=32,
                              epochs=30,
                              callbacks=[logging])
        nprint("Training Complete ")

        model_final_wloss.save_weights('trained_stage_3.h5')




    def yolo_loss(self,
                  args,
                  anchors,
                  num_classes,
                  rescore_confidence=False,
                  print_loss=True) :

        (yolo_output_layer, labels_tensor_Y) = args

        # Hyperparameters
        prediction_scale_factor = 1
        no_class_scale_factor = 1
        class_scale_factor = 5
        coordinates_scale_factor = 1


        num_anchors = len(self.anchors())

        # Generate Predictions
        pred_xy, pred_wh, pred_confidence, pred_class_prob = yad2k.yolo_head(yolo_output_layer, self.anchors(), len(self.class_names()))

        yolo_output_shape = K.shape(yolo_output_layer)

        # Restructure predictions so it will match labels_Y dimensions
        # should be (None, 19,19,len(anchors),(1+4+len(classes))

        #labels_pred_Y = K.reshape(yolo_output_layer, [
        #    -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
        #    num_classes + 5
        #])
        labels_pred_Y = labels_tensor_Y

        # Use Keras Indexing .. nice
        label_indicator_box = labels_pred_Y[..., 0:1]
        no_label_indicator_box = -1  * (label_indicator_box-1) # bit inversion here ..0->1, 1->0
        label_box_xy = labels_pred_Y[..., 1:3]
        label_box_wh = labels_pred_Y[..., 4:5]
        label_class = labels_pred_Y[..., 5:5+len(self.class_names())]

        # Coordinate Loss xy
        coordinate_loss_xy = coordinates_scale_factor * label_indicator_box * K.square( label_box_xy - pred_xy)
        #coordinate_loss_xy = K.print_tensor(coordinate_loss_xy, message="coordinate_loss_xy is: ")
        #label_indicator_box = K.print_tensor(label_indicator_box, message="label_indicator_box is: ")
        coordinate_loss_xy = K.sum(coordinate_loss_xy)

        # Coordinate Loss wh
        coordinate_loss_wh = coordinates_scale_factor * label_indicator_box * K.square( K.sqrt(label_box_wh) - K.sqrt(pred_wh))
        coordinate_loss_wh = K.sum(coordinate_loss_wh)

        # Class Loss
        class_loss = class_scale_factor * label_indicator_box * K.square( label_class-pred_class_prob)
        class_loss = K.sum(class_loss)

        no_label_indicator_box = K.print_tensor(no_label_indicator_box, message="no_label_indicator_box is: ")
        no_class_loss = no_class_scale_factor * no_label_indicator_box * K.square( label_class-pred_class_prob)
        no_class_loss = K.sum(no_class_loss)

        # Prediction Loss
        prediction_loss = K.square( label_indicator_box - pred_confidence )
        prediction_loss = K.sum(prediction_loss)

        total_loss = coordinate_loss_xy + coordinate_loss_wh + class_loss + no_class_loss + prediction_loss
        if print_loss:
            total_loss = tf.Print(
            total_loss, [
                    total_loss,coordinate_loss_xy, coordinate_loss_wh, class_loss, no_class_loss, prediction_loss],
            message='total_loss,coordinate_loss_xy, coordinate_loss_wh, class_loss, no_class_loss, prediction_loss:')


        return total_loss




    def create_model(self, load_pretrained=True, freeze_body=True):
        '''
        returns the body of the model and the model

        # Params:

        load_pretrained: whether or not to load the pretrained model or initialize all weights

        freeze_body: whether or not to freeze all weights except for the last layer's

        # Returns:

        model_body: YOLOv2 with new output layer

        model: YOLOv2 with custom loss Lambda layer

        '''

        # Preloaded yolo uses 19x19 detector
        detectors_mask_shape = (19, 19, 5, 1) # 1 / 0 .. Pc value
        matching_boxes_shape = (19, 19, 5, 5) # boxes and class index value : class index needs to be explode into one hot

        matching_boxes_shape = (19, 19, 5, 5) # boxes and class index value : class index needs to be explode into one hot

        # Create model input layers.
        image_input = Input(shape=(608, 608, 3))
        
        boxes_input = Input(shape=(None, 5))
        detectors_mask_input = Input(shape=detectors_mask_shape)
        matching_boxes_input = Input(shape=matching_boxes_shape)
        labels_Y = Input(shape=(19,19,len(self.anchors()),1+4+len(self.class_names())))

        # Create model body and remove last layer.
        yolo_model = load_model(MODEL_PATH)
        # print(yolo_model.summary())

        #____________________________________________________________________________________________________
        #batch_normalization_22 (BatchNor (None, 19, 19, 1024)  4096        conv2d_22[0][0]                  
        #____________________________________________________________________________________________________
        #leaky_re_lu_22 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_22[0][0]     
        #____________________________________________________________________________________________________
        #conv2d_23 (Conv2D)               (None, 19, 19, 425)   435625      leaky_re_lu_22[0][0]             
        #====================================================================================================
        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

        if freeze_body:
            for layer in topless_yolo.layers:
                layer.trainable = False

        filters = len(self.anchors())*(5+len(self.class_names()))
        final_layer = Conv2D(filters, (1, 1), activation='linear',name='conv2d_final')(topless_yolo.output)

        model_final = Model(topless_yolo.input, final_layer)
        print(model_final.summary())


        # Place model loss on CPU to reduce GPU memory usage.
        with tf.device('/cpu:0'):
            # TODO: Replace Lambda with custom Keras layer for loss.
            model_loss = Lambda(
                self.yolo_loss,
                output_shape=(1, ),
                name='yolo_loss',
                arguments={'anchors': self.anchors(),
                           'num_classes': len(self.class_names())}
            )([  model_final.output, labels_Y ])

            model_final_wloss = Model([model_final.input, labels_Y], model_loss)
            print(model_final_wloss.summary())

        #return model_final, model_final_wloss
        return model_final, model_final_wloss

    def create_labels(self, image_box_dict, anchors):
        '''
        Precompute detectors_mask and box_data and one hot class vector for training.
        These will be concatenated Pc,x,y,w,h,c1..cN
        As example, for my 19x19 grid, will return a 19x19x8 tensor for a 3 class example.
        In a followon step, I will map to anchor boxes.  Training step assumes I wont have overlapping 
        anchor boxes ...

        '''


        """Find detector in YOLO where ground truth box should appear.
    
        Parameters
        ----------
        image_box_dict : List of image / box labels
        
        image_size : array-like
            List of image dimensions in form of h, w in pixels.
    
        Returns
        -------
        detectors_mask : array
            0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
            that should be compared with a matching ground truth box.
        matching_true_boxes: array
            Same shape as detectors_mask with the corresponding ground truth box
            adjusted for comparison with predicted parameters at training time.
        """

        # Create a loop to and initially assign all zeros to grid
        # Then modify the grid cells that have boxes


        height, width = self.yolo_model_image_shape
        num_anchors = len(anchors)
        # Downsampling factor of 5x 2-stride max_pools == 32.
        # TODO: Remove hardcoding of downscaling calculations.
        assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        conv_height = int(height // 32)
        conv_width = int(width // 32)


        label_detectors_mask = np.zeros(
            (conv_height, conv_width, num_anchors, 1), dtype=np.float32)

        label_boxes = np.zeros(
            (conv_height, conv_width, num_anchors, 4),
            dtype=np.float32)

        label_classes = np.zeros(
            (conv_height, conv_width, num_anchors, len(self.class_names())),
            dtype=np.float32)

        image_X = []
        # This loop fills in the label (Y values) with information about the boxes
        labels_Y = []
        for img,box_list in image_box_dict.values() :
            image_X.append(img)
            for box in box_list :
                (x,y) = get_grid_location(box,conv_height,conv_width)
                k = select_anchorbox_index(box,self.anchors(),conv_width,conv_height) # select anchorbox index with the best IOU

                # Now write Labels ....

                # TODO : understand why they log the box ...
                # ok found it! --> https://github.com/pjreddie/darknet/blob/master/src/box.c
                # basically there is an encode box / decode box.  This is a better implementation than below .

                if(k != None ) :
                    label_detectors_mask[x, y, k] = 1
                    label_boxes[x, y, k] = encode_box(box,conv_width,conv_height,self.anchors()[k])
                    # Set the Class One hot vector
                    class_index = box[4]
                    class_one_hot = np.zeros(len(self.class_names()))
                    class_one_hot[class_index] = 1
                    label_classes[x,y,k] = class_one_hot
            labels_Y.append(np.concatenate((label_detectors_mask,label_boxes,label_classes),axis=3))


        # Cast outer lists to ndarrays
        image_X = np.asarray(image_X, dtype=float)
        labels_Y = np.asarray(labels_Y, dtype=float)

        return (image_X,labels_Y)

    def parse_train_data(self) :
        '''
        Reads in json file of this format
                #{"image_id": "orig-2.jpg", "category_id": 0, "category_name": "person", "bbox_xymin_xymax": [319.5470275878906, 909.2236328125, 382.7581787109375, 991.5352172851562], "score": 0.6394317746162415},
        and parses it, such that the returned object is a data structure of the type
        rv['image_id'] => List(img:ndarray, List(box data : [xc,yc,w,h,class])
        There will be a variable # of boxes per image passed back.  Followon routines will standardize into the 19x19 grid
        :return: 
        '''
        boxes_image_ptr_json =open(self._retrain_file).read()
        data = json.loads(boxes_image_ptr_json)

        rv = {}
        # Bui
        for i in data :

            # Record initial image size
            image_np = cv2.imread(i["image_id"])
            (r,c,rgb) = image_np.shape

            box_list = i["bbox_xymin_xymax"] + [i["category_id"]]
            box_xywhc = box_min_max_to_xcycwh(box_list , r, c)

            if i["image_id"] in rv.keys():
                rv[i["image_id"]][1].append( box_xywhc )
            else:
                # Lets Scale these images here ... and load into an np array
                # TODO : modify self.yolo_model_image_shape to return int instead of float
                image_np = cv2.resize(image_np,(608,608),cv2.INTER_CUBIC)
                image_np = np.array(image_np, dtype=np.float)
                image_np = image_np/255.0
                rv[i["image_id"]] = [image_np, [box_xywhc]]


        # Convert Everything to numpy arrays ...
        #boxes =np.array([np.array( box_xy[i] + box_wh[i] +[ class_ary[i] ]) for i, box in enumerate(box_ary)])

        return rv



    def set_image_shape(self):
        # open input stream just once and cache ....
        cap = cv2.VideoCapture(self.input_stream())
        ret, frame = cap.read()
        nprint("Setting self.image_shape to {0}".format(frame.shape))
        self.image_shape = frame.shape
        cap.release()
        cv2.destroyAllWindows()

    def get_image_shape(self, dim="height"):
        rv = None
        if(self.image_shape == None) :
            self.set_image_shape()

        if(dim == "height") :
            rv = self.image_shape[0]
        elif (dim == "width") :
            rv = self.image_shape[1]
        elif (dim == "channels") :
            rv = self.image_shape[2]
        else :
            nprint("error")
        return rv


    def yolo_eval(self,  max_boxes=10, score_threshold=.6, iou_threshold=.5):
        """
        Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
        
        Arguments:
        yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                        box_confidence: tensor of shape (None, 19, 19, 5, 1)
                        box_xy: tensor of shape (None, 19, 19, 5, 2)
                        box_wh: tensor of shape (None, 19, 19, 5, 2)
                        box_class_probs: tensor of shape (None, 19, 19, 5, 80)
        image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
        max_boxes -- integer, maximum number of predicted boxes you'd like
        score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
        
        Returns:
        scores -- tensor of shape (None, ), predicted score for each box
        boxes -- tensor of shape (None, 4), predicted box coordinates
        classes -- tensor of shape (None,), predicted class for each box
        """

        # Retrieve outputs of the YOLO model
        # This output contains the boxes for the entire batch
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_outputs

        # Convert boxes to be ready for filtering functions
        box_corners_xy = yolo_boxes_to_corners(box_xy, box_wh)

        # use a for loop, and lets see what happens ...
        #trainable=False
        # all_scores = tf.get_variable(name="all_scores",shape=(),initializer=tf.zeros_initializer(),dtype=tf.float32,collections=[tf.GraphKeys.LOCAL_VARIABLES])

        all_boxes_xy = []
        all_scores = []
        all_classes = []

        for m in range(0,self.batch_size) :
            tmp_box_class_probs = tf.slice(box_class_probs, [m,0,0,0,0], [1,19,19,5,80], name=None)
            tmp_boxes_xy = tf.slice(box_corners_xy, [m,0,0,0,0], [1,19,19,5,4], name=None)
            tmp_confidence = tf.slice(box_confidence, [m,0,0,0,0], [1,19,19,5,1], name=None)


            tmp_scores, tmp_boxes_xy, tmp_classes = self.yolo_filter_boxes(box_class_probs=tmp_box_class_probs,box_confidence=tmp_confidence,boxes_xy=tmp_boxes_xy,threshold=iou_threshold)

            # Scale boxes back to original image shape.
            tmp_scaled_boxes_xy = scale_boxes(tmp_boxes_xy, self.image_shape)

            # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
            ts,tb,tc = self.yolo_non_max_suppression(tmp_scores, tmp_scaled_boxes_xy, tmp_classes, max_boxes, iou_threshold)

            all_scores.append(ts)
            all_boxes_xy.append(tb)
            all_classes.append(tc)

        return all_scores, all_boxes_xy, all_classes

    def yolo_filter_boxes(self, box_confidence, boxes_xy, box_class_probs, threshold = .6):
        """Filters YOLO boxes by thresholding on object and class confidence.
        
        Arguments:
        box_confidence -- tensor of shape (?,19, 19, 5, 1)
        boxes -- tensor of shape (?,19, 19, 5, 4)
        box_class_probs -- tensor of shape (?,19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
        
        Sets :
        all_scores -- tensor of shape (None,), containing the class probability score for selected boxes
        all_boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        all_classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
        
        Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
        For example, the actual output size of scores would be (10,) if there are 10 boxes.
        """

        # Step 1: Compute box scores
        ### START CODE HERE ### (≈ 1 line)
        box_scores = box_confidence * box_class_probs
        # should be 19x19x5x80 ...
        #print(box_scores.shape)
        #assert(box_scores.shape == (19,19, 5, 80))
        ### END CODE HERE ###

        # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
        # Basically pick the highest probablity category for each anchor box and each grid location
        ### START CODE HERE ### (≈ 2 lines)
        box_classes = K.argmax(box_scores, axis=-1)
        box_class_scores = K.max(box_scores, axis=-1)
        ### END CODE HERE ###

        # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
        # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
        ### START CODE HERE ### (≈ 1 line)
        filtering_mask = box_class_scores > threshold
        ### END CODE HERE ###

        # Step 4: Apply the mask to scores, boxes and classes
        ### START CODE HERE ### (≈ 3 lines)
        all_scores = K.max(tf.boolean_mask(box_scores,filtering_mask,name='score_mask'),axis = -1)
        all_boxes_xy = tf.boolean_mask(boxes_xy,filtering_mask,name='score_mask')
        all_classes = tf.boolean_mask(box_classes,filtering_mask,name='score_mask')

        return all_scores, all_boxes_xy, all_classes

    def yolo_non_max_suppression(self, scores, boxes_xy, classes, max_boxes = 10, iou_threshold = 0.5):
        """
        Applies Non-max suppression (NMS) to set of boxes
        
        Arguments:
        scores -- tensor of shape (None,), output of yolo_filter_boxes()
        boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
        classes -- tensor of shape (None,), output of yolo_filter_boxes()
        max_boxes -- integer, maximum number of predicted boxes you'd like
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
        
        Returns:
        scores -- tensor of shape (, None), predicted score for each box
        boxes -- tensor of shape (4, None), predicted box coordinates
        classes -- tensor of shape (, None), predicted class for each box
        
        Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
        function will transpose the shapes of scores, boxes, classes. This is made for convenience.
        """

        max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
        K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor

        # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
        # Remap due to tensorflow nms requirements
        boxes_yx = K.concatenate([
            boxes_xy[..., 1:2],  # y_min
            boxes_xy[..., 0:1],  # x_min
            boxes_xy[..., 3:4],  # y_max
            boxes_xy[..., 2:3]   # x_max
        ])

        boxes_xy = K.print_tensor(boxes_xy, message="boxes_xy is: ")
        boxes_yx = K.print_tensor(boxes_yx, message="boxes_yx is: ")

        nms_indices = tf.image.non_max_suppression( boxes_yx, scores, max_boxes, iou_threshold)

        # Use K.gather() to select only nms_indices from scores, boxes and classes
        scores = K.gather(scores, nms_indices)
        boxes_xy = K.gather(boxes_xy, nms_indices)
        classes = K.gather(classes, nms_indices)

        return scores, boxes_xy, classes



    def preprocess_image_cpu(self, image):
        '''
        
        :param image: a 4D tensor.  (batch_size,H,W,RGB)
        
        :config self.rotation - degrees to rotate image.  Did this because opencv rotates the mpg files for some reason
        
        :return: 
            rv_rotated_image : a 4D tensor with the original image rotated (not scaled) (batch_size,H,W,RGB)
            rv_scaled_image : a 4D tensor with the original image rotated and scaled down.  
        '''

        # Make sure model image size are ints ...
        model_image_size = tuple(map( (lambda x : int(x)), self.yolo_model_image_shape))
        examples,rows,cols,rgb = image.shape

        nprint("Yolo model image size requirement = {}".format(model_image_size))
        nprint("Raw  input image size             = {} {} {}".format(rows,cols,rgb))

        # Opencv method to rotate images
        M = cv2.getRotationMatrix2D((cols/2,rows/2),self.rotation,1)

        image_size_tuple = (cols,rows)
        if(self.rotation % 180 == 90) :
            image_size_tuple = (rows,cols)



        rv_rotated_image = np.zeros((examples,rows,cols,rgb))
        rv_scaled_image = np.zeros((examples,model_image_size[0],model_image_size[1],rgb))

        for i in range(0,examples) :
            cur_image = image[i]

            #                                            (w,   h    )
            rotated_image1 = cv2.warpAffine(cur_image,M,(image_size_tuple))
            resized_image1 = cv2.resize(rotated_image1,model_image_size,cv2.INTER_CUBIC)
            resized_image2 = np.array(resized_image1, dtype='float32')
            resized_image3 = resized_image2 / 255.0
            rv_rotated_image[i] = rotated_image1

            rv_scaled_image[i] = resized_image3
            #plot_image(resized_image3)

        return rv_rotated_image,rv_scaled_image


    def draw_boxes(self, image_data, out_scores, out_boxes_xy, out_classes, colors):
        '''
        
        :param image_data: h,w,c tensor
        :param out_scores: 
        :param out_boxes: 
        :param out_classes: 
        :param colors: 
        :return:  image_data: h,w,c tensor
        '''
        # assert(image_data.shape == (1920,1920,3))
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names()[c]
            box_xy = out_boxes_xy[i]
            score = out_scores[i]

            nprint("score = {0}, boxes={1}".format(score,box_xy))
            label = '{} {:.2f}'.format(predicted_class, score)

            # label_size = draw.textsize(label, font)

            left, top, right, bottom  = box_xy
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))

            bottom = min(image_data.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image_data.shape[1], np.floor(right + 0.5).astype('int32'))
            nprint("i={} c={} label={} xmin,ymin={} xmax,ymax={}".format(i,c,label, (left, top), (right, bottom) ))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_data,predicted_class,(left,top), font, 1,colors[c],2,cv2.LINE_AA)
            image_data = cv2.rectangle(image_data,(left,top),(right,bottom),colors[c],3)

        # draw a 100 x 100 grid for obj labelling ...
            image_data = self.draw_grid(image_data, 100)

        return image_data

    # Probably doesnt need to be part of this class.  At some point rearrage this better
    def draw_grid(self, image, xy_grid_size) :
        (rows,cols,depth) = image.shape
        x_max_ind = int(cols/xy_grid_size)
        y_max_ind = int(rows/xy_grid_size)
        color_tuple = (127,255,0)

        for x in range(0,x_max_ind) :
            for y in range(0,y_max_ind) :
                x1 = x*xy_grid_size
                y1 = y*xy_grid_size
                x2 = x*xy_grid_size + (xy_grid_size-1)
                y2 = y*xy_grid_size + (xy_grid_size-1)
                image = cv2.rectangle(image,(x1,y1),(x2,y2),color_tuple,2)
        return image

    def play_video(self, video_to_play=None, num_frames=90) :
        if(video_to_play == None) :
            video_to_play = self.input_stream()
        nprint("Starting Video : {0}".format(video_to_play) )

        cap = cv2.VideoCapture(video_to_play)
        
        frame_cnt=0
        while(cap.isOpened() and frame_cnt < num_frames):
            # Capture frame-by-frame
            ret, frame = cap.read()
            #assert(frame.dtype.name == 'uint8')
        
            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_cnt += 1
        
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def write_coco_labels(self, image_fn, scores, boxes_xy, classes) :
        #COCO frmat
        #[{
        #"image_id" : int, "category_id" : int, "bbox" : [x,y,width,height], "score" : float,
        #}]
        labels_file  = self.output_dir()+"/"+"labels.json"

        f = open(labels_file, 'a')
        #loop thru all the boxes in the boxes array!
        for idx in range(0,len(boxes_xy)) :
            json_dict = {}
            json_dict['image_id'] = image_fn
            json_dict['category_id'] = int(classes[idx])
            json_dict['category_name'] = self._class_names[classes[idx]]
            json_dict['bbox_xymin_xymax'] = boxes_xy[idx].tolist()
            json_dict['score'] = float(scores[idx])
            #f.write(json_dict)
            str = json.dumps(json_dict)
            f.write(str + ",\n")
        f.close()


    def process_video(self, output_filename="tmp.mov"):
        """
        Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
        
        Arguments:
        
        Returns:
        
        """

        output_filename = self.append_output_path(output_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)
        nprint("Saving new movie here {}".format(output_filename))

        # Read in video stream
        cap = cv2.VideoCapture(self.input_stream())

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mpv4')

        out = cv2.VideoWriter(output_filename, fourcc, 10.0, (self.get_image_shape("width"),self.get_image_shape("height")), True)


        loop_cnt = 0
        num_batches_to_process = 1
        frame = np.ones((16,self.get_image_shape("height") ,self.get_image_shape("width"),self.get_image_shape("channels")),dtype="uint8")
        while(cap.isOpened() and loop_cnt < num_batches_to_process ):
            # frame is an ndarray of nhwc
            for j in range(0,self.batch_size) :
                ret,  frame[j] = cap.read()
                for i in range(0, self.frame_stride) :
                    ret, dummy_frame = cap.read()


            # Load an color image in grayscale
            # frame = cv2.imread(self.input_stream)
            nprint("loop count {} : Frame shape = {}".format(loop_cnt,frame.shape))

            # preprocess returns a 4D tensor (examples, x, y, RGB)

            image_rotate, image_data = self.preprocess_image_cpu(frame)

            # can plot image data b/c of added dimension ... plot_image(image_data)

            # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
            out_boxes_xy, out_scores, out_classes = self.sess.run([self.boxes_xy, self.scores, self.classes],feed_dict={self.yolo_model.input: image_data , K.learning_phase(): 0})

            # Print predictions info
            for i in range(0,self.batch_size):
                print('Found {} boxes for current batch {}'.format(len(out_boxes_xy[i]),i))

            # Generate colors for drawing bounding boxes.
            colors = generate_colors(self.class_names())

            # Create a Video  .... 
            for m in range(0,self.batch_size):
                # Draw bounding boxes on the image file
                #plot_image(image_rotate[m].astype('uint8'))
                image_modified = self.draw_boxes(image_data=image_rotate[m], out_scores=out_scores[m], out_boxes_xy=out_boxes_xy[m], out_classes=out_classes[m], colors=colors)
                #plot_image(image_modified.astype('uint8'))

                # plot_image(image_modified)
                # Display the resulting frame
                im_uint8 = image_modified.astype('uint8')
                cv2.imshow('frame',im_uint8)
                nprint("Output image shape {}".format(im_uint8.shape))
                out.write(im_uint8)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            # If Audit Mode Set, save stills in output dir for a future training run!  .... 

            if(self.audit_mode() == True ) :
                nprint("Entering Audit Mode")
                for m in range(0,self.batch_size):

                    image_modified = self.draw_boxes(image_data=image_rotate[m], out_scores=out_scores[m], out_boxes_xy=out_boxes_xy[m], out_classes=out_classes[m], colors=colors)
                    # plot_image(image_modified)
                    # Display the resulting frame
                    im_uint8 = image_modified.astype('uint8')
                    fn_audit = self.output_dir()+"/"+"audit-"+ str(m) + ".jpg"
                    fn_orig  = self.output_dir()+"/"+"orig-"+ str(m) + ".jpg"

                    cv2.imwrite( fn_orig,  image_rotate[m] );
                    cv2.imwrite( fn_audit, im_uint8 );
                    self.write_coco_labels(fn_orig,out_scores[m],out_boxes_xy[m], out_classes[m])


            loop_cnt += 1

        out.release()
        cap.release()
        cv2.destroyAllWindows()



#################### End Class yolo_demo ################################



#################### End Class threaded_yolo ################################

class SmartFormatterMixin(ap.HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return ap.HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ap.RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def _parser():
    parser = ap.ArgumentParser(description='Desc.: Deep PDE. '
                               'https://arxiv.org/abs/1706.04702v1',
                               formatter_class=CustomFormatter)

    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='S|Batch size. Default: %(default)s')

    parser.add_argument(
        '--neuron_size', type=int, default=100,
        help='S|PDE network size. '
        'Default: %(default)s')

    parser.add_argument(
        '--time_steps', type=int, default=20,
        help='S|Algorithm time steps.'
        'Default: %(default)s')

    parser.add_argument(
        '--maxsteps', type=int, default=4000,
        help='S|Number of steps to run preferably divisible by 100. '
        'Default: %(default)s')

    parser.add_argument(
        '--device', action='store', nargs='?', type=str.lower, const='gpu',
        choices=['gpu', 'cpu'], default='gpu',
        help='S|Run on GPU or CPU. Seems to be faster on cpu. '
        'Default: %(default)s')

    parser.add_argument(
        '--dtype', action='store', nargs='?', type=str.lower, const='float32',
        choices=['float32', 'float64'], default='float32',
        help='S|Default type to use. On GPU float32 should be faster.\n'
        'If TF  < 1.4.0 then float32 is used.\n'
        'Default: %(default)s')

    parser.add_argument(
        '--valid_feed', action='store', nargs='?', type=int,
        const=256,  # 256 if valid_feed is specified but value not provided
        help='S|Run validation via feed_dict. Decouples validation but\n'
        'runs a bit slower. Set this flag otherwise not decoupled.\n'
        'Optionally specify validation size: 256 by default.')

    args = parser.parse_args()

    return args

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_xy = K.print_tensor(box_xy, message="box_xy is: ")
    box_wh = K.print_tensor(box_wh, message="box_wh is: ")

    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    box_mins = K.print_tensor(box_mins, message="box_mins is: ")
    box_maxes = K.print_tensor(box_maxes, message="box_maxes is: ")


    return K.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1], # x_max
        box_maxes[..., 1:2]  # y_max
    ])


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes_xy, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""

    #nprint("image_shape = {}".format(image_shape))
    height = float(image_shape[0]) # scale_y
    width = float(image_shape[1]) # scale_x
    image_dims = K.stack([width, height, width,height]) # xyxy
    image_dims = K.reshape(image_dims, [1, 4])
    boxes_xy = boxes_xy * image_dims
    return boxes_xy

def box_min_max_to_xcycwh( box_min_max, img_rows, img_cols) :
    '''
    
    :param box_min_max: list of values, [xmin,xmax,w,h,class]
    :return: 
    '''
    # Lets scale the boxes, and change coordinates from xmin,ymin, xmax, ymax to xcenter,ycenter, width, height
    # Change coordiantes
    i=box_min_max
    box_xy = [0.5 * (i[0]+i[2]), 0.5*(i[1]+i[3])]
    box_wh = [i[2]-i[0],i[3] - i[1]]

    #Scale down to a value between 0 and 1
    box_xy =  [box_xy[0]/img_cols,box_xy[1]/img_rows]
    box_wh =  [box_wh[0]/img_cols,box_wh[1]/img_rows]

    assert(box_xy[0] < 1.0)
    assert(box_xy[1] < 1.0)

    box_xywhc = box_xy + box_wh + [box_min_max[4]]
    return box_xywhc

def get_grid_location(box, grid_rows, grid_cols) :
    x = box[0]
    y = box[1]
    # validate x,y are > 0 and < 1
    assert(x >= 0.0 and y >= 0.0 and x <= 1.0 and y <= 1.0)

    xscale = 1.0/grid_cols
    yscale = 1.0/grid_rows
    xind = int(x / xscale)
    yind = int(y / yscale)
    return (xind,yind)

# select anchorbox index with the best IOU
def select_anchorbox_index(box_xywhc,anchor_list,conv_width,conv_height) :

    ## scale box to convolutional feature spatial dimensions
    box_class = box_xywhc[4]
    box_xywh = box_xywhc[0:4] * np.array(
        [conv_width, conv_height, conv_width, conv_height])

    i = np.floor(box_xywh[1]).astype('int')
    j = np.floor(box_xywh[0]).astype('int')
    best_iou = 0
    best_anchor = 0
    #
    for k, anchor in enumerate(anchor_list):
        # Find IOU between box shifted to origin and anchor box.
         box_maxes = box_xywh[2:4] / 2.
         box_mins = -box_maxes
         anchor_maxes = (anchor / 2.)
         anchor_mins = -anchor_maxes
         intersect_mins = np.maximum(box_mins, anchor_mins)
         intersect_maxes = np.minimum(box_maxes, anchor_maxes)
         intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
         intersect_area = intersect_wh[0] * intersect_wh[1]
         box_area = box_xywh[2] * box_xywh[3]
         anchor_area = anchor[0] * anchor[1]
         iou = intersect_area / (box_area + anchor_area - intersect_area)
         if iou > best_iou:
             best_iou = iou
             best_anchor = k

    return k

# Encoding box with numpy
def encode_box(box_xywh, conv_width, conv_height, best_anchor) :
    (x,y) = get_grid_location(box_xywh,conv_height,conv_width)

    # Centers of the box relative to the assigned grid cell only
    adjusted_box = np.array(
        [
            box_xywh[0]*conv_width - x, box_xywh[1]*conv_height - y,
            np.log(box_xywh[2]*conv_width / best_anchor[0]),
            np.log(box_xywh[3]*conv_height / best_anchor[1])
        ],
        dtype=np.float32)
    assert(adjusted_box[0] > 0.0 and adjusted_box[0] < 1.0 and adjusted_box[0] > 0.0 and adjusted_box[0] < 1.0)
    return adjusted_box

# Decoding box with Keras
def decode_box() :
    a=1


def plot_image(img) :
    assert(img.dtype.name == 'uint8')

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))


def infer(audit_mode=False, output_dir="./output"):
    args = _parser()
    dev_cnt = 0 if args.device == 'cpu' else 1
    sess = K.get_session()

    #input_stream = "./sampleVideos/ian.mov"
    #input_stream = "./sampleVideos/ElephantStampede.mp4"
    input_stream = "/data/work/osa/2018-02-cleartechnologies-b8p021/crate_1min.mp4"

    # 270 for mov // 0 for mp4...
    mydemo = yolo_demo(sess, 
        _input_stream=input_stream, 
        _class_names=read_classes("./model_data/coco_classes.txt"), 
        _anchors=read_anchors("./model_data/yolo_anchors.txt"), 
        batch_size=16, 
        frame_stride=10)
    

    # mydemo.play_video()
    mydemo.output_dir(output_dir, overwrite=True)
    mydemo.audit_mode(audit_mode)


    mydemo.set_image_shape()
    mydemo.load_and_build_graph()
    mydemo.print_model_summary()

    mydemo.process_video("clear.mov")

    #mydemo.train()

def retrain():
    args = _parser()
    dev_cnt = 0 if args.device == 'cpu' else 1
    sess = K.get_session()

    # TF Debugger... try standard!
    #from tensorflow.python import debug as tf_debug
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)

    mydemo = yolo_demo(sess)    
    mydemo.class_names(read_classes("./retrain/clear_classes.txt"))
    mydemo.anchors(read_anchors("./retrain/yolo_anchors.txt"))
    mydemo.retrain_file("./retrain/labels.json")
    mydemo.retrain()

# Either retrain, or infer ...
if __name__ == '__main__':
    np.random.seed(1)
    #infer(audit_mode=True, output_dir="./output")
    retrain()




    
'''
            # (1080, 1920, 3)
            # assert(frame.shape == (1080,1920,3))

            # Pad my frame with zeros to square it up .. Hardcoded fo
            #frame = np.lib.pad(frame,((0,1200),(0,640),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            #frame = np.lib.pad(frame,((420,420),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            #assert(frame.shape == (1920,1920,3))
            
                        #if top - label_size[1] >= 0:
            #    text_origin = np.array([left, top - label_size[1]])
            #else:
            #    text_origin = np.array([left, top + 1])
            #
            ## My kingdom for a good redistributable image drawing library.
            #for i in range(thickness):
            #    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            #draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            
            
            
'''