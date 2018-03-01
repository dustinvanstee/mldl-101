'''
Program that reads in a video stream and outputs an object detection stream  [WIP]

credits :
  https://www.ibm.com/developerworks/aix/library/au-threadingpython/
  https://www.troyfawkes.com/learn-python-multithreading-queues-basics/
'''

# add imports here ..
import argparse as ap
import os
import numpy as np
import sys
import cv2
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
import yad2k.models.keras_yolo as yad2k
import random
import colorsys
import time


from threading import Thread
import queue


# Shared queue
queue = queue.Queue()

# Globals
yolo_graph = None

# Constants
MODEL_PATH = "/data/work/git-repos/yad2k/model_data/yolo.h5"


class yolo_demo(object):
    """The fully-connected neural network model."""

    def __init__(self, sess, input_stream, class_names, anchors,
            image_shape=(720.,1280.), 
            rotation=0,
            batch_size=64, 
            frame_stride=1):
        
        self.sess = sess
        self.input_stream = input_stream
        self.class_names = class_names
        self.anchors = anchors
        self.batch_size = batch_size
        self.frame_stride = frame_stride
        self.image_shape = image_shape # in future, auto set this ...
        self.rotation = rotation
        self.yolo_model_image_shape = (608.0,608.0)
        self.yolo_model = None
        self.yolo_outputs = None
        self.scores = None
        self.boxes = None
        self.classes = None

    def load_and_build_graph(self) :
        self.yolo_model = load_model(MODEL_PATH)
        #yolo_model.summary()
        self.yolo_outputs = yad2k.yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))
        self.scores, self.boxes, self.classes = self.yolo_eval() # sets self.scores, self.boxes, self.classes structures

    def build_model(self) :
        '''
        stub for case when yolo.h5 is missing
        '''
        a=1

    def shade_image_19_19(self) :
        '''
        stub for image mod ....
        '''
        a=1

    def train(self):
        '''
        stub for training step ...
        '''
        a=1

    def set_image_shape(self):
        # open input stream just once and cache ....
        cap = cv2.VideoCapture(self.input_stream)
        ret, frame = cap.read()
        print("Setting self.image_shape to {0}".format(frame.shape))
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
            print("error")
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
        box_corners = yad2k.yolo_boxes_to_corners(box_xy, box_wh)

        # use a for loop, and lets see what happens ...
        #trainable=False
        # all_scores = tf.get_variable(name="all_scores",shape=(),initializer=tf.zeros_initializer(),dtype=tf.float32,collections=[tf.GraphKeys.LOCAL_VARIABLES])

        all_boxes = []
        all_scores = []
        all_classes = []

        for m in range(0,self.batch_size) :
            tmp_box_class_probs = tf.slice(box_class_probs, [m,0,0,0,0], [1,19,19,5,80], name=None)
            tmp_boxes = tf.slice(box_corners, [m,0,0,0,0], [1,19,19,5,4], name=None)
            tmp_confidence = tf.slice(box_confidence, [m,0,0,0,0], [1,19,19,5,1], name=None)


            tmp_scores, tmp_boxes, tmp_classes = self.yolo_filter_boxes(box_class_probs=tmp_box_class_probs,box_confidence=tmp_confidence,boxes=tmp_boxes,threshold=iou_threshold)

            # Scale boxes back to original image shape.
            tmp_scaled_boxes = scale_boxes(tmp_boxes, self.image_shape)

            # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
            ts,tb,tc = self.yolo_non_max_suppression(tmp_scores, tmp_scaled_boxes, tmp_classes, max_boxes, iou_threshold)

            all_scores.append(ts)
            all_boxes.append(tb)
            all_classes.append(tc)

        return all_scores, all_boxes, all_classes

    def yolo_filter_boxes(self, box_confidence, boxes, box_class_probs, threshold = .6):
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
        all_boxes = tf.boolean_mask(boxes,filtering_mask,name='score_mask')
        all_classes = tf.boolean_mask(box_classes,filtering_mask,name='score_mask')

        return all_scores, all_boxes, all_classes

    def yolo_non_max_suppression(self, scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
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
        ### START CODE HERE ### (≈ 1 line)
        nms_indices = tf.image.non_max_suppression( boxes, scores, max_boxes, iou_threshold)
        ### END CODE HERE ###

        # Use K.gather() to select only nms_indices from scores, boxes and classes
        ### START CODE HERE ### (≈ 3 lines)
        scores = K.gather(scores, nms_indices)
        boxes = K.gather(boxes, nms_indices)
        classes = K.gather(classes, nms_indices)
        ### END CODE HERE ###

        return scores, boxes, classes



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

        M = cv2.getRotationMatrix2D((cols/2,rows/2),self.rotation,1)

        rv_rotated_image = np.zeros((examples,rows,cols,rgb))
        rv_scaled_image = np.zeros((examples,model_image_size[0],model_image_size[1],rgb))

        for i in range(0,examples) :
            cur_image = image[i]

            rotated_image1 = cv2.warpAffine(cur_image,M,(cols,rows))
            resized_image1 = cv2.resize(rotated_image1,model_image_size,cv2.INTER_CUBIC)
            resized_image2 = np.array(resized_image1, dtype='float32')
            resized_image3 = resized_image2 / 255.0
            #expanded_image4 = np.expand_dims(resized_image3, 0)  # Add batch dimension.
            rv_rotated_image[i] = rotated_image1
            #plot_image(rv_rotated_image[i].astype('uint8'))

            rv_scaled_image[i] = resized_image3
            #plot_image(resized_image3)

        return rv_rotated_image,rv_scaled_image


    def draw_boxes(self, image_data, out_scores, out_boxes, out_classes, colors):
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
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            print("score = {0}, boxes={1}".format(score,box))
            label = '{} {:.2f}'.format(predicted_class, score)

            # label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))

            bottom = min(image_data.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image_data.shape[1], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

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

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_data,predicted_class,(left,top), font, 1,colors[c],2,cv2.LINE_AA)
            image_data = cv2.rectangle(image_data,(left,top),(right,bottom),colors[c],3)

        for x in range(0,9) :
            for y in range(0,9) :
                x1 = x*200
                y1 = y*200
                x2 = x*200+199
                y2 = y*200+199
                image_data = cv2.rectangle(image_data,(x1,y1),(x2,y2),(127,255,0),2)

        return image_data



    def yolo_producer(self, tensor_queue):
        """
        Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
        
        Arguments:
        
        Returns:
        
        """
        global yolo_graph
        with yolo_graph.as_default():
            cap = cv2.VideoCapture('/data/work/osa/2018-01-yolo/test.mov')

            loop_cnt = 0
            num_frames_to_show = 20
            frame = np.ones((16,self.get_image_shape("height") ,self.get_image_shape("width"),self.get_image_shape("channels")),dtype="uint8")
            while(cap.isOpened() and loop_cnt < num_frames_to_show ):
                nprint("Qsize = {}".format(tensor_queue.qsize()))
                # frame is an ndarray of nhwc
                for j in range(0,self.batch_size) :
                    ret,  frame[j] = cap.read()
                    for i in range(0, self.frame_stride) :
                        ret, dummy_frame = cap.read()

                nprint("Frame shape = {}".format(frame.shape))

                # preprocess returns a 4D tensor (examples, x, y, RGB)

                image_rotate, image_data = self.preprocess_image_cpu(frame)

                # can plot image data b/c of added dimension ... plot_image(image_data)

                # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
                abcd = K.learning_phase()
                out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],feed_dict={self.yolo_model.input: image_data , K.learning_phase(): 0})
                tensor_queue.put(item=(image_rotate, out_boxes, out_scores, out_classes))

                # Print predictions info
                for i in range(0,self.batch_size):
                    print('Found {} boxes for current batch {}'.format(len(out_boxes[i]),i))

                loop_cnt += 1

            cap.release()
        nprint("yolo_producer completed")



    def yolo_consumer(self, tensor_queue):
        """
        Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
        
        Arguments:
        
        Returns:
        
        """
        import cv2
        while True:
            nprint("Qsize = {}".format(tensor_queue.qsize()))
            image_rotate, out_boxes, out_scores, out_classes = tensor_queue.get(block=True)

            # Generate colors for drawing bounding boxes.
            colors = generate_colors(self.class_names)
            # Draw bounding boxes on the image file
            #draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

            for m in range(0,self.batch_size):
                image_modified = self.draw_boxes(image_data=image_rotate[m], out_scores=out_scores[m], out_boxes=out_boxes[m], out_classes=out_classes[m], colors=colors)
                # plot_image(image_modified)
                # Display the resulting frame
                cv2.imshow('frame',image_modified.astype('uint8'))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            tensor_queue.task_done()

        cv2.destroyAllWindows()


#################### End Class yolo_demo ################################


class threaded_yolo(Thread):
    '''
    Class : threaded_yolo
      1. uses a shared queue to process a video stream via tensor flow yolo algorithm
      2. loads shared queue 
      3. video consumer used to display results when available
    '''
    def __init__(self, queue, yolo_demo):
        self.queue = queue
        self.frame_rate = 1.0 # process every second
        self.yolo_demo = yolo_demo
        # Spin up some threads upon initialization
        self.producer = Thread(target=self.produce_boxes, args=())
        self.consumer = Thread(target=self.consume_boxes, args=())

        self.producer.setDaemon(True) # threads will stay alive until main exits
        self.producer.start()
        self.consumer.setDaemon(True) # threads will stay alive until main exits
        self.consumer.start()

    def produce_boxes(self):
        # Processes a video stream and loads tensor into shared queue
        nprint("starting")
        self.yolo_demo.yolo_producer(self.queue)

    def consume_boxes(self):
        # seperate thread to draw boxes and display results
        nprint("starting")
        self.yolo_demo.yolo_consumer(self.queue)

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

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""

    #nprint("image_shape = {}".format(image_shape))
    height = float(image_shape[0])
    width = float(image_shape[1])
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def plot_image(img) :
    assert(img.dtype.name == 'uint8')

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def nprint(mystring) :
    print("{} : {}".format(sys._getframe(1).f_code.co_name,mystring))




def main():
    args = _parser()
    dev_cnt = 0 if args.device == 'cpu' else 1
    sess = K.get_session()
    class_names = read_classes("/data/work/git-repos/yad2k/model_data/coco_classes.txt")
    anchors = read_anchors("/data/work/git-repos/yad2k/model_data/yolo_anchors.txt")
    input_stream = "/data/work/osa/2018-01-yolo/test.mov"

    mydemo = yolo_demo(sess, input_stream, class_names, anchors, batch_size=16, frame_stride=20, rotation=270)
    mydemo.set_image_shape()
    mydemo.load_and_build_graph()
    global yolo_graph
    yolo_graph = tf.get_default_graph()

    # Simultaneously run producer / consumer
    t = threaded_yolo(queue,mydemo)
    time.sleep(130)





if __name__ == '__main__':
    np.random.seed(1)
    main()

'''
General Notes about Threading and tensorflow

After I had threaded this tensorflow program, I kept getting this error
  -> Cannot interpret feed_dict key as Tensor:

The error occured when I would try to inference at the sess.run line.  Basically, sess.run didnt have access to default 
tensorflow graph that I was using.  
  
After researching on google, I came across this post that help me solve this but
#  https://github.com/jaungiers/Multidimensional-LSTM-BitCoin-Time-Series/issues/1

This post discusses how new threads spawned dont have accesss to the default graph that was created in a different thread.  
You need to save the graph, and then use a python global variable to access in a different thread.

I used this article as a refresher for global vars in python https://www.geeksforgeeks.org/global-local-variables-python/.

'''
#
#
#

# Load an color image in grayscale
#frame = cv2.imread('/data/work/osa/2018-01-yolo/coursera/cnn/week3/AutonDriving/images/test.jpg')
# frame = cv2.imread(self.input_stream)
# (1080, 1920, 3)
# assert(frame.shape == (1080,1920,3))

# Pad my frame with zeros to square it up .. Hardcoded fo
#frame = np.lib.pad(frame,((0,1200),(0,640),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
#frame = np.lib.pad(frame,((420,420),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
#assert(frame.shape == (1920,1920,3))
