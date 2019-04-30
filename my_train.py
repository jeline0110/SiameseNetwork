import tensorflow as tf
import numpy as np
import numpy.random as nr
slim = tf.contrib.slim
import os
import cv2
from random import shuffle
import pickle as cPickle
import pdb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "128", "batch size for training")
tf.flags.DEFINE_integer("num_class", "10000", "Total class for this dataset.")
tf.flags.DEFINE_integer("init_num_class", "10000", "How many nodes to be actived in this training phase?")
tf.flags.DEFINE_string("logs_dir", "./logs8/", "path to logs directory")
tf.flags.DEFINE_bool("fintune", "True", "If we train from the old model?")
tf.flags.DEFINE_string("finetune_ckpt_path", "./logs8/model.ckpt-2063", "path for finetune.")
tf.flags.DEFINE_string("seal_bg_dir", "/home/scf/seal_paper_experiment/seal_bg_train/", "path to images with background")
tf.flags.DEFINE_string("seal_simple_dir", "/home/scf/seal_paper_experiment/seal_10k/", "path to simple images without background")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool("sia_and_softmax", "True", "If we need the siamese loss?")
tf.flags.DEFINE_bool("data_aug_slight", "True", "If add data augmentation operation? ")
tf.flags.DEFINE_bool("debug", "False", "If output deubg image?")
tf.flags.DEFINE_bool("div_input", "False", "If diversify input?")

MAX_ITERATION = 200000

def mypickle(filename, data):
    fo = open(filename, "wb")
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def myunpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)

    fo = open(filename, 'rb')
    dict = cPickle.load(fo)    
    fo.close()
    return dict

class DataLayer:
    """Data layer for training"""   
    def __init__(self): 
        self.img_size = 128
        self.num_class = FLAGS.num_class
        self.init_num_class = FLAGS.init_num_class
        self.batch_size = FLAGS.batch_size
        self.seal_bg_dir =  FLAGS.seal_bg_dir
        self.seal_simple_dir =  FLAGS.seal_simple_dir
        self.seal_bg_list, self.seal_simple_list, self.seal_idx_list = self.data_processor()  #seal_bg_list consist of dirs  seal_simple_list consist of paths
        self.idx = 0
        self.data_num = len(self.seal_bg_list) # num of data pairs
        self.rnd_list = np.arange(self.data_num) # for random shuffle the images list
        shuffle(self.rnd_list)
        if not os.path.exists(FLAGS.logs_dir): os.makedirs(FLAGS.logs_dir)

    def next_batch(self):
        # load image + label pairs
        self.images_a = []
        self.images_b = []
        self.labels_a = []
        self.labels_b = []
        self.label_sia = []

        for i in range(self.batch_size):
            if self.idx == self.data_num:
                self.idx = 0            # start one new epoch of this dataset
                shuffle(self.rnd_list)
            cur_idx = self.rnd_list[self.idx]
            seal_idx_a = self.seal_idx_list[cur_idx]  # actually cur_idx = seal_idx_a
            img_a_dir = self.seal_bg_list[seal_idx_a]
            img_a = self.load_data(img_a_dir)
            label_a = np.zeros((self.num_class),dtype=np.int64)
            label_a[seal_idx_a] = 1
            self.idx += 1
            # Now, we need to select another sample to build the siamese pairs.
            if nr.rand()>0.5:
                seal_idx_b = nr.randint(max(self.seal_idx_list))
            else:
                seal_idx_b = seal_idx_a

            #pdb.set_trace()
            if FLAGS.div_input:
                if nr.rand()>0.5:
                    img_b_path = self.seal_simple_list[seal_idx_b] # if seal_idx_b = seal_idx_a  for closer to the center
                    img_b = self.load_data(img_b_path)
                else:
                    img_b_dir = self.seal_bg_list[seal_idx_b] # if seal_idx_b = seal_idx_a  for closer to the other data in this class
                    img_b = self.load_data(img_b_dir)
                label_b = np.zeros((self.num_class),dtype=np.int64)
                label_b[seal_idx_b] = 1
            
                if nr.rand()>0.5: # also random send the data to each branch. May could delte this operation,it's not useful training.
                    self.images_a.append(img_a)
                    self.images_b.append(img_b)
                    self.labels_a.append(label_a)
                    self.labels_b.append(label_b)
                else:
                    self.images_a.append(img_b)
                    self.images_b.append(img_a)
                    self.labels_a.append(label_b)
                    self.labels_b.append(label_a)
            else:
                #img_b_dir = self.seal_bg_list[seal_idx_b]
                #img_b = self.load_data(img_b_dir)
                img_b_path = self.seal_simple_list[seal_idx_b]
                img_b = self.load_data(img_b_path)
                #pdb.set_trace()
                label_b = np.zeros((self.num_class),dtype=np.int64)
                label_b[seal_idx_b] = 1
                self.images_a.append(img_a)
                self.images_b.append(img_b)
                self.labels_a.append(label_a)
                self.labels_b.append(label_b)

            #pdb.set_trace()
            if seal_idx_a == seal_idx_b:
                self.label_sia.append(1)
            else:
                self.label_sia.append(0)

        self.images_a = np.array(self.images_a).astype(np.float32)
        self.images_b = np.array(self.images_b).astype(np.float32)
        self.labels_a = np.array(self.labels_a).astype(np.int64)
        self.labels_b = np.array(self.labels_b).astype(np.int64)
        self.label_sia = np.array(self.label_sia).astype(np.int64)

        return self.images_a, self.labels_a, self.images_b, self.labels_b, self.label_sia
        
    def data_processor(self):
        data_dic = './10k_seal_bg_simple_dic_%d' %self.init_num_class
        if not os.path.exists(data_dic):
            seal_bg_list = []
            seal_simple_list = []
            seal_idx_list = []
            seal_bg_subdirs = np.sort(os.listdir(self.seal_bg_dir))
            idx_now = 0
            for seal_bg_subdir in seal_bg_subdirs[:self.init_num_class]:
                seal_bg_subdir2 = os.path.join(self.seal_bg_dir, seal_bg_subdir)
                seal_simple_path = os.path.join(self.seal_simple_dir, seal_bg_subdir+'.jpg')
                seal_bg_list.append(seal_bg_subdir2)
                seal_simple_list.append(seal_simple_path)
                seal_idx_list.append(idx_now)
                idx_now += 1
            dic = {'seal_bg_list':seal_bg_list,'seal_simple_list':seal_simple_list,'seal_idx_list':seal_idx_list}
            mypickle(data_dic, dic)
        # load saved data (to resume)
        else:
            dic = myunpickle(data_dic)
            seal_bg_list = dic['seal_bg_list']
            seal_simple_list = dic['seal_simple_list']
            seal_idx_list = dic['seal_idx_list']
        return seal_bg_list, seal_simple_list, seal_idx_list
    
    def load_data(self, path):
        if os.path.isdir(path):
            img_list = np.sort(os.listdir(path))
            img_path = img_list[nr.randint(len(img_list))]
            img_path = os.path.join(path, img_path)
            img = cv2.imread(img_path)
        elif os.path.isfile(path):
            img = cv2.imread(path)

        #pdb.set_trace()
        if FLAGS.data_aug_slight:
            if nr.rand()>0.2:
                img = self.image_augment_slight(img)
        newImage = np.zeros((self.img_size,self.img_size,3), dtype=np.int)
        newImage += 255
        resize_ratio = self.img_size / max(img.shape[0], img.shape[1])
        new_hight = int(resize_ratio * img.shape[0])
        new_width = int(resize_ratio * img.shape[1])
        try:
            inputImage = cv2.resize(img, (new_width, new_hight))
        except Exception as e:    
            pdb.set_trace()
        start_x = int(0.5 * (self.img_size - new_width))
        end_x = start_x + new_width
        start_y = int(0.5 * (self.img_size - new_hight))
        end_y = start_y + new_hight
        newImage[start_y:end_y,start_x:end_x,:] = inputImage[:,:,:]
        if FLAGS.debug:
            this_img_name = os.path.join('./debug/%04d.jpg'%self.idx)
            cv2.imwrite(this_img_name,newImage)
            print ('Saved to %s'%this_img_name)
            if self.idx==50:
                pdb.set_trace()
        newImage = np.array(newImage, dtype=np.float32)        
        # substract mean values for better training
        newImage = newImage - 127.5

        return newImage
        
    def image_augment_slight(self, inputImage):
        """
        For data augmentation, most used tricks are listed bellow.
        - rescale
        - rotate a very small degree(+-5)
        - brightness
        - random nosize
        """
        rescale_crop_pad = True
        bright = True
        nosize = True
        rotate = True
        perspective = True
        width = inputImage.shape[0]
        height = inputImage.shape[1]
        if rescale_crop_pad:
            pad_width = max(int(0.05 * min(width,height)),1)
            inputImage = np.pad(inputImage, ((pad_width,pad_width),(pad_width,pad_width),(0,0)), 'constant', constant_values=(255))
            start_x = nr.randint(0,pad_width*2)
            end_x = nr.randint(width,width+pad_width*2)
            start_y = nr.randint(0,pad_width*2)
            end_y = nr.randint(height,height+pad_width*2)
            inputImage = inputImage[start_x:end_x,start_y:end_y,:]
            
        if bright:
            bright_scale = 0.90 + 0.20 * nr.rand()# there will be 60%-140% bright augmentation
            inputImage = inputImage * bright_scale
            inputImage[inputImage>255] = 255
            inputImage[inputImage<0] = 0
            
        if nosize:
            nosize_mask = 10.0 - 20.0 * nr.rand(inputImage.shape[0],inputImage.shape[1],inputImage.shape[2])# add a normal distirbute nosize between(-10,+10)
            inputImage = inputImage + nosize_mask
            inputImage[inputImage>255] = 255
            inputImage[inputImage<0] = 0

        if rotate:
            pad_width = int(0.3 * min(width,height))
            inputImage = np.pad(inputImage, ((pad_width,pad_width),(pad_width,pad_width),(0,0)), 'constant', constant_values=(255))
            (w, h) = inputImage.shape[:2]
            center = (w / 2, h / 2)
            angle = -3.0 + 6.0* nr.rand()
            # rotate the image by 180 degrees
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            inputImage = cv2.warpAffine(inputImage, M, (h, w))
            (w, h) = inputImage.shape[:2]
            start_x = int(w/2-width/2)
            end_x = start_x + width
            start_y = int(h/2-height/2)
            end_y = start_y + height
            inputImage = inputImage[start_x:end_x,start_y:end_y,:]

        if perspective:
            perspect_ratio = 0.05
            pad_width = int(0.5 * max(width,height))
            inputImage = np.pad(inputImage, ((pad_width,pad_width),(pad_width,pad_width),(0,0)), 'constant', constant_values=(255))
            src_x1 = pad_width
            src_y1 = pad_width
            src_x2 = pad_width + width
            src_y2 = pad_width
            src_x3 = pad_width
            src_y3 = pad_width + height
            src_x4 = pad_width + width
            src_y4 = pad_width + height

            obj_x1 = int(src_x1 + (nr.randint(0,perspect_ratio*width) - perspect_ratio * 0.5 * width))
            obj_y1 = int(src_y1 + (nr.randint(0,perspect_ratio*height) - perspect_ratio * 0.5 * height))
            obj_x2 = int(src_x2 + (nr.randint(0,perspect_ratio*width) - perspect_ratio * 0.5 * width))
            obj_y2 = int(src_y2 + (nr.randint(0,perspect_ratio*height) - perspect_ratio * 0.5 * height))
            obj_x3 = int(src_x3 + (nr.randint(0,perspect_ratio*width) - perspect_ratio * 0.5 * width))
            obj_y3 = int(src_y3 + (nr.randint(0,perspect_ratio*height) - perspect_ratio * 0.5 * height))
            obj_x4 = int(src_x4 + (nr.randint(0,perspect_ratio*width) - perspect_ratio * 0.5 * width))
            obj_y4 = int(src_y4 + (nr.randint(0,perspect_ratio*height) - perspect_ratio * 0.5 * height))
            pts1 = np.float32([[src_y1, src_x1],[src_y2,src_x2],[src_y3,src_x3],[src_y4,src_x4]])
            pts2 = np.float32([[obj_y1, obj_x1],[obj_y2,obj_x2],[obj_y3,obj_x3],[obj_y4,obj_x4]])

            M = cv2.getPerspectiveTransform(pts1,pts2)
            w,h = inputImage.shape[:2]

            inputImage = cv2.warpPerspective(inputImage,M,(w,h))
            start_x = int(min(obj_x1, obj_x2, obj_x3, obj_x4))
            end_x = int(max(obj_x1, obj_x2, obj_x3, obj_x4))
            start_y = int(min(obj_y1, obj_y2, obj_y3, obj_y4))
            end_y = int(max(obj_y1, obj_y2, obj_y3, obj_y4))
            inputImage = inputImage[start_x:end_x,start_y:end_y,:]        
        return inputImage        

def stamp_net_sia(image):
   with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(image, 64, [5, 5], padding='SAME', scope='conv1')
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 128, [5, 5], padding='SAME', scope='conv2')
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 128, [3, 3], stride = [2,2], padding='SAME', scope='conv3')
        net = slim.conv2d(net, 192, [3, 3], padding='SAME', scope='conv4')
        net = slim.conv2d(net, 192, [3, 3], stride = [2,2], padding='SAME', scope='conv5')
        net = slim.conv2d(net, 192, [3, 3], padding='SAME', scope='conv6')
        net = slim.conv2d(net, 192, [3, 3], stride = [2,2], padding='SAME', scope='conv7')
        net = slim.conv2d(net, 192, [3, 3], padding='SAME', scope='conv8')
        net = slim.dropout(net, 0.5, scope='dropout8')
        net_fea = slim.fully_connected(slim.flatten(net), 1024, scope='fc9')
        net_id = slim.fully_connected(net_fea, FLAGS.num_class, activation_fn=None, scope='fc10')
        return net_fea, net_id
    
def contrastive_loss(model1, model2, y, margin):
	with tf.name_scope("contrastive-loss"):
		d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
		tmp= y * tf.square(d)    
		tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
		return 0.1 * tf.reduce_mean(tmp + tmp2) /2

def main(argv=None):
    image_a = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name="input_image_a")
    image_b = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name="input_image_b")
    label_a = tf.placeholder(tf.float32, shape=[None,FLAGS.num_class], name="label_a")
    label_b = tf.placeholder(tf.float32, shape=[None,FLAGS.num_class], name="label_b")
    label_sia = tf.placeholder(tf.float32, shape=[None], name="label_sia")
    with tf.variable_scope("siamese") as scope:
        fea_a, logits_a = stamp_net_sia(image_a)
        scope.reuse_variables()
        fea_b, logits_b = stamp_net_sia(image_b)
    loss_id_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=label_a, logits=logits_a))
    loss_id_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=label_b, logits=logits_b))
    loss_sia = contrastive_loss(fea_a,fea_b,label_sia,10.0)
    
    if FLAGS.sia_and_softmax:
        loss = loss_id_a + loss_id_b + loss_sia
    else:
        loss = loss_id_a + loss_id_b

    correct_prediction_a = tf.equal(tf.argmax(logits_a,1), tf.argmax(label_a,1))
    correct_prediction_b = tf.equal(tf.argmax(logits_b,1), tf.argmax(label_b,1))
    accuracy = 0.5 * tf.reduce_mean(tf.cast(correct_prediction_a, tf.float32)) + 0.5 * tf.reduce_mean(tf.cast(correct_prediction_b, tf.float32))
    
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
    
    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())
    start_step = 0
    if FLAGS.fintune:
        saver.restore(sess, FLAGS.finetune_ckpt_path)
        print("Model restored...")
        start_step = int(FLAGS.finetune_ckpt_path.split('-')[1]) + 1
        print("Continue train model at %d step" %(start_step-1))
 
    dataprovider = DataLayer()
    stop_num = 0
    for itr in range(start_step,MAX_ITERATION+1):
        train_images_a, train_labels_a, train_images_b, train_labels_b, train_label_sia = dataprovider.next_batch()
        feed_dict = {image_a: train_images_a, label_a: train_labels_a,
                     image_b: train_images_b, label_b: train_labels_b,
                     label_sia:train_label_sia}

        sess.run(train_op, feed_dict=feed_dict)
        train_loss,train_sia_loss, train_acc = sess.run([loss, loss_sia, accuracy], feed_dict=feed_dict)

        if itr % 1 == 0:              
            print("Step: %d, Train_loss:%g, Sia_loss: %g, Accuracy=%02f" % (itr, train_loss,train_sia_loss, train_acc))
       
        if not FLAGS.sia_and_softmax:
            if train_acc >= 0.98:
                stop_num += 1

        elif FLAGS.sia_and_softmax and FLAGS.data_aug_slight:
            if train_acc >= 0.99 and train_sia_loss <= 2:
                stop_num += 1

        '''
        elif FLAGS.sia_and_softmax and FLAGS.data_aug_slight and not FLAGS.div_input:
            if train_acc >= 0.9 and train_sia_loss <= 2:
                stop_num += 1
        
        elif FLAGS.div_input and FLAGS.learning_rate == 1e-3:
            if train_acc >= 0.96:
               stop_num += 1

        elif FLAGS.div_input and FLAGS.learning_rate == 1e-4:
            if train_acc >= 0.99:
               stop_num += 1
        '''     
        if itr % 1000 == 0:
            print("Model saved")
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

        if stop_num >= 30:
            print("Model training finished")
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            break

if __name__ == "__main__":
    tf.app.run()