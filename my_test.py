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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#test_num = 10*5000
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("seal_bg_dir", "/home/scf/seal_paper_experiment/seal_bg_test/", "path to images with background")
tf.flags.DEFINE_string("seal_simple_dir", "/home/scf/seal_paper_experiment/seal_5k/", "path to simple images without background")
tf.flags.DEFINE_string("model_ckpt_path", "/home/scf/seal_paper_experiment/my_net/train_siamese_net/logs8/model.ckpt-5000", "path for restore.")
tf.flags.DEFINE_integer("test_num", '50000', "The number of test seals with bg, files' number of single dir*number of dirs")
tf.flags.DEFINE_integer("top_k", "1", "acquire top_num accuracy")
tf.flags.DEFINE_string("dic_saved_dir", "./dic8/", "path for saving dic(data,fea,sim).")

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
    def __init__(self): 
        self.img_size = 128
        self.data_aug_slight = False
        self.image = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name="input_image")
        with tf.variable_scope("siamese") as scope: self.feature = self.seal_net_sia()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, FLAGS.model_ckpt_path)
        print("Model restored...")
        if not os.path.exists(FLAGS.dic_saved_dir): os.makedirs(FLAGS.dic_saved_dir)
        
    def get_one_from_path(self,img_path):
        self.images = []
        img = self.load_data(img_path)
        self.images.append (img)
        self.images = np.array(self.images).astype(np.float32)

        return self.images

    def load_data(self, img_path):
        img = cv2.imread(img_path)
        if self.data_aug_slight:
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

    def seal_net_sia(self):
        with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
            net = slim.conv2d(self.image, 64, [5, 5], padding='SAME', scope='conv1')
            net = slim.max_pool2d(net)
            net = slim.conv2d(net, 128, [5, 5], padding='SAME', scope='conv2')
            net = slim.max_pool2d(net)
            net = slim.conv2d(net, 128, [3, 3], stride = [2,2], padding='SAME', scope='conv3')
            net = slim.conv2d(net, 192, [3, 3], padding='SAME', scope='conv4')
            net = slim.conv2d(net, 192, [3, 3], stride = [2,2], padding='SAME', scope='conv5')
            net = slim.conv2d(net, 192, [3, 3], padding='SAME', scope='conv6')
            net = slim.conv2d(net, 192, [3, 3], stride = [2,2], padding='SAME', scope='conv7')
            net = slim.conv2d(net, 192, [3, 3], padding='SAME', scope='conv8')
            net_fea = slim.fully_connected(slim.flatten(net), 1024, scope='fc9')
            #net_id = slim.fully_connected(net_fea, FLAGS.num_nodes, activation_fn=None, scope='fc10')
            
            return net_fea

    def get_feature(self, idx, img_dir, data_type='seal_simple'):
        gallery_fea = []
        gallery_lab = []
        gallery_name = []
        
        for img in np.sort(os.listdir(img_dir)):
            img_path = os.path.join(img_dir,img)
            img_name = os.path.basename(img_path)
            train_images = self.get_one_from_path(img_path)
            feed_dict = {self.image: train_images}
            fea = self.sess.run(self.feature, feed_dict=feed_dict)
            gallery_fea.append(fea)
            gallery_lab.append(idx)
            gallery_name.append(img_name)

            if data_type == 'seal_simple':
            	idx += 1
            	if idx % 100 ==0:
                    print ('--->%d files are done!' %idx) 

        return gallery_fea, gallery_lab, gallery_name

    def make_gallery(self, img_dir, data_type='seal_simple'):
        data_dic = FLAGS.dic_saved_dir + data_type + '_gallary'
        if os.path.exists(data_dic):
            print(data_dic.split('/')[-1] + ' existed!')
            dic = myunpickle(data_dic)
            gallery_fea = dic['gallery_fea']
            gallery_lab = dic['gallery_lab']
            gallery_name = dic['gallery_name']
        else:
            if data_type == 'seal_simple':
                file_idx = 0
                self.data_aug_slight = False
                gallery_fea, gallery_lab, gallery_name = self.get_feature(file_idx, img_dir, data_type)
                dic = {'gallery_fea':gallery_fea,'gallery_lab':gallery_lab,'gallery_name':gallery_name}

            elif data_type == 'seal_bg':
                gallery_fea = []
                gallery_lab = []
                gallery_name = []
                img_subdirs =  np.sort(os.listdir(img_dir))
                subdir2_idx = 0
                for img_subdir in img_subdirs:
                    img_subdir2 = os.path.join(img_dir, img_subdir)
                    self.data_aug_slight = True
                    temp_gallery_fea, temp_gallery_lab, temp_gallery_name = self.get_feature(subdir2_idx, img_subdir2, data_type)
                    gallery_fea.extend(temp_gallery_fea)
                    gallery_lab.extend(temp_gallery_lab)
                    gallery_name.extend(temp_gallery_name)

                    subdir2_idx += 1
                    if subdir2_idx % 10 ==0:
                        print ('--->%d dirs are done!' %subdir2_idx) 
                dic = {'gallery_fea':gallery_fea,'gallery_lab':gallery_lab,'gallery_name':gallery_name}

            mypickle(data_dic, dic)
            print(data_type  + '_gallery fea is done!!!')

        return gallery_fea, gallery_lab, gallery_name

def get_top_k(simlarity, pred_label, simple_lab, top_k):
    sim_index = np.argsort(simlarity,axis=0)  # asceding sort
    match_flag = 0
    for i in range(top_k):
        if simple_lab[sim_index[i]] == pred_label:
            match_flag = 1

    return match_flag, sim_index[:top_k]
    
def get_acc(pred_fea, pred_lab, pred_name, simple_fea, simple_lab, simple_name, top_k):
    right_cnt = 0
    total_cnt = 0
    test_file_name = []
    top_k_names = []
    sim = []

    sim_dic = FLAGS.dic_saved_dir + 'bg_simple_sim_dic'
    if os.path.exists(sim_dic):
        print(sim_dic.split('/')[-1] + ' existed!')
        dic = myunpickle(sim_dic)
        sim = dic['sim']
    else:
        for idx1 in range(len(pred_lab)):
            this_pred_fea = pred_fea[idx1]
            simlarity = np.zeros(len(simple_lab), dtype=np.float32)
            for idx2 in range(len(simple_lab)):
                this_simple_fea = simple_fea[idx2]
                simlarity[idx2] = ((this_pred_fea-this_simple_fea)**2).sum()
            sim.append(simlarity)
            if idx1 % 100 == 0:
                print('%d files simlarity have beed calculated' %idx1)
        dic = {'sim':sim}
        mypickle(sim_dic, dic)
        print('simlarity dictionary is done!!!')
    
    for idx1 in range(len(pred_lab)):
        this_pred_label = pred_lab[idx1]
        test_file_name.append(pred_name[idx1])
        match_flag, top_k_idx = get_top_k(sim[idx1], this_pred_label, simple_lab, top_k)

        this_top_k_name = []
        for i in range(len(top_k_idx)):
        	this_top_k_name.append(simple_name[top_k_idx[i]])
        top_k_names.append(this_top_k_name)

        print ('file_name: %s, pred_lab: %d ---> test_result: %d'
        	%(pred_name[idx1], this_pred_label, top_k_idx[0]))
        right_cnt += match_flag
        total_cnt += 1
    acc = float(right_cnt) / (total_cnt)*100.0

    return acc, test_file_name, top_k_names

def main(argv=None):
    dataprovider = DataLayer()
    simple_fea, simple_lab, simple_name = dataprovider.make_gallery(FLAGS.seal_simple_dir, data_type='seal_simple')
    pred_fea, pred_lab, pred_name = dataprovider.make_gallery(FLAGS.seal_bg_dir, data_type='seal_bg')
    #pdb.set_trace()
    top_k_num_acc = 'top_' + str(FLAGS.top_k) + '_' + str(FLAGS.test_num) + '_acc'
    test_dic = FLAGS.dic_saved_dir + top_k_num_acc
    if os.path.exists(test_dic):
        print(test_dic.split('/')[-1] + ' existed!')
        dic = myunpickle(test_dic)
        top_k = dic['top_k']
        acc = dic['acc']
        test_file_name = dic['test_file_name']
        top_k_names = dic['top_k_names']
    else:
        #pdb.set_trace()
        acc, test_file_name, top_k_names = get_acc(pred_fea[:FLAGS.test_num], pred_lab[:FLAGS.test_num], 
            pred_name[:FLAGS.test_num], simple_fea, simple_lab, simple_name, FLAGS.top_k)

        dic = {'top_k':FLAGS.top_k,'acc':acc,'test_file_name':test_file_name,'top_k_names':top_k_names}
        mypickle(test_dic, dic)
    
    print(top_k_num_acc + ' is %.3f!' %acc)
    #print(test_file_name[0], top_k_names[0])

if __name__ == "__main__":
    tf.app.run()