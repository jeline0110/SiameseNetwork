import cv2
import os
import numpy as np
import numpy.random as nr
from random import shuffle
from multiprocessing.dummy import Pool as ThreadPool
import pdb
import glob as gb

img_dir = '/home/scf/seal_paper_experiment/seal_5k' 
bg_dir = '/home/scf/seal_paper_experiment/bg_data'
out_dir = '/home/scf/seal_paper_experiment/seal_bg_test'
file_num = 0

def img_add_bg(img_path, img_name, bg_dir, save_dir):
  """
  Generate new seal images with different backgrounds.
  """
  selected_bg_num = 10  #For each give image, we randomly select 50 cropped background regions.
  bg_list = np.sort(os.listdir(bg_dir))
  img = cv2.imread(img_path) 

  img_h = img.shape[0]
  img_w = img.shape[1]

  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  threshold = img_gray.mean()
  rnd_list = np.arange(len(bg_list)) 
  shuffle(rnd_list)  #random the images list
  for select in range(selected_bg_num):
    bg_path = os.path.join(bg_dir, bg_list[rnd_list[select]])
    bg = cv2.imread(bg_path)
    #pdb.set_trace()
    if min(bg.shape[0],bg.shape[1]) > max(img.shape[0], img.shape[1]):
      resize_ratio = nr.randint(1,6)
    else:
      resize_ratio = max(img.shape[0], img.shape[1]) / min(bg.shape[0],bg.shape[1])
    new_bg_h = int(resize_ratio * bg.shape[0]) + 1 # +1 ensure new_bg_h >= img_h
    new_bg_w = int(resize_ratio * bg.shape[1]) + 1
    bg = cv2.resize(bg,(new_bg_w,new_bg_h))

    start_y = nr.randint(0,bg.shape[0]-img_h+1) # +1 ensure end_y could equal boudary vaule
    end_y = start_y + img_h        
    start_x = nr.randint(0,bg.shape[1]-img_w+1)
    end_x = start_x + img_w
    crop_bg = bg[start_y:end_y,start_x:end_x,:]

    combine_ratio = 0.4 + 0.4 * nr.rand()  #We randomly compute the combining ratio.
    crop_bg[img_gray<threshold,:]= combine_ratio* img[img_gray<threshold,:] + (1.0 - combine_ratio)* crop_bg[img_gray<threshold,:]
    save_path = os.path.join(save_dir, img_name + '_%d.jpg' % select)
    cv2.imwrite(save_path,crop_bg)

def deal_img(img_path):
    global file_num
    if img_path.lower().endswith('.jpg'): 
      img_name = os.path.basename(img_path).split('.')[0]
      save_dir = os.path.join(out_dir, img_name) 
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      file_num += 1
      img = img_add_bg(img_path, img_name, bg_dir, save_dir)
      if file_num % 100 == 0:
        print ('---------> dealing %d' % file_num)

if __name__ == '__main__':
  img_path = gb.glob(img_dir + "/*.jpg")
  pool = ThreadPool(10)
  pool.map(deal_img, img_path)
  pool.close()
  pool.join()