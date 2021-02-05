from project_images import project_image
from align_images import unpack_bz2
import argparse
import os
import sys
import bz2
import shutil
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import projector
import dataset_tool
from training import dataset
from training import misc

from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

class Api:
  def __init__(
    self,
    raw_dir,
    src_dir,
    dst_dir,
    network_pkl_gcloud_id,
    vgg16_pkl_gcloud_id,
    tmp_dir='.stylegan2-tmp',
    num_steps=1000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    dlatent_avg_fname=None,
    verbose=False
    ):
    self.raw_dir = raw_dir
    self.src_dir = src_dir
    self.dst_dir = dst_dir
    self.network_pkl_gcloud_id = network_pkl_gcloud_id
    self.vgg16_pkl_gcloud_id = vgg16_pkl_gcloud_id
    self.tmp_dir = tmp_dir
    self.num_steps = num_steps
    self.initial_learning_rate = initial_learning_rate
    self.initial_noise_factor = initial_noise_factor
    self.dlatent_avg_fname = dlatent_avg_fname
    self.verbose = verbose

  def align(self):
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    RAW_IMAGES_DIR = self.raw_dir
    ALIGNED_IMAGES_DIR = self.src_dir

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)

  def project(self):
    network_pkl = make_gcloud_link(self.network_pkl_gcloud_id)
    vgg16_pkl = make_gcloud_link(self.vgg16_pkl_gcloud_id)
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector(
        vgg16_pkl             = vgg16_pkl,
        num_steps             = self.num_steps,
        initial_learning_rate = self.initial_learning_rate,
        initial_noise_factor  = self.initial_noise_factor,
        verbose               = self.verbose,
        dlatent_avg_fname     = self.dlatent_avg_fname
    )
    proj.set_network(Gs)

    src_files = sorted([os.path.join(self.src_dir, f) for f in os.listdir(self.src_dir) if f[0] not in '._'])
    for src_file in src_files:
        project_image(proj, src_file, self.dst_dir, self.tmp_dir, video=False)
        shutil.rmtree(self.tmp_dir)

def make_gcloud_link(id):
  return f"https://drive.google.com/u/0/uc?id={id}&export=download"