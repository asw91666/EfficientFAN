import torch.utils.data as data
from h5py import File
import ref2 as ref
import numpy as np
import cv2
from utils.utils import rnd, flip, shuffle_lr, flip_vertical
from utils.img import transform, crop, draw_gaussian
import pdb
from data.augmentation import load_occluders, occlude_with_objects # data augmentation
import math

data_train_dir = '/home/jsh/Desktop/chun/dataset/COFW/COFW_train_color.mat'
data_test_dir = '/home/jsh/Desktop/chun/dataset/COFW/COFW_test_color.mat'
occlude_img_path = '/home/jsh/git_workspace/synthetic-occlusion/VOCdevkit/VOC2012' # occlusion으로 augment할 때 쓸 이미지의 경

class COFW(data.Dataset):
    def __init__(self, is_train):
        '''
        init에서 해줘야하는것들 :
        pts뽑아서 ndarray 로 만들어주기

        annot 은 지워도 됨. 교수님 코드에서는 annot이라는 h5 -> dic 으로 처리했으나 나는 txt->list로 처리할 예정
        is_train
        num_sample
        '''
        # occluders는 train 에서만 적용한다.
        if is_train == True:
            print('==> Get Occluders')
            self.occluders = load_occluders(pascal_voc_root_path=occlude_img_path)  # occluders가 segmentation된 물체.

        # print(f'Occluder 사용 안함 -> 디버깅 끝난후 바꿔줘야 함')

        print('==> Initializing face landmark data')

        if is_train:
            f = File(data_train_dir, 'r') # color
            self.bboxed = f['bboxesTr'] # <HDF5 dataset "bboxesTr": shape (4, 1345), type "<f8">
            self.images = f['IsTr']
            self.landmark = f['phisTr'] # <HDF5 dataset "phisTr": shape (87, 1345), type "<f8"> ==> x,y,z 좌표가 각각 29,29,29개씩 저장.
            self.f = f
        else:
            f = File(data_test_dir, 'r')  # test
            self.bboxed = f['bboxesT']  # <HDF5 dataset "bboxesT": shape (4, 507), type "<f8">
            self.images = f['IsT']
            self.landmark = f['phisT']  # <HDF5 dataset "phisT": shape (87, 507), type "<f8"> ==> x,y,z 좌표가 각각 29,29,29개씩 저장.
            self.f = f

        self.images_ = self.images[0]

        # outer eye index of cofw
        self.outer_eye_idx = [8, 9]  # [0] : right, [1] : left

        self.is_train = is_train
        self.num_samples = self.bboxed.shape[1]

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_dev = [0.229, 0.224, 0.225]

        print('Load %d samples' % self.num_samples)


    def load_image(self, index): # 인덱싱할 때 마다 여기에 접근하는데 이 때 img파일이 없어서 nonetype만 가져오게 되는거야.
        '''
        output : cv2 image [C,H,W]
        '''
        image_ref = self.images_[index]
        image = self.f[image_ref]

        # if gray img
        if image.ndim == 2:
            image = np.array(image)[np.newaxis, :, :]
            image = np.concatenate([image, image, image], axis=0)  # (3,H,W)

        image = np.transpose(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def load_landmarks(self, index): # 여기서 index는 txt파일에서 읽을 line의 index.
        '''
        output : pts(ndarray) [landmark, 2]
        '''
        landmark = np.transpose(self.landmark)[index]
        coordi_x = landmark[:29]
        coordi_y = landmark[29:58]

        coordi_x = coordi_x[:, np.newaxis]
        coordi_y = coordi_y[:, np.newaxis]
        pts = np.concatenate((coordi_x, coordi_y), axis=1) # [num_landmark,2]

        return pts

    def __getitem__(self, index):
        '''
        inp : crop and resized image
        pts : annotation인데 resized image에 맞게 좌표도 바뀐것의 집합.
        hmap : pts를 기준으로 가우시안 분포를 이루는 heatmap
        '''
        # Get raw data
        img = self.load_image(index) # ndarray
        pts = self.load_landmarks(index) # ndarray
        c = np.array((img.shape[1]/2.0, img.shape[0]/2.0))
        s = img.shape[0]/2.0
        r = 0.0
        # print(f'img shape : {img.shape}')
        # print(f'pts : {pts.shape}')

        # Compute tight bounding box and center/scale
        if True:
            bbox_min = np.min(pts, axis=0)  # [2,]
            bbox_max = np.max(pts, axis=0)  # [2,]
            bbox = np.array([bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1]])  # (x_min, y_min, x_max, y_max)
            c = np.array((bbox[2] - (bbox[2] - bbox[0]) / 2.0,
                          bbox[3] - (bbox[3] - bbox[1]) / 2.0))  # crop된 이미지의 center 좌표(original 이미지 좌표계를 기준으로함)
            s = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.5  # bounding box를 얼마나 넉넉하게 자를것인지 판별.

        # Global constants
        res_ratio = ref.res_in / ref.res_out

        # Data augmentation (scaling and rotation)
        if self.is_train == True:
            s = s * (2 ** rnd(ref.scale))
            r = 0 if np.random.random() < 0.6 else rnd(ref.rotate)
        inp = crop(img, c, s, r, ref.res_in)  # image crop and resize to network input size

        ### start augmentation code
        if self.is_train == True:
            inp = occlude_with_objects(inp, self.occluders)  # original_im 가 내가 augmentation하고자 하는 대상 이미지.
        ### end augmentation code

        ### start test code : occluder까지 적용후 변형된 이미지, 변형된 landmark잘 찍혀있나 확인용 코드
        # inp_flip = inp # flip 한 이미지용
        # inp_ = inp # 원래 이미지
        ### end test code

        # transforms.toTensor()
        inp = inp.transpose(2, 0, 1).astype(np.float32) / 255.  # [H,W,C] ==> [C,H,W], input image normalize

        # Data augmentation (jittering, flip, blur)
        do_flip = False
        if self.is_train == True:
            # flip
            # if np.random.random() < 0.5:
            #     do_flip = True
            #     inp = flip(inp)
                # inp_flip = flip(inp_flip) # debugging code

            if np.random.random() < 0.5:
                inp = cv2.GaussianBlur(inp, (9, 9), 0)

            # jitter
            inp[0] = np.clip(inp[0] * (np.random.random() * .4 + .6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * .4 + .6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * .4 + .6), 0, 1)

        # transforms.Normalization()
        '''
        inp[0] = (inp[0] - self.norm_mean[0]) / self.norm_dev[0]
        inp[1] = (inp[1] - self.norm_mean[1]) / self.norm_dev[1]
        inp[2] = (inp[2] - self.norm_mean[2]) / self.norm_dev[2]
        '''

        # pts flip horizontal
        '''if do_flip == True:
            pts = shuffle_lr(pts)  # annot index edit
            for i in range(ref.num_landmarks):  # annot coordi flip
                pts[i][0] = ref.res_in - pts[i][0] - 1'''

        # set annotation
        # heatmap loss를 걸어주기 위해 초기에 heatmap gt를 만들어준다. 이 때, annotation을 기준으로 가우시안분포가 되게끔 만들어준다.
        # Set output heatmap and gt landmarks
        hmap = np.zeros((ref.num_landmarks, ref.res_out, ref.res_out), dtype=np.float32)
        for i in range(ref.num_landmarks):
            pts[i] = transform(pts[i], c, s, r, ref.res_in)
            hmap[i] = draw_gaussian(hmap[i], pts[i] / res_ratio + 0.5, ref.std)

        # debugging code
        # for i in range(ref.num_landmarks):
        #     inp_ = cv2.line(inp_, (int(pts[i][0]), int(pts[i][1])), (int(pts[i][0]), int(pts[i][1])), (0, 0, 255), 8)

        # debugging code : occluder까지 적용후 변형된 이미지, 변형된 landmark잘 찍혀있나 확인용 코드
        # for i in range(ref.num_landmarks):
        #     inp_flip = cv2.line(inp_flip, (int(pts[i][0]), int(pts[i][1])), (int(pts[i][0]), int(pts[i][1])), (0, 0, 255), 5)
        #     cv2.imshow('flip_inp', inp_flip)
        #     cv2.waitKey(0)

        # cv2.imshow('inp', inp_)
        # cv2.imshow('inp_blur', inp_blur)
        # cv2.imshow('flip_inp', inp_flip)
        # cv2.waitKey(0)

        # get normalization factor
        norm_factor = []
        _norm_factor = math.sqrt((pts[self.outer_eye_idx[0]][0] - pts[self.outer_eye_idx[1]][0]) ** 2 +
                                 (pts[self.outer_eye_idx[0]][1] - pts[self.outer_eye_idx[1]][1]) ** 2)
        for i in range(ref.num_landmarks):
            norm_factor.append(_norm_factor)  # shape : [num_landmark]
        norm_factor = np.array(norm_factor)  # ndarray

        return inp, hmap, pts, norm_factor

    def __len__(self):
        return self.num_samples

