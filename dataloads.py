import torch
from torch.utils.data import DataLoader, Dataset, distributed
import glob
import os
from pathlib import Path
import cv2
import math
import numpy as np
from augmentaton import letterBox,randon_perspective
from PIL import Image,ImageDraw,ExifTags,ImageOps
import random
import psutil
from tqdm import tqdm
import contextlib
from multiprocessing.pool import Pool
import torch.distributed as dist
from general import LOGGER
from contextlib import contextmanager
import hashlib
from itertools import repeat

# HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
# LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
# RANK = int(os.getenv('RANK', -1))
# PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
# IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
# TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

# NUM_THREADS = min(8, max(1, os.cpu_count() - 1)) 


# def seed_worker(worker_id):
#     # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# def img2label_paths(img_paths):
#     # Define label paths as a function of image paths
#     sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labs{os.sep}'  # /images/, /labels/ substrings
#     return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
# def get_hash(paths):
#     # Returns a single hash value of a list of paths (files or dirs)
#     size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
#     h = hashlib.sha256(str(size).encode())  # hash sizes
#     h.update(''.join(paths).encode())  # hash paths
#     return h.hexdigest()  # return hash

# for orientation in ExifTags.TAGS.keys():
#     if ExifTags.TAGS[orientation] == 'Orientation':
#         break

# def exif_size(img):
#     # Returns exif-corrected PIL size
#     s = img.size  # (width, height)
#     with contextlib.suppress(Exception):
#         rotation = dict(img._getexif().items())[orientation]
#         if rotation in [6, 8]:  # rotation 270 or 90
#             s = (s[1], s[0])
#     return s


# def verify_image_label(args):
#     # Verify one image-label pair
#     im_file, lb_file, prefix = args
#     nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
#     try:
#         # verify images
#         im = Image.open(im_file)
#         im.verify()  # PIL verify
#         shape = exif_size(im)  # image size
#         assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
#         assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
#         if im.format.lower() in ('jpg', 'jpeg'):
#             with open(im_file, 'rb') as f:
#                 f.seek(-2, 2)
#                 if f.read() != b'\xff\xd9':  # corrupt JPEG
#                     ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
#                     msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

#         # verify labels
#         if os.path.isfile(lb_file):
#             nf = 1  # label found
#             with open(lb_file) as f:
#                 lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
#                 # if any(len(x) > 6 for x in lb):  # is segment
#                 #     classes = np.array([x[0] for x in lb], dtype=np.float32)
#                 #     segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
#                 #     lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
#                 lb = np.array(lb, dtype=np.float32)
#             nl = len(lb)
#             if nl:
#                 assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
#                 assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
#                 assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
#                 _, i = np.unique(lb, axis=0, return_index=True)
#                 if len(i) < nl:  # duplicate row check
#                     lb = lb[i]  # remove duplicates
#                     if segments:
#                         segments = [segments[x] for x in i]
#                     msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
#             else:
#                 ne = 1  # label empty
#                 lb = np.zeros((0, 5), dtype=np.float32)
#         else:
#             nm = 1  # label missing
#             lb = np.zeros((0, 5), dtype=np.float32)
#         return im_file, lb, shape, segments, nm, nf, ne, nc, msg
#     except Exception as e:
#         nc = 1
#         msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
#         return [None, None, None, None, nm, nf, ne, nc, msg]


# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (dw, dh)

# def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
#     # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
#     y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
#     y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
#     y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
#     return y

# def clip_boxes(boxes, shape):
#     # Clip boxes (xyxy) to image shape (height, width)
#     if isinstance(boxes, torch.Tensor):  # faster individually
#         boxes[..., 0].clamp_(0, shape[1])  # x1
#         boxes[..., 1].clamp_(0, shape[0])  # y1
#         boxes[..., 2].clamp_(0, shape[1])  # x2
#         boxes[..., 3].clamp_(0, shape[0])  # y2
#     else:  # np.array (faster grouped)
#         boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
#         boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


# def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
#     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
#     if clip:
#         clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
#     y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
#     y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
#     y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
#     return y

# def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
#     # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
#     w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
#     w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
#     ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
#     return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

# def random_perspective(im,
#                        targets=(),
#                        segments=(),
#                        degrees=10,
#                        translate=.1,
#                        scale=.1,
#                        shear=10,
#                        perspective=0.0,
#                        border=(0, 0)):
#     # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
#     # targets = [cls, xyxy]

#     height = im.shape[0] + border[0] * 2  # shape(h,w,c)
#     width = im.shape[1] + border[1] * 2

#     # Center
#     C = np.eye(3)
#     C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
#     C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

#     # Perspective
#     P = np.eye(3)
#     P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
#     P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

#     # Rotation and Scale
#     R = np.eye(3)
#     a = random.uniform(-degrees, degrees)
    
#     s = random.uniform(1 - scale, 1 + scale)
   
#     R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

#     # Shear
#     S = np.eye(3)
#     S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
#     S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

#     # Translation
#     T = np.eye(3)
#     T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
#     T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

#     # Combined rotation matrix
#     M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
#     if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
#         if perspective:
#             im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
#         else:  # affine
#             im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    
#     n = len(targets)
#     if n:
        
#         new = np.zeros((n, 4))
        
#         xy = np.ones((n * 4, 3))
#         xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
#         xy = xy @ M.T  # transform
#         xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

#         # create new boxes
#         x = xy[:, [0, 2, 4, 6]]
#         y = xy[:, [1, 3, 5, 7]]
#         new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

#         # clip
#         new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
#         new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

#         # filter candidates
#         i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr= 0.10)
#         targets = targets[i]
#         targets[:, 1:5] = new[i]

#     return im, targets


# class LoadImagesAndLabels(Dataset):
#     # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
#     cache_version = 0.6  # dataset labels *.cache version
#     rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

#     def __init__(self,
#                  path,
#                  img_size=640,
#                  batch_size=16,
#                  augment=False,
#                  hyp=None,
#                  rect=False,
#                  image_weights=False,
#                 #  cache_images=False,
#                  single_cls=False,
#                  stride=32,
#                  pad=0.0,
#                  min_items=0,
#                  prefix=''):
#         self.img_size = img_size
#         self.augment = augment
#         self.hyp = hyp
#         self.image_weights = image_weights
#         self.rect = False if image_weights else rect
#         self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
#         self.mosaic_border = [-img_size // 2, -img_size // 2]
#         self.stride = stride
#         self.path = path
#         self.albumentations =   None  #Albumentations(size=img_size) if augment else None

#         try:
#             f = []  # image files
#             for p in path if isinstance(path, list) else [path]:
#                 p = Path(p)  # os-agnostic
#                 if p.is_dir():  # dir
#                     f += glob.glob(str(p / '**' / '*.*'), recursive=True)
#                     # f = list(p.rglob('*.*'))  # pathlib
#                 elif p.is_file():  # file
#                     with open(p) as t:
#                         t = t.read().strip().splitlines()
#                         parent = str(p.parent) + os.sep
#                         f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
#                         # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
#                 else:
#                     raise FileNotFoundError(f'{prefix}{p} does not exist')
#             self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
#             # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
#             assert self.im_files, f'{prefix}No images found'
#         except Exception as e:
#             raise Exception(f'{prefix}Error loading data from {path}: {e}\n{HELP_URL}') from e

#         # Check cache
#         self.label_files = img2label_paths(self.im_files)  # labels
#         cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
#         try:
#             cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
#             assert cache['version'] == self.cache_version  # matches current version
#             assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
#         except Exception:
#             cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

#         # Display cache
#         nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
#         if exists and LOCAL_RANK in {-1, 0}:
#             d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
#             tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
#             if cache['msgs']:
#                 LOGGER.info('\n'.join(cache['msgs']))  # display warnings
#         assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}'

#         # Read cache
#         [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
#         labels, shapes, self.segments = zip(*cache.values())
#         nl = len(np.concatenate(labels, 0))  # number of labels
#         assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}'
#         self.labels = list(labels)
#         self.shapes = np.array(shapes)
#         self.im_files = list(cache.keys())  # update
#         self.label_files = img2label_paths(cache.keys())  # update

#         # Filter images
#         if min_items:
#             include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
#             LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
#             self.im_files = [self.im_files[i] for i in include]
#             self.label_files = [self.label_files[i] for i in include]
#             self.labels = [self.labels[i] for i in include]
#             self.segments = [self.segments[i] for i in include]
#             self.shapes = self.shapes[include]  # wh

#         # Create indices
#         n = len(self.shapes)  # number of images
#         bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
#         nb = bi[-1] + 1  # number of batches
#         self.batch = bi  # batch index of image
#         self.n = n
#         self.indices = range(n)

#         # Update labels
#         include_class = []  # filter labels to include only these classes (optional)
#         self.segments = list(self.segments)
#         include_class_array = np.array(include_class).reshape(1, -1)
#         for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
#             if include_class:
#                 j = (label[:, 0:1] == include_class_array).any(1)
#                 self.labels[i] = label[j]
#                 # if segment:
#                 #     self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
#             if single_cls:  # single-class training, merge all classes into 0
#                 self.labels[i][:, 0] = 0

#         # Rectangular Training
#         if self.rect:
#             # Sort by aspect ratio
#             s = self.shapes  # wh
#             ar = s[:, 1] / s[:, 0]  # aspect ratio
#             irect = ar.argsort()
#             self.im_files = [self.im_files[i] for i in irect]
#             self.label_files = [self.label_files[i] for i in irect]
#             self.labels = [self.labels[i] for i in irect]
#             self.segments = [self.segments[i] for i in irect]
#             self.shapes = s[irect]  # wh
#             ar = ar[irect]

#             # Set training image shapes
#             shapes = [[1, 1]] * nb
#             for i in range(nb):
#                 ari = ar[bi == i]
#                 mini, maxi = ari.min(), ari.max()
#                 if maxi < 1:
#                     shapes[i] = [maxi, 1]
#                 elif mini > 1:
#                     shapes[i] = [1, 1 / mini]

#             self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

#         # Cache images into RAM/disk for faster training
#         # if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
#         #     cache_images = False
#         # self.ims = [None] * n
#         # self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
#         # if cache_images:
#         #     b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
#         #     self.im_hw0, self.im_hw = [None] * n, [None] * n
#         #     fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
#         #     results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
#         #     pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
#         #     for i, x in pbar:
#         #         if cache_images == 'disk':
#         #             b += self.npy_files[i].stat().st_size
#         #         else:  # 'ram'
#         #             self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
#         #             b += self.ims[i].nbytes
#         #         pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
#         #     pbar.close()

#     def check_cache_ram(self, safety_margin=0.1, prefix=''):
#         # Check image caching requirements vs available memory
#         b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
#         n = min(self.n, 30)  # extrapolate from 30 random images
#         for _ in range(n):
#             im = cv2.imread(random.choice(self.im_files))  # sample image
#             ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
#             b += im.nbytes * ratio ** 2
#         mem_required = b * self.n / n  # GB required to cache dataset into RAM
#         mem = psutil.virtual_memory()
#         cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
#         if not cache:
#             LOGGER.info(f'{prefix}{mem_required / gb:.1f}GB RAM required, '
#                         f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
#                         f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
#         return cache

#     def cache_labels(self, path=Path('./labels.cache'), prefix=''):
#         # Cache dataset labels, check images and read shapes
#         x = {}  # dict
#         nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
#         desc = f'{prefix}Scanning {path.parent / path.stem}...'
#         with Pool(NUM_THREADS) as pool:
#             pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
#                         desc=desc,
#                         total=len(self.im_files),
#                         bar_format=TQDM_BAR_FORMAT)
#             for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
#                 nm += nm_f
#                 nf += nf_f
#                 ne += ne_f
#                 nc += nc_f
#                 if im_file:
#                     x[im_file] = [lb, shape, segments]
#                 if msg:
#                     msgs.append(msg)
#                 pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'

#         pbar.close()
#         if msgs:
#             LOGGER.info('\n'.join(msgs))
#         if nf == 0:
#             LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
#         x['hash'] = get_hash(self.label_files + self.im_files)
#         x['results'] = nf, nm, ne, nc, len(self.im_files)
#         x['msgs'] = msgs  # warnings
#         x['version'] = self.cache_version  # cache version
#         try:
#             np.save(path, x)  # save cache for next time
#             path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
#             LOGGER.info(f'{prefix}New cache created: {path}')
#         except Exception as e:
#             LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # not writeable
#         return x

#     def __len__(self):
#         return len(self.im_files)

#     # def __iter__(self):
#     #     self.count = -1
#     #     print('ran dataset iter')
#     #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
#     #     return self

#     def __getitem__(self, index):
#         index = self.indices[index]  # linear, shuffled, or image_weights

#         hyp = self.hyp
#         mosaic = self.mosaic #and random.random() < hyp['mosaic']
#         if mosaic:
#             # Load mosaic
#             img, labels = self.load_mosaic(index)
#             shapes = None

#             # MixUp augmentation
#             # if random.random() < hyp['mixup']:
#             #     img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

#         else:
#             # Load image
#             img, (h0, w0), (h, w) = self.load_image(index)

#             # Letterbox
#             shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
#             img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
#             shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

#             labels = self.labels[index].copy()
#             if labels.size:  # normalized xywh to pixel xyxy format
#                 labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

#             # if self.augment:
#             #     img, labels = random_perspective(img,
#             #                                      labels,
#             #                                      degrees=hyp['degrees'],
#             #                                      translate=hyp['translate'],
#             #                                      scale=hyp['scale'],
#             #                                      shear=hyp['shear'],
#             #                                      perspective=hyp['perspective'])

#         nl = len(labels)  # number of labels
#         if nl:
#             labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

#         if self.augment:
#             pass
#             # Albumentations
#             # img, labels = self.albumentations(img, labels)
#             # nl = len(labels)  # update after albumentations

#             # # HSV color-space
#             # augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

#             # # Flip up-down
#             # if random.random() < hyp['flipud']:
#             #     img = np.flipud(img)
#             #     if nl:
#             #         labels[:, 2] = 1 - labels[:, 2]

#             # # Flip left-right
#             # if random.random() < hyp['fliplr']:
#             #     img = np.fliplr(img)
#             #     if nl:
#             #         labels[:, 1] = 1 - labels[:, 1]

#             # # Cutouts
#             # # labels = cutout(img, labels, p=0.5)
#             # # nl = len(labels)  # update after cutout

#         labels_out = torch.zeros((nl, 6))
#         if nl:
#             labels_out[:, 1:] = torch.from_numpy(labels)

#         # Convert
#         img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         img = np.ascontiguousarray(img)

#         return torch.from_numpy(img), labels_out, self.im_files[index], shapes

#     def load_image(self, i):
#         # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
#         # im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
#         # if im is None:  # not cached in RAM
#         #     if fn.exists():  # load npy
#         #         im = np.load(fn)
#         #     else:  # read image
#         #         im = cv2.imread(f)  # BGR
#         #         assert im is not None, f'Image Not Found {f}'
#         #     h0, w0 = im.shape[:2]  # orig hw
#         #     r = self.img_size / max(h0, w0)  # ratio
#         #     if r != 1:  # if sizes are not equal
#         #         interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
#         #         im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
#         #     return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
#         # return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

#         im = cv2.imread(self.im_files[i])
#         h0,w0 = im.shape[:2]  #[h,w,c]
#         r = self.img_size / max(h0,w0)
#         if r != 1:
#             im = cv2.resize(im,(math.ceil(w0 * r), math.ceil(h0 * r)),interpolation=cv2.INTER_AREA)

#         return im,(h0,w0),im.shape[:2]

#     def cache_images_to_disk(self, i):
#         # Saves an image as an *.npy file for faster loading
#         f = self.npy_files[i]
#         if not f.exists():
#             np.save(f.as_posix(), cv2.imread(self.im_files[i]))

#     def load_mosaic(self, index):
#         # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
#         labels4, segments4 = [], []
#         s = self.img_size
#         yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
#         indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
#         random.shuffle(indices)
#         for i, index in enumerate(indices):
#             # Load image
#             img, _, (h, w) = self.load_image(index)

#             # place img in img4
#             if i == 0:  # top left
#                 img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
#                 x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
#                 x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
#             elif i == 1:  # top right
#                 x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
#                 x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
#             elif i == 2:  # bottom left
#                 x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
#                 x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
#             elif i == 3:  # bottom right
#                 x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
#                 x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

#             img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
#             padw = x1a - x1b
#             padh = y1a - y1b

#             # Labels
#             labels, segments = self.labels[index].copy(), self.segments[index].copy()
#             if labels.size:
#                 labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
#                 # segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
#             labels4.append(labels)
#             segments4.extend(segments)

#         # Concat/clip labels
#         labels4 = np.concatenate(labels4, 0)
#         for x in (labels4[:, 1:], *segments4):
#             np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
#         # img4, labels4 = replicate(img4, labels4)  # replicate

#         # Augment
#         # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
#         img4, labels4 = random_perspective(img4,
#                                            labels4,
#                                            segments4,
#                                            degrees=self.hyp['degrees'],
#                                            translate=self.hyp['translate'],
#                                            scale=self.hyp['scale'],
#                                            shear=self.hyp['shear'],
#                                            perspective=self.hyp['perspective'],
#                                            border=self.mosaic_border)  # border to remove

#         return img4, labels4

    

#     @staticmethod
#     def collate_fn(batch):
#         im, label, path, shapes = zip(*batch)  # transposed
#         for i, lb in enumerate(label):
#             lb[:, 0] = i  # add target image index for build_targets()
#         return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    

# @contextmanager
# def torch_distributed_zero_first(local_rank: int):
#     # Decorator to make all processes in distributed training wait for each local_master to do something
#     if local_rank not in [-1, 0]:
#         dist.barrier(device_ids=[local_rank])
#     yield
#     if local_rank == 0:
#         dist.barrier(device_ids=[0])

# def create_dataloader(path,
#                       imgsz,
#                       batch_size,
#                       stride,
#                       single_cls=False,
#                       hyp=None,
#                       augment=False,
#                       cache=False,
#                       pad=0.0,
#                       rect=False,
#                       rank=-1,
#                       workers=8,
#                       image_weights=False,
#                     #   quad=False,
#                       prefix='',
#                       shuffle=False,
#                       seed=0):
#     # if rect and shuffle:
#     #     LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
#     #     shuffle = False
#     # with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
#     dataset = LoadImagesAndLabels(
#         path,
#         imgsz,
#         batch_size,
#         augment=augment,  # augmentation
#         hyp=hyp,  # hyperparameters
#         rect=rect,  # rectangular batches
#         # cache_images=cache,
#         single_cls=single_cls,
#         stride=int(stride),
#         pad=pad,
#         image_weights=image_weights,
#         prefix=prefix)

#     batch_size = min(batch_size, len(dataset))
#     nd = torch.cuda.device_count()  # number of CUDA devices
#     # nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
#     sampler = None #if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
#     loader = DataLoader #if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
#     generator = torch.Generator()
#     generator.manual_seed(6148914691236517205 + seed -1)
#     return loader(dataset,
#                   batch_size=batch_size,
#                   shuffle=shuffle,   # and sampler is None,
#                   num_workers=1,
#                   sampler=sampler,
#                   pin_memory=PIN_MEMORY,
#                   collate_fn=LoadImagesAndLabels.collate_fn,
#                   worker_init_fn=seed_worker,
#                   generator=generator), dataset




NUM_THREADS = min(4, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'

class yolodateset(Dataset):
    def __init__(
            self,
            path,
            img_size=640,
            batch_size = 4,
            hyp = None,
            augment = False,
            rect = False,
            stride = 32,
            pad = 0
            
            ) -> None:
        super().__init__()
        self.path = path
        self.im_size = img_size
        self.im_files = glob.glob(os.path.join(self.path,'images/*'))
        self.im_labs =  glob.glob(os.path.join(self.path,'labs/*'))
        self.lab_files = []
        self.hyp = hyp
        self.rect = rect
        self.augment = augment
        self.mosaic = augment and not rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        for lb_file in self.im_labs:
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            self.lab_files.append(lb)

        LOGGER.info(f'im_labs len is {len(self.im_labs)}')
        
        cache_path = Path(self.im_labs[0]).parent.with_suffix('.cache')
        # try:

        #     cache,exists = np.load(cache_path,allow_pickle=True)
        # except:
        cache,exists = self.cache_labels(cache_path), False
        
        (nf,nm,ne,nc,n) = cache.pop('results')
        if exists:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
        
        [cache.pop('msgs')]
        labels,shapes = zip(*cache.values())
        

        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())

        print(f'labels 0 is {self.labels[0]}')
        print(f'labels 1 is {self.labels[1]}')
        print(f'labels 2 is {self.labels[2]}')
        print(f'labels 3 is {self.labels[3]}')

        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        if self.rect:
             # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.im_labs = [self.im_labs[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride
        
        # if cache_images == 'ram' and not self.check_cache_ram():
        #     cache_images = False
        # self.ims = [None] * n
        # self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        # if cache_images:
        #     b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        #     self.im_hw0, self.im_hw = [None] * n, [None] * n
        #     fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
        #     results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
        #     pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT)
        #     for i, x in pbar:
        #         if cache_images == 'disk':
        #             b += self.npy_files[i].stat().st_size
        #         else:  # 'ram'
        #             self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
        #             b += self.ims[i].nbytes
        #         pbar.desc = f'Caching images ({b / gb:.1f}GB {cache_images})'
        #     pbar.close()


    def __len__(self):
        return len(self.im_files)
    def __getitem__(self, index):
        
        hyp = self.hyp
        mosaic = self.mosaic #and random.random() < hyp['mosaic']
        shapes = None
        if mosaic:
            img, labels = self.load_mosaic(index)
            shapes = None

        else:

            img, (h0,w0),(h, w) = self.load_img(index)
            shape = self.batch_shapes[self.batch[index]] if self.rect   else (self.im_size,self.im_size)
            img,r,dw,dh = letterBox(im=img,new_shape=shape,auto=False)

            shapes = (h0,w0),((h/h0,w/w0),(dw,dh))

            labels = self.lab_files[index].copy()
            
            if labels.size:
                labels[:,1:] = self.xywhn2xyxy(labels[:,1:],r*w,r*h,dw,dh)
                # labels[:,1:] = self.xyxy2xywhn(labels[:,1:],img.shape[1],img.shape[0])

            if self.augment:
                img,labels = randon_perspective(
                im=img,
                targets=labels,
                degrees=hyp['degrees'],
                translate=hyp['translate'],
                scale=hyp['scale'],
                shear=hyp['shear'],
                perspective=hyp['perspective']
                )

        num_lab = len(labels)
        labels_out = torch.zeros((num_lab, 6))
        if num_lab:
            labels[:, 1:] = self.xyxy2xywhn(labels[:, 1:], w=img.shape[1], h=img.shape[0])
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2,0,1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img),labels_out,shapes
    def check_cache_ram(self,safety_margin=0.1):
        #check image cacheing requirements available memory
        b = 0
        n = min(self.n,30)
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))
            ratio = self.im_size / max(im.shape[0],im.shape[1])
            b += im.nbytes*ratio*ratio

        mem_required = b*self.n/n
        mem = psutil.virtual_memory() #  total, available, percent, used  
        cache = mem_required*(1+safety_margin)<mem.available
        if not cache:
            pass
        return cache
    def cache_labels(self,path=Path('./labels.cache')):
        """
        
        input   path  cache path
        output  x{}     'im_file':[lb, shape] 
                                        lb, [cxywh] Normalization
                                        shape,[h,w]image shape
                        'results':nf,nm,ne,nc,len(self.im_files),
                        'msgs':msgs
                        '

        """
        x = {}
        nm, nf, ne, nc, msgs = 0,0,0,0,[]  ## number missing, found, empty, corrupt, messages
        desc = f'Scanning{path.parent/path.stem}'
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label,zip(self.im_files,self.im_labs)),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=TQDM_BAR_FORMAT
                        )
            for im_file, lb, shape,nm_f,nf_f,ne_f,nc_f,msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape]
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()
        if nf == 0:
            pass

        x['results'] = nf,nm,ne,nc,len(self.im_files)
        x['msgs'] = msgs

        try:
            np.save(path , x)
            path.with_suffix('.cache.npy').rename(path)
        except:
            pass
        return x
    def cache_images_to_disk(self, i):
            # Saves an image as an *.npy file for faster loading
            f = self.npy_files[i]
            if not f.exists():
                np.save(f.as_posix(), cv2.imread(self.im_files[i]))
    
    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.im_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_img(index=index) 

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.labels[index].copy()#, self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = self.xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                # segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            # segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:]):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = randon_perspective(img4,
                                           labels4,
                                           degrees=self.hyp['degrees'],# degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4
    
    # def load_mosaic(self,index):
    #     """
    #         output  img [hwc]   type:numpy
    #                 lables [class xyxy]  pixels
    #     """
    #     labels4 = []
    #     size = self.im_size
    #     hc,wc = (int(random.uniform(-x,2*size+x)) for x in self.mosaic_border)
    #     indices = [index]+random.choices(self.indices,k=3)
    #     random.shuffle(indices)
    #     for i,index in enumerate(indices):
    #         img,_,(h,w) = self.load_img(index=index)

    #         if i == 0:  # top left
    #             img4 = np.full((size * 2, size * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
    #             x1a, y1a, x2a, y2a = max(wc - w, 0), max(hc - h, 0), wc, hc  # xmin, ymin, xmax, ymax (large image)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
    #         elif i == 1:  # top right
    #             x1a, y1a, x2a, y2a = wc, max(hc - h, 0), min(wc + w, size * 2), hc
    #             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    #         elif i == 2:  # bottom left
    #             x1a, y1a, x2a, y2a = max(wc - w, 0), hc, wc, min(size * 2, hc + h)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
    #         elif i == 3:  # bottom right
    #             x1a, y1a, x2a, y2a = wc, hc, min(wc + w, size * 2), min(size * 2, hc + h)
    #             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

    #         img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            
    #         padw = x1a - x1b
    #         padh = y1a - y1b

    #         labels = self.labels[index].copy()

    #         if labels.size:
    #             labels[:,1:] = self.xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  
            
    #         labels4.append(labels)
    #     labels4 = np.concatenate(labels4,0)
    #     np.clip(labels4[:,1:],0,2*size,out=labels4[:,1:])
        
    #     img4 , labels4 = randon_perspective(
    #         im = img4,
    #         targets=labels4,
    #         degrees=self.hyp['degrees'],
    #         translate=self.hyp['translate'],
    #         scale=self.hyp['scale'],
    #         shear=self.hyp['shear'],
    #         perspective=self.hyp['perspective'],
    #         border=self.mosaic_border
    #     )
    #     return img4,labels4
    def load_img(self,index):
        """
                no pad
                input   index 
                output  im      [hwc]   type:numpy
                        (h0,w0)   image original h  w    
                        im.shape[:2]  image resize h w
        """
        print(f'im file index is{index}')
        im = cv2.imread(self.im_files[index])
        h0,w0 = im.shape[:2]  #[h,w,c]
        r = self.im_size / max(h0,w0)
        if r != 1:
            im = cv2.resize(im,(math.ceil(w0 * r), math.ceil(h0 * r)),interpolation=cv2.INTER_AREA)

        return im,(h0,w0),im.shape[:2]
    def xywhn2xyxy(self,x,w=640,h=640,dw=0,dh=0):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + dw  # top left x
        y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + dh  # top left y
        y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + dw  # bottom right x
        y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + dh  # bottom right
        return y
    def xyxy2xywhn(self,x,w=640, h=640,clip = True,eps=1E-3):
        if clip:
            
            if isinstance(x, torch.Tensor):  # faster individually
                x[..., 0].clamp_(0, w-eps)  # x1
                x[..., 1].clamp_(0, h-eps)  # y1
                x[..., 2].clamp_(0, w-eps)  # x2
                x[..., 3].clamp_(0, h-eps)  # y2
            else:  # np.array (faster grouped)
                x[..., [0, 2]] = x[..., [0, 2]].clip(0, w-eps)  # x1, x2
                x[..., [1, 3]] = x[..., [1, 3]].clip(0, h-eps)
        
        
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
        y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
        y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
        y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
        return y
    @staticmethod
    def collate_fn(data):
        im,lb,shapes = zip(*data)
        for i,lab in enumerate(lb):
            lab[:,0] = i
        return torch.stack(im,0) ,torch.cat(lb,0),shapes


        


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

class_name = ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']

def xywh2xyxy(x,w=640,h=640):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = w * (x[..., 2] - x[..., 4] / 2)   # top left x
    y[..., 3] = h * (x[..., 3] - x[..., 5] / 2)   # top left y
    y[..., 4] = w * (x[..., 2] + x[..., 4] / 2)   # bottom right x
    y[..., 5] = h * (x[..., 3] + x[..., 5] / 2)   # bottom right
    return y

def draw_targets(img,labs):
    """
    input   img [batch channel h w ] type:tensor
            labs [class,xywh]  type:tensor
    output  draw image and target
    
    """    
    shape = img.shape
    labs = xywh2xyxy(labs,shape[2],shape[3])

    for i in range(img.shape[0]):
        # lb = torch.ones(labs.shape)
        j = labs[:,0]==i
        lb = labs[j]
        im = Image.fromarray(img[i].numpy().transpose(1,2,0))
        draw = ImageDraw.Draw(im)
        for k in range(lb.shape[0]):

            draw.rectangle(lb[k,2:].numpy(),outline='red')
            draw.text(lb[k,2:4].numpy().astype(np.uint)+[0,-8],class_name[lb[k,1].numpy().astype(np.uint)],fill='red')
        del draw
        im.show()
        im.save('D:\python\yolov5-mysely\ccc.jpg',format='png')

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s



def verify_image_label(args):
    """
        input       argss:im_fiel,lb_file
        output      im_file     im_fiel  
                    lb,         [cxywh] Normalization
                    shape,      [h,w]image shape
                    nm,         if nm = 1 label missing  
                    nf,         if nf = 1 label found
                    ne,         if ne = 1 label empty
                    nc          if nc = 1 file corrupt
                    msg 
    """
    
    im_file,lb_file=args
    nm, nf, ne, nc, msg, = 0, 0, 0, 0, ''
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        # assert (shape[0]>9 & shape[1]>9),f'image size{shape}<10 pixels'
        if im.format.lower() in ('jpg','jpeg'):
            with open(im_file,'rb') as f:
                f.seek(-2,2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'WARNING  {im_file}: corrupt JPEG restored and saved'

        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)

            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 {lb.shape[1]} columns detected'
                assert (lb>=0).all,f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'

                _, i = np.unique(lb,axis=0,return_index=True)
                if len(i)<nl:
                    lb=lb[i]
            else:
                ne = 1   #label empty
                lb = np.zeros((0,5),dtype=np.float32)
        else:
            nm = 1  #label missing
            lb = np.zeros((0,5),dtype=np.float32)
        return  [im_file, lb, shape, nm, nf, ne, nc, msg]   

    except Exception as e:
        nc = 1
        msg = f'WARNING {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, nm, nf, ne, nc, msg]
        


class InfiniteDataLoader(DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
        path,
        imgsz,
        batch_size,
        stride,
        hyp=None,
        augment=False,
        pad=0.0,
        rect=False,
        workers=4,
        shuffle=False,
        seed=0
    ):
    if rect and shuffle:
        LOGGER.warning('WARNING  --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    dataset = yolodateset(
        path = path,
        img_size = imgsz,
        batch_size = batch_size,
        hyp=hyp,  # hyperparameters
        augment=augment,  # augmentation
        rect=rect,  # rectangular batches
        stride=stride,
        pad=pad
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
     
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed -1)
    return DataLoader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle,
                  num_workers=1,
                  sampler=None,
                  pin_memory=PIN_MEMORY,
                  collate_fn= yolodateset.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset







if __name__ == "__main__":
    
    
    path = r'D:\\python\\my_pcb_dataset\\train\\'

    hyp = {'box':0.05,
        'degrees':90.0,
        'translate':0.2,
        'scale':0.5,
        'shear':90.0,
        'fliplr':0.5,
        'mosaic':1.0,
        'mixup':0.0,
        'copy_paste':0.0,
        'perspective':0.0
        }
    
    
    
    
    yolov5dataloader, yolov5dataset = create_dataloader(
        path=Path(path),
        imgsz=640,
        batch_size=4,
        stride=32,
        hyp=hyp,
        augment=False,
        pad=0.0,
        rect=False,
        workers=1,
        shuffle=True,
        seed=0
    )


    for im,lab,_ in yolov5dataloader:
        
        draw_targets(im,lab)
        
        # plt.figure()
        # plt.axis('off')
        # for i in range(im.shape[0]):
        #     plt.imshow(im[i].numpy().transpose(1,2,0))
        # plt.ion()
        # plt.pause(30)
        # plt.close()
        
        # for i in range(im.shape[0]): 
        #     image = Image.fromarray(im[i].numpy().transpose(1,2,0))
        #     draw = ImageDraw.Draw(image)
        #     draw.rectangle(lab[:,1])
        #     draw.text([100, 200], text="pcb", font=font, fill=(100, 185, 179))
        #     del draw
        #     image.show()

        # for i in range(im.shape[0]):
        #     imagee = im[i].numpy().transpose(1,2,0)
        #     image = cv2.rectangle(image, start_point, end_point, color, thickness)
        #     image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA) 
        #     cv2.imshow('pcb',imagee)


    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # def visualize_bbox(img, bbox, class_name, color, ax):
    #     """
    #     img:图片数据 (H, W, C)数据格式
    #     bbox:array或者tensor， 假定数据格式是 [x_mid, y_mid, width, height]
    #     classname:str 目标框对应的种类
    #     color:str
    #     thickness:目标框的宽度
    #     """
    #     x_mid, y_mid, width, height = bbox
    #     x_min = int(x_mid - width / 2)
    #     y_min = int(y_mid - height / 2)
    #     # 画目标检测框
    #     rect = patches.Rectangle((x_min, y_min), 
    #                                 width, 
    #                                 height, 
    #                                 linewidth=3,
    #                                 edgecolor=color,
    #                                 facecolor="none"
    #                                 )
    #     ax.imshow(img)
    #     ax.add_patch(rect)
    #     ax.text(x_min + 1, y_min - 3, class_name, fontSize=10, bbox={'facecolor':color, 'pad': 3, 'edgecolor':color})

    # def visualize(img, bboxes, category_ids, category_id_to_name, category_id_to_color):
    #     fig, ax = plt.subplots(1, figsize=(8, 8))
    #     ax.axis('off')
    #     for box, category in zip(bboxes, category_ids):
    #         class_name = category_id_to_name[category]
    #         color = category_id_to_color[category]
    #         visualize_bbox(img, box, class_name, color, ax)
    #     plt.show()