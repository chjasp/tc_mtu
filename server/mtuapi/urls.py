from django.contrib import admin
from django.urls import path
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from google.cloud import storage


from PIL import Image
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import json
import os
import numpy as np 
import cv2

# MODEL SETUP
model = models.regnet_x_32gf(pretrained=True)
new_classifier = nn.Sequential(*(list(model.children())[:-1]))
model.classifier = new_classifier

# PATHS
vid_dir = './vid_dir/'
frames_dir ='./frames/'
reference_dir = './reference_frames/'
abs_dir = os.path.abspath(os.getcwd())
extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

# GCP SETUP
# Servicekey of GCP storage bucket. Deleted for safety reasons.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "servicekey.json"
storage_client = storage.Client()
bucket = storage_client.get_bucket("mtu")


# GCP FUNCTIONALITY
def upload_to_bucket(blob_name, file_path):
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True
    except Exception as e:
        print(e)
        return False

def delFrames():
    blobs = bucket.list_blobs(prefix='frames/')
    for blob in blobs:
        blob.delete()

def uploadFramesToBucket():
    print("\nUPLOADING TO GOOGLE CLOUD\n")
    for i in os.listdir(frames_dir):
        img_path = os.path.join(frames_dir, i)
        res = upload_to_bucket("frames/{}".format(i), img_path)
        if res:
            print("Uploading {} successful".format(i))
        else:
            print("Upload of {} failed".format(i))


# VIDEO HADLING FUNCTIONALITY
def handleUpload(request):
    vid_dir = './vid_dir/'
    for f in os.listdir(vid_dir):
        os.remove(os.path.join(vid_dir, f))

    file = request.FILES["video"]

    file_name = default_storage.save(vid_dir + file.name, file)
    extractFrames(file_name)

def extractFrames(video_path):
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break

        if idx % 30 == 0:
            cv2.imwrite(os.path.join(abs_dir, "frames/", f"{idx}.png"), frame)
        
        idx += 1   

    delFrames()
    uploadFramesToBucket()
    return None


# VIDEO-ANALYSIS FUNCTIONALITY
def extract_features(path):
    input_image = Image.open(path)
    preprocess = transforms.Compose([transforms.Resize(224), 
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
        
    return output

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                filepath = os.path.join(root, filename)
                if os.path.exists(filepath):
                  file_list.append(filepath)
                else:
                  print(filepath)
    return file_list

def getPathList(name_path):
    name_array = sorted(get_file_list(name_path))
    return name_array

def getFeatures(name_array):
    feature_list = [None] * len(name_array)
    for i, path in enumerate(name_array):
        print("Computing image features [{}/{}]".format(i+1,len(name_array)))
        feat = extract_features(path)
        feat = feat[0].numpy()
        feature_list[i] = feat
    feats = np.asarray(feature_list)
    return feats

def dist(mat, vec, num_elems):
    dists = cdist(mat, np.atleast_2d(vec)).ravel()
    idxs = np.argsort(dists)[:num_elems]
    return idxs

def distPerRef():
    print("\nREFERENCES: \n")
    reference_paths = getPathList(reference_dir)
    reference_feats = getFeatures(reference_paths)

    print("\nFRAMES: \n")
    frame_paths = getPathList(frames_dir)
    frame_feats = getFeatures(frame_paths)

    best_frames = {}
    for i in range(len(reference_paths)):
        print("\nCalculating distance [{}/{}]".format(i+1,len(reference_paths)))
        min_dist_i = dist(frame_feats, reference_feats[i], 1)[0]
        short_ref_path = reference_paths[i].split("/")[-1].split(".")[0]
        best_frames[short_ref_path] = frame_paths[min_dist_i].split("/")[-1]
    print("\n", best_frames)
    return best_frames  

@csrf_exempt
@require_http_methods(["POST"])
def processingPipeling(request):
    handleUpload(request)
    frames_dict = distPerRef()

    return HttpResponse(json.dumps(frames_dict), content_type="json")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/upload/', processingPipeling)
]
