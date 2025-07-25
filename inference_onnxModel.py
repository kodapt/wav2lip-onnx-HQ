import os, sys
import subprocess
import platform
import numpy as np
import cv2
import argparse
import audio
import shutil
import librosa
from os import listdir, path
from tqdm import tqdm
from PIL import Image
from scipy.io.wavfile import write
import gc

import onnxruntime
onnxruntime.set_default_logger_severity(3)

# Resolve all resource files relative to script location (not cwd)
def resource_path(rel_path):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

# face detection and alignment
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256

# specific face selector
from faceID.faceID import FaceRecognition

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--denoise', default=False, action="store_true", help="Denoise input audio to avoid unwanted lipmovement")
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', default='results/result_voice.mp4')
parser.add_argument('--hq_output', default=False, action='store_true',help='HQ output')

parser.add_argument('--static', default=False, action='store_true', help='If True, then use only first video frame for inference')
parser.add_argument('--pingpong', default=False, action='store_true',help='pingpong loop if audio is longer than video')

parser.add_argument('--cut_in', type=int, default=0, help="Frame to start inference")
parser.add_argument('--cut_out', type=int, default=0, help="Frame to end inference")
parser.add_argument('--fade', action="store_true", help="Fade in/out")

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
parser.add_argument('--resize_factor', default=1, type=int, help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--enhancer', default='none', choices=['none', 'gpen', 'gfpgan', 'codeformer', 'restoreformer'])
parser.add_argument('--blending', default=10, type=float, help='Amount of face enhancement blending 1 - 10')
parser.add_argument('--sharpen', default=False, action="store_true", help="Slightly sharpen swapped face")
parser.add_argument('--frame_enhancer', action="store_true", help="Use frame enhancer")

parser.add_argument('--face_mask', action="store_true", help="Use face mask")
parser.add_argument('--face_occluder', action="store_true", help="Use x-seg occluder face mask")

parser.add_argument('--pads', type=int, default=4, help='Padding top, bottom to adjust best mouth position, move crop up/down, between -15 to 15') # pos value mov synced mouth up
parser.add_argument('--face_mode', type=int, default=0, help='Face crop mode, 0 or 1, rect or square, affects mouth opening' )

parser.add_argument('--preview', default=False, action='store_true', help='Preview during inference')
parser.add_argument('--headless', default=False, action='store_true', help="Run in headless mode (Colab/Docker, disables OpenCV windows)")

args = parser.parse_args()

if args.checkpoint_path.endswith('wav2lip_384.onnx') or args.checkpoint_path.endswith('wav2lip_384_fp16.onnx'):
    args.img_size = 384
else:
    args.img_size = 96

mel_step_size = 16
padY = max(-15, min(args.pads, 15))

device = 'cpu'
if onnxruntime.get_device() == 'GPU':
    device = 'cuda'
print("Running on " + device)

# Load detector/recognition AFTER args parsing!
providers = ["CPUExecutionProvider"]
if device == 'cuda':
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]

# --- FIXED: all model file paths resolve relative to script ---
detector = RetinaFace(resource_path("utils/scrfd_2.5g_bnkps.onnx"), provider=providers, session_options=None)
recognition = FaceRecognition(resource_path('faceID/recognition.onnx'))

if args.enhancer == 'gpen':
    from enhancers.GPEN.GPEN import GPEN
    enhancer = GPEN(model_path=resource_path("enhancers/GPEN/GPEN-BFR-256-sim.onnx"), device=device) #GPEN-BFR-256-sim

if args.enhancer == 'codeformer':
    from enhancers.Codeformer.Codeformer import CodeFormer
    enhancer = CodeFormer(model_path=resource_path("enhancers/Codeformer/codeformerfixed.onnx"), device=device)
    
if args.enhancer == 'restoreformer':
    from enhancers.restoreformer.restoreformer16 import RestoreFormer
    enhancer = RestoreFormer(model_path=resource_path("enhancers/restoreformer/restoreformer16.onnx"), device=device)
        
if args.enhancer == 'gfpgan':
    from enhancers.GFPGAN.GFPGAN import GFPGAN
    enhancer = GFPGAN(model_path=resource_path("enhancers/GFPGAN/GFPGANv1.4.onnx"), device=device)

if args.frame_enhancer:
    from enhancers.RealEsrgan.esrganONNX import RealESRGAN_ONNX
    frame_enhancer = RealESRGAN_ONNX(model_path=resource_path("enhancers/RealEsrgan/clear_reality_x4.onnx"), device=device)
    
if args.face_mask:
    from blendmasker.blendmask import BLENDMASK
    masker = BLENDMASK(model_path=resource_path("blendmasker/blendmasker.onnx"), device=device)
        
if args.face_occluder:
    from xseg.xseg import MASK
    occluder = MASK(model_path=resource_path("xseg/xseg.onnx"), device=device)

if args.denoise:
    from resemble_denoiser.resemble_denoiser import ResembleDenoiser
    denoiser = ResembleDenoiser(model_path=resource_path('resemble_denoiser/denoiser.onnx'), device=device)
                                    
if os.path.isfile(args.face) and args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
    args.static = True

def load_model(device):
    model_path = args.checkpoint_path
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if device == 'cuda':
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)	
    return session

def select_specific_face(model, spec_img, size, crop_scale=1.0):
    h, w = spec_img.shape[:-1]
    if args.headless:
        roi = (0, 0, w, h)
    else:
        roi = cv2.selectROI("Select speaker face", spec_img, showCrosshair=False)
        if roi == (0,0,0,0): roi = (0,0,w,h)
        cv2.destroyAllWindows()
    cropped_roi = spec_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    bboxes, kpss = model.detect(cropped_roi, input_size = (320,320), det_thresh=0.3)
    assert len(kpss) != 0, "No face detected"
    target_face, mat = get_cropped_head_256(cropped_roi, kpss[0], size=size, scale=crop_scale)
    target_face = cv2.resize(target_face,(112,112))
    target_id = recognition(target_face)[0].flatten()
    return target_id

# ... your unchanged function definitions for process_video_specific, face_detect, datagen go here ...
# (not repeating them for brevity, but leave them as they are in your script)

def main():
    # ... your unchanged setup code ...
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    if args.face.split('.')[-1] in ['jpg', 'png', 'jpeg', 'bmp']:
        orig_frame = cv2.imread(args.face)
        orig_frame = cv2.resize(orig_frame, (orig_frame.shape[1]//args.resize_factor, orig_frame.shape[0]//args.resize_factor))	
        orig_frames = [orig_frame]
        fps = args.fps

        h, w = orig_frame.shape[:-1]
        if args.headless:
            roi = (0, 0, w, h)
        else:
            roi = cv2.selectROI("Crop final video", orig_frame, showCrosshair=False)
            if roi == (0,0,0,0): roi = (0,0,w,h)
            cv2.destroyAllWindows()
        cropped_roi = orig_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        full_frames = [cropped_roi]
        orig_h, orig_w = cropped_roi.shape[:-1]
        target_id = select_specific_face(detector, cropped_roi, 256, crop_scale=1)
            
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        video_stream.set(1,args.cut_in)
        print('Reading video frames...')
        if args.cut_out == 0:
            args.cut_out = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) - args.cut_in
        new_duration = args.cut_out - args.cut_in
        if args.static:
            new_duration = 1
        video_stream.set(1,args.cut_in)
        full_frames = []
        orig_frames = []
        for l in range(new_duration):
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))			
            if l == 0:
                h, w = frame.shape[:-1]
                if args.headless:
                    roi = (0, 0, w, h)
                else:
                    roi = cv2.selectROI("Crop final video", frame, showCrosshair=False)
                    if roi == (0,0,0,0): roi = (0,0,w,h)
                    cv2.destroyAllWindows()
                cropped_roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                # os.system('cls') # Remove for Colab/Unix compatibility
                target_id = select_specific_face(detector, cropped_roi, 256, crop_scale=1)
                orig_h, orig_w = cropped_roi.shape[:-1]
                print("Reading frames....")
            print(f'\r{l}', end=' ', flush=True)
            cropped_roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            full_frames.append(cropped_roi)
            orig_frames.append(cropped_roi)

    # --------- PREPROCESS AUDIO AND MEL ----------
    print('Extracting raw audio...')
    os.makedirs('temp', exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-i', args.audio, '-ac', '1', '-strict', '-2', 'temp/temp.wav'])

    print('Raw audio extracted')

    if args.denoise:
        print('Denoising audio...')
        wav, sr = librosa.load('temp/temp.wav', sr=44100, mono=True)
        wav_denoised, new_sr = denoiser.denoise(wav, sr, batch_process_chunks=False)
        write('temp/temp.wav', new_sr, (wav_denoised * 32767).astype(np.int16))
        try:
            if hasattr(denoiser, 'session'):
                del denoiser.session
                gc.collect()
        except:
            pass

    wav = audio.load_wav('temp/temp.wav', 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    print("Length of mel chunks: {}".format(len(mel_chunks)))
    full_frames = full_frames[:len(mel_chunks)]

    # --------- FACE DETECTION ---------
    aligned_faces, sub_faces, matrix, no_face = face_detect(full_frames, target_id)

    if args.pingpong:
        orig_frames = orig_frames + orig_frames[::-1]
        full_frames = full_frames + full_frames[::-1]
        aligned_faces = aligned_faces + aligned_faces[::-1]
        sub_faces = sub_faces + sub_faces[::-1]
        matrix = matrix + matrix[::-1]
        no_face = no_face + no_face[::-1]

    # --------- GENERATOR ---------
    gen = datagen(sub_faces.copy(), mel_chunks)

    # --------- REST OF INFERENCE LOOP (UNCHANGED) ---------
    # ... (Your full existing inference code here) ...

if __name__ == '__main__':
    main()
