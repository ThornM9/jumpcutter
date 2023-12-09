from contextlib import closing
from PIL import Image
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import threading
import re
import math
from shutil import copyfile, rmtree
import os
import argparse
from pytube import YouTube
import time

def downloadFile(url):
    name = YouTube(url).streams.first().download()
    newname = name.replace(' ','_')
    os.rename(name,newname)
    return newname

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

def copyFrame(TEMP_FOLDER, inputFrame, outputFrame):
    src = TEMP_FOLDER+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = TEMP_FOLDER+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if outputFrame%20 == 19:
        print(str(outputFrame+1)+" time-altered frames saved.")
    return True

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createPath(s):
    try:  
        os.makedirs(s)
    except OSError:  
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

def deletePath(s): # Dangerous! Watch out!
    try:  
        rmtree(s,ignore_errors=False)
    except OSError:  
        print ("Deletion of the directory %s failed" % s)
        print(OSError)

def extract_filename(directory_path):
    filename_with_extension = os.path.basename(directory_path)
    filename, ext = os.path.splitext(filename_with_extension)
    
    return (filename, ext)

parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
parser.add_argument('--input_file', type=str,  help='the video file you want modified')
parser.add_argument('--input_folder', type=str, help='the folder of video files to modify')
parser.add_argument('--url', type=str, help='A youtube url to download and process')
parser.add_argument('--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name). disregarded when the input is a folder")
parser.add_argument('--silent_threshold', type=float, default=0.03, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")
parser.add_argument('--sounded_speed', type=float, default=1.00, help="the speed that sounded (spoken) frames should be played at. Typically 1.")
parser.add_argument('--silent_speed', type=float, default=5.00, help="the speed that silent frames should be played at. 999999 for jumpcutting.")
parser.add_argument('--frame_margin', type=float, default=1, help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")
parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
parser.add_argument('--frame_rate', type=float, default=30, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
parser.add_argument('--frame_quality', type=int, default=3, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.")
parser.add_argument('--threads', type=int, default=1, help="number of threads to use while processing videos")

args = parser.parse_args()

SAMPLE_RATE = args.sample_rate
SILENT_THRESHOLD = args.silent_threshold
FRAME_SPREADAGE = args.frame_margin
NEW_SPEED = [args.silent_speed, args.sounded_speed]
if args.url != None:
    INPUT_FILES = [downloadFile(args.url)]
elif args.input_folder != None:
    # This assumes that all files in the directory are videos, ffmpeg will raise an error otherwise
    INPUT_FILES = []
    for fn in os.listdir(args.input_folder):    
        if "ALTERED" not in fn:
            INPUT_FILES.append(os.path.join(args.input_folder, fn))
else:
    INPUT_FILES = [args.input_file]
URL = args.url
FRAME_QUALITY = args.frame_quality

def process_file(INPUT_FILE):
    frameRate = args.frame_rate

    start = time.time()
    assert INPUT_FILE != None , "why u put no input file, that dum"
        
    if len(args.output_file) >= 1 and len(INPUT_FILES) == 1:
        OUTPUT_FILE = args.output_file
    else:
        OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

    if os.path.exists(OUTPUT_FILE):
        print(f"{OUTPUT_FILE} already exists, skipping this file")
        return

    TEMP_FOLDER = f"TEMP/{extract_filename(INPUT_FILE)[0]}"
    AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)
        
    createPath(TEMP_FOLDER)

    command = "ffmpeg -i "+INPUT_FILE+" -qscale:v "+str(FRAME_QUALITY)+" "+TEMP_FOLDER+"/frame%06d.jpg -hide_banner"
    subprocess.call(command, shell=True)

    command = "ffmpeg -i "+INPUT_FILE+" -ab 160k -ac 2 -ar "+str(SAMPLE_RATE)+" -vn "+TEMP_FOLDER+"/audio.wav"

    subprocess.call(command, shell=True)

    command = "ffmpeg -i "+TEMP_FOLDER+"/input.mp4 2>&1"
    f = open(TEMP_FOLDER+"/params.txt", "w")
    subprocess.call(command, shell=True, stdout=f)

    sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")
    audioSampleCount = audioData.shape[0]
    maxAudioVolume = getMaxVolume(audioData)

    f = open(TEMP_FOLDER+"/params.txt", 'r+')
    pre_params = f.read()
    f.close()
    params = pre_params.split('\n')
    for line in params:
        m = re.search('Stream #.*Video.* ([0-9]*) fps',line)
        if m is not None:
            frameRate = float(m.group(1))

    samplesPerFrame = sampleRate/frameRate

    audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))

    hasLoudAudio = np.zeros((audioFrameCount))



    for i in range(audioFrameCount):
        start = int(i*samplesPerFrame)
        end = min(int((i+1)*samplesPerFrame),audioSampleCount)
        audiochunks = audioData[start:end]
        maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume
        if maxchunksVolume >= SILENT_THRESHOLD:
            hasLoudAudio[i] = 1

    chunks = [[0,0,0]]
    shouldIncludeFrame = np.zeros((audioFrameCount))
    for i in range(audioFrameCount):
        start = int(max(0,i-FRAME_SPREADAGE))
        end = int(min(audioFrameCount,i+1+FRAME_SPREADAGE))
        shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
        if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]): # Did we flip?
            chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]])

    chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])
    chunks = chunks[1:]

    outputAudioData = np.zeros((0,audioData.shape[1]))
    outputPointer = 0

    lastExistingFrame = None
    for chunk in chunks:
        audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)]
        
        sFile = TEMP_FOLDER+"/tempStart.wav"
        eFile = TEMP_FOLDER+"/tempEnd.wav"
        wavfile.write(sFile,SAMPLE_RATE,audioChunk)
        with WavReader(sFile) as reader:
            with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                tsm.run(reader, writer)
        _, alteredAudioData = wavfile.read(eFile)
        leng = alteredAudioData.shape[0]
        endPointer = outputPointer+leng
        outputAudioData = np.concatenate((outputAudioData,alteredAudioData/maxAudioVolume))

        #outputAudioData[outputPointer:endPointer] = alteredAudioData/maxAudioVolume

        # smooth out transitiion's audio by quickly fading in/out
        
        if leng < AUDIO_FADE_ENVELOPE_SIZE:
            outputAudioData[outputPointer:endPointer] = 0 # audio is less than 0.01 sec, let's just remove it.
        else:
            premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE
            mask = np.repeat(premask[:, np.newaxis],2,axis=1) # make the fade-envelope mask stereo
            outputAudioData[outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask
            outputAudioData[endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask

        startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
        endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
        for outputFrame in range(startOutputFrame, endOutputFrame):
            inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))
            didItWork = copyFrame(TEMP_FOLDER, inputFrame, outputFrame)
            if didItWork:
                lastExistingFrame = inputFrame
            else:
                copyFrame(TEMP_FOLDER, lastExistingFrame,outputFrame)

        outputPointer = endPointer

    wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData)

    # '''
    # outputFrame = math.ceil(outputPointer/samplesPerFrame)
    # for endGap in range(outputFrame,audioFrameCount):
    #     copyFrame(int(audioSampleCount/samplesPerFrame)-1,endGap)
    # '''

    command = "ffmpeg -framerate "+str(frameRate)+" -i "+TEMP_FOLDER+"/newFrame%06d.jpg -i "+TEMP_FOLDER+"/audioNew.wav -strict -2 "+OUTPUT_FILE
    subprocess.call(command, shell=True)

    deletePath(TEMP_FOLDER)

    end = time.time()
    print(f"Completed {INPUT_FILE} in {end - start}s")


def process_files(thread_id, files_per_thread):
    for i in range(thread_id * files_per_thread, min((thread_id + 1) * files_per_thread, len(INPUT_FILES))):
        process_file(INPUT_FILES[i])


def main():
    num_threads = min(args.threads, os.cpu_count(), len(INPUT_FILES)) # can't be more threads than available cores or files to process
    files_per_thread = len(INPUT_FILES) // num_threads # spread the work evenly as possible across threads
    remaining_files = (len(INPUT_FILES) % num_threads) + 1
    threads = []

    for i in range(num_threads):
        thread_files = files_per_thread
        if i < remaining_files:
            thread_files += 1

        thread = threading.Thread(target=process_files, args=(i, thread_files))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()

