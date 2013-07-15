#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import codecs
import glob

## numpy
import numpy as np

## echonest remix
import echonest.remix.audio as audio
import echonest.remix.action as action

## scikit-learn
from sklearn import cluster

key_string = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

def segmentDistance(segA, segB):
    pitch_dist = np.linalg.norm(np.array(segA.pitches) - np.array(segB.pitches))
    timbre_dist = np.linalg.norm(np.array(segA.timbre) - np.array(segB.timbre))
    loudness_dist = (abs(segA.loudness_begin - segB.loudness_begin) +
                    abs(segA.loudness_max - segB.loudness_max))
    return pitch_dist, timbre_dist, loudness_dist

def computeSimilarityMatrix(segments, p_gamma, t_gamma, l_gamma):
    num_segments = len(segments)
    pD = []
    tD = []
    lD = []
    for i in range(num_segments):
        for j in range(i+1, num_segments):
            dist = segmentDistance(segments[i], segments[j])
            pD.append(dist[0])
            tD.append(dist[1])
            lD.append(dist[2])
    max_pD = np.max(pD)
    max_tD = np.max(tD)
    max_lD = np.max(lD)
    
    ## compute normalized mean of distances
    simMat = np.eye(num_segments)
    
    idx = 0
    for i in range(num_segments):
        for j in range(i+1, num_segments):
            p = pD[idx]/max_pD
            t = tD[idx]/max_tD
            l = lD[idx]/max_lD
            
            pw = np.exp(-p_gamma * p**2)
            tw = np.exp(-t_gamma * t**2)
            lw = np.exp(-l_gamma * l**2)
            
            simMat[i,j] = simMat[j,i] = pw * tw * lw
            idx += 1
    
    return simMat

def computePitchSimilarityMatrix(segments):
    return computeSimilarityMatrix(segments, 1.0, 0.0, 0.0)

def computeTimbreSimilarityMatrix(segments):
    return computeSimilarityMatrix(segments, 0.0, 1.0, 0.0)

def clusterSegmentsByPitch(segments, audio_source, out_directory, pitch_factor, file_prefix=""):
    print "clustering " + str(len(segments)) + " segments by pitch...",
    similarity = computePitchSimilarityMatrix(segments)
    
    p_value = pitch_factor*np.median(similarity)
    CL = cluster.AffinityPropagation(preference=p_value, affinity="precomputed")
    
    CL.fit(similarity)
    labels = CL.labels_
    
    unique_labels = np.unique(labels)
    print len(unique_labels), "clusters found"
    
    for l in unique_labels:
        idx = labels == l
        org_idx = np.nonzero(idx)[0]
        if(len(org_idx)) < 3:
            continue
        sim_sub = similarity[idx,:][:,idx]
        sim_mean = np.mean(sim_sub, axis=0)
        
        sim_idx = np.argmax(sim_mean)
        min_idx = org_idx[sim_idx]
        
        ## save
        seg = segments[min_idx]
        seg.set_source(audio_source)
        
        note = ""
        for p in range(len(seg.pitches)):
            pitch = seg.pitches[p]
            if pitch > 0.3:
                note += key_string[p]
        
        out_filename = file_prefix + str(l).zfill(2) + "_" + note + ".wav"
        out_path = os.path.join(out_directory, out_filename)
        
        seg.encode(out_path)
    
def clusterSegmentsByTimbre(segments_all, audio_source, out_directory, timbre_factor, pitch_factor, file_prefix=""):
    print "rejecting segments..."
    segments = []
    for seg in segments_all:
        if seg.duration > 0.2 and seg.loudness_max > -30:
            segments.append(seg)
    
    num_segments = len(segments)
    print num_segments, "segments accepted"
    
    print "clustering segments by timbre...",
    similarity = computeTimbreSimilarityMatrix(segments)
    
    p_value = timbre_factor*np.median(similarity)
    CL = cluster.AffinityPropagation(preference=p_value, affinity="precomputed")
    
    CL.fit(similarity)
    labels = CL.labels_
    
    unique_labels = np.unique(labels)
    print len(unique_labels), "clusters found"
    
    min_segments = 20
    for label in unique_labels:
        label_indices = np.argwhere(labels == label)
        label_indices = list(label_indices.flatten())
        
        cluster_segments = [segments[i] for i in label_indices]
        
        ## further cluster segments by pitch
        if len(cluster_segments) > min_segments:
            sub_out_directory = os.path.join(out_directory, str(label).zfill(2))
            if not os.path.exists(sub_out_directory):
                os.makedirs(sub_out_directory)
            
            clusterSegmentsByPitch(cluster_segments, audio_source, sub_out_directory, pitch_factor)

def chopAudio(in_filename, out_directory, timbre_factor=0.9, pitch_factor=0.8):
    ## clean up output directory
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    else:
        wav_path = os.path.join(out_directory, "*.wav")
        wavs = glob.glob(wav_path)
        for w in wavs:
            os.remove(w)
    
    ## use saved analysis if exists
    local_analysis = in_filename + ".analysis.en"
    if os.path.exists(local_analysis):
        in_filename = local_analysis
    
    ## load audio file
    audiofile = audio.LocalAudioFile(in_filename)
    clusterSegmentsByTimbre(audiofile.analysis.segments, audiofile.analysis.source, out_directory, timbre_factor, pitch_factor)
    
    ## save audio info
    songinfo_path = os.path.join(out_directory, "info.txt")
    songinfo = codecs.open(songinfo_path, "wb", encoding="utf-8")
    ## ----------------------------------------------------
    track = audiofile.analysis.pyechonest_track
    
    songinfo.write(track.artist + " - " + track.title + "\n")
    
    songinfo.write("Key: " + key_string[track.key])
    songinfo.write(" (" + str(track.key_confidence) + ")\n")
    
    songinfo.write("Mode: " + ("Major" if track.mode == 0 else "Minor"))
    songinfo.write(" (" + str(track.mode_confidence) + ")\n")
    
    songinfo.write("BPM: " + str(track.tempo))
    songinfo.write(" (" + str(track.tempo_confidence) + ")\n")
    ## ----------------------------------------------------
    songinfo.close()
    
    ## save analysis
    audiofile.save()

if __name__=="__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Automatic Audio Chopper via Echo Nest Remix API")
    
    parser.add_argument("input_filename", nargs=1, help="input mp3 file")
    parser.add_argument("-o", "--output_directory", help="output directory (use input filename if not specified)")
    parser.add_argument("--timbre_factor", type=float, default=0.9, help="preference factor for timbre clustering (0.0-1.0)")
    parser.add_argument("--pitch_factor", type=float, default=0.8, help="preference factor for pitch clustering (0.0-1.0)")
    
    args = parser.parse_args()
    
    input_filename = args.input_filename[0]
    
    base, ext = os.path.splitext(input_filename)
    
    if ext == ".mp3":
        if args.output_directory != None:
            output_directory = args.output_directory
        else:
            output_directory = base
        
        chopAudio(input_filename, output_directory, args.timbre_factor, args.pitch_factor)
        
    else:
        print "only accepts mp3 files"
