from Detector import *
import os

def main():
    videoPath = "D:/Projects/TEAM-5(MINI_PROJECT)/vehicles.mp4"

    configPath = os.path.join("model_data", "D:/Projects/TEAM-5(MINI_PROJECT)/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "D:/Projects/TEAM-5(MINI_PROJECT)/frozen_inference_graph.pb")
    classesPath = os.path.join("model_path", "D:/Projects/TEAM-5(MINI_PROJECT)/coco.names")


    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()



