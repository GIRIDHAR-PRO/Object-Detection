# import cv2
# import numpy as np
# import os

# class Detector:
#     def __init__(self, videoPath, configPath, modelPath, classesPath):
#         self.videoPath = videoPath
#         self.configPath = configPath
#         self.modelPath = modelPath
#         self.classesPath = classesPath

#         # Initialize the model
#         self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
#         self.net.setInputSize(320, 320)
#         self.net.setInputScale(1.0 / 255)  # Correct scale value
#         self.net.setInputMean((127.5, 127.5, 127.5))
#         self.net.setInputSwapRB(True)

#         self.readClasses()

#     def readClasses(self):
#         with open(self.classesPath, 'r') as f:
#             self.classesList = f.read().splitlines()  # Read class names
#         self.classesList.insert(0, '__Background__')  # Add background class
#         self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))  # Random colors
#         print(self.classesList)

#     def onVideo(self):
#         cap = cv2.VideoCapture(self.videoPath)

#         if not cap.isOpened():
#             print("Error opening video file")
#             return

#         while True:
#             success, image = cap.read()
#             if not success:
#                 break

#             classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.5)
#             bboxs = list(bboxs)
#             confidences = list(confidences.flatten())
            
#             bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

#             if len(bboxIdx) > 0:
#                 for i in bboxIdx.flatten():
#                     bbox = bboxs[i]
#                     classConfidence = confidences[i]
#                     classLabelID = classLabelIDs[i]
#                     classLabel = self.classesList[classLabelID]
#                     classColor = [int(c) for c in self.colorList[classLabelID]]

#                     displayText = "{}:{:.2f}".format(classLabel, classConfidence)
#                     x, y, w, h = bbox
#                     cv2.rectangle(image, (x, y), (x+w, y+h), color=classColor, thickness=2)
#                     cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, 2)

#             # Display image using OpenCV
#             cv2.imshow("Detection", image)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# def main():
#     videoPath = "D:/#Team - 5 Mini Project\Y2meta.app-Cars, Busy Streets, City Traffic - No Copyright Royalty Free Stock Videos.mp4"

#     configPath = os.path.join("model_data", "D:/#Team - 5 Mini Project/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
#     modelPath = os.path.join("model_data", "D:/#Team - 5 Mini Project/frozen_inference_graph.pb")
#     classesPath = os.path.join("model_path", "D:/#Team - 5 Mini Project/coco.names")


#     detector = Detector(videoPath, configPath, modelPath, classesPath)
#     detector.onVideo()

# if __name__ == '__main__':
#     main()


import cv2
import numpy as np
import os

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Initialize the model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 255)  # Correct scale value
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()  # Read class names
        self.classesList.insert(0, '__Background__')  # Add background class
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))  # Random colors
        print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening video file")
            return

        while True:
            success, image = cap.read()
            if not success:
                break

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.5)
            bboxs = list(bboxs)
            confidences = list(confidences.flatten())
            
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) > 0:
                for i in bboxIdx.flatten():
                    bbox = bboxs[i]
                    classConfidence = confidences[i]
                    classLabelID = classLabelIDs[i]
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x+w, y+h), color=classColor, thickness=2)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, 2)

            # Resize image to a medium size (e.g., 640x480)
            image_resized = cv2.resize(image, (640, 480))

            # Display image using OpenCV
            cv2.imshow("Detection", image_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    videoPath = "D:/#Team - 5 Mini Project\Y2meta.app-Cars, Busy Streets, City Traffic - No Copyright Royalty Free Stock Videos.mp4"

    configPath = os.path.join("model_data", "D:/#Team - 5 Mini Project/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "D:/#Team - 5 Mini Project/frozen_inference_graph.pb")
    classesPath = os.path.join("model_path", "D:/#Team - 5 Mini Project/coco.names")


    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()
