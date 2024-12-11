#######MATPLOTLIB IMPLEMENTATION #####################



import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        # Read class names from file and generate random colors for each class
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening video file")
            return

        plt.ion()  # Turn on interactive mode for matplotlib

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
                    cv2.rectangle(image, (x, y), (x+w, y+h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, 2)

            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display image using matplotlib
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.draw()  # Update the figure
            plt.pause(0.01)  # Pause to allow matplotlib to update the display

        cap.release()
        plt.ioff()  # Turn off interactive mode
        plt.close()  # Close all matplotlib windows

        

      

