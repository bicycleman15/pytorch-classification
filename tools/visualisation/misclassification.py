import os
import cv2 
import torch
import pickle
import numpy as np
import seaborn as sn
from tqdm import tqdm
from torch.utils import data
import matplotlib.pyplot as plt
import torchvision.datasets as datasets



def makeDir (dir):
    if(not os.path.isdir(dir)):
        os.makedirs(dir)


class saver():
    def __init__(self, prefix, dataset, classes):
        self.prefix = prefix
        self.dataset = dataset
        self.classes = classes
        self.misclassified = dict()
        self.misclassified_set = set()
        self.image_id_counter = 0

        self.data_collector_path = "outputs/misclassification"
        makeDir(self.data_collector_path)

        self.dataset_dir = os.path.join(self.data_collector_path, dataset)
        makeDir(self.dataset_dir)
        
        self.prefix_dir = os.path.join(self.dataset_dir, prefix)
        makeDir(self.prefix_dir)

    def forward(self, image_id, prob, target ):
        '''
        image_id: from a non shuffled test set
        prob: numpy Probability vector [b, n_class]
        target: Ground Truth  [b, 1] 

        Assuming batch size is 1
        '''

        # reducing the  extra first axis
        prob = np.squeeze(prob, axis = 0)
        target = np.squeeze(target, axis = 0)
        saveData = []
        if(np.argmax(prob) != target):
            self.misclassified_set.add(image_id)
            for (p, c) in zip(prob, self.classes):
                saveData.append([p,c])

            saveData.sort(key = lambda x: x[0], reverse = True)
            self.misclassified[image_id] = saveData


    def batch_forward(self, probs, targets ):
        '''
        prob: torch Probability vector [b, n_class]
        target: Ground Truth  [b] 

        Assuming batch size is b
        '''
        for prob , target in tqdm(zip(probs, targets)):
            # it will be squeezed
            saveData = []
            if(np.argmax(prob) != target):
                self.misclassified_set.add(self.image_id_counter)
                for (p, c) in zip(prob, self.classes):
                    saveData.append([p,c])

                saveData.sort(key = lambda x: x[0], reverse = True)
                self.misclassified[self.image_id_counter] = saveData

            self.image_id_counter += 1

        self.unload_data()

    def unload_data(self):
        file_path = os.path.join(self.prefix_dir , "dict.pickle")
        file = open(file_path, "wb")
        pickle.dump(self.misclassified, file)
        file.close()

        file_path = os.path.join(self.prefix_dir , "set.pickle")
        file = open(file_path, "wb")
        pickle.dump(self.misclassified_set, file)
        file.close()

class retiever():
    def __init__(self, dataset, classes, outfolder = "outputs/misclassification/results_images"):
        self.dataset = dataset

        self.data_collector_path = "outputs/misclassification"

        self.dataset_dir = os.path.join(self.data_collector_path, dataset)

        self.prefix_list = os.listdir(self.dataset_dir)

        self.out_folder = outfolder
        makeDir(self.out_folder)

        print("Found the following runs")
        for i , prefix in enumerate(self.prefix_list):
            print(f"[ {i} : {prefix} ]")

        self.misclass_dict = []
        self.misclass_set = []
        self.final_set = []
        self.classes = classes
        self.preferences = []

        self.load_data()




    def set_perferences(self, preferences):
        '''
        preferences the index of runs that we want to merge
        '''
        # empty initialisation

        self.preferences = preferences
        self.final_set = []

        self.final_set = self.misclass_set[preferences[0]]

        for p in preferences:
            self.final_set = self.final_set.intersection(self.misclass_set[p])


        print(f"Final set prepared with {len(self.final_set)} common misclassifications")

        
        

        
    def load_data(self):
        self.misclass_dict = []
        self.misclass_set = []
        for i, prefix in enumerate(self.prefix_list):
            file_path = os.path.join(self.dataset_dir, prefix, "dict.pickle")
            file = open(file_path, "rb")
            self.misclass_dict.append(pickle.load(file))
            file.close()

            file_path = os.path.join(self.dataset_dir, prefix, "set.pickle")
            file = open(file_path, "rb")
            self.misclass_set.append(pickle.load(file))
            file.close()


    def save_imgs(self, dataset):
        '''
        Assuming batch size is 1
        '''

        print("Setting up to save images out")

        stopper = 0
        for img_no in tqdm(self.final_set):
            # print(f"Printing image {img_no}")
            if(stopper > 10):
                break
            stopper += 1
            plt.clf()
            img , tar = dataset[img_no]
            img_numpy = np.array(img)

            n = len(self.preferences)
            plt.rcParams["figure.figsize"] = (3*n,5)
            figure, axis = plt.subplots(1, n)


            '''
            figure, axis = plt.subplots(1, 1)
            returns a figure with only one single subplot, 
            so axs already holds it without indexing.

            figure, axis = plt.subplots(1, n)
            gives a linear axes

            figure, axis = plt.subplots(1, n)
            gives a 2D
            '''

            # TODO optimise for 2D usecase
            if(n == 1 ):
                axis = [axis]

            for i, preference in enumerate(self.preferences):
                axis[i].imshow(img_numpy)

                # print prefix
                
                axis[i].text(0, (1.2)*img_numpy.shape[0], f"Run : {self.prefix_list[preference]}")
                axis[i].text(0, (1.4)*img_numpy.shape[0], f"GT : {self.classes[tar]}")

                for j,(conf, class_name) in enumerate(self.misclass_dict[preference][img_no]):
                    axis[i].text(0, (1.5 + j*0.1)*img_numpy.shape[0], "{} : {:.4}]".format(class_name, conf))


            figure.savefig(os.path.join(self.out_folder,f"{img_no}.jpg"), bbox_inches='tight')
