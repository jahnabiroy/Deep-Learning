import sys
if __name__ == "__main__": 
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "default/model/path"


# import torch
# import torch.nn as nn



# class birdClassifier(nn.Module):
#     def __init__(self):
#         super(birdClassifier, self).__init__()
        
        


if trainStatus == "train":
    print("training")
else:
    print("infer")
print(f"Training: {trainStatus}")
print(f"path to dataset: {dataPath}")
print(f"path to model: {modelPath}")
