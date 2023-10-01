import mne
from MaxViT.MaxDaVit import MaxDaViT
import torch
import matplotlib.pyplot as plt
import numpy as np
import requests

#only use cpu, to be able to run on every device
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data from edf directly from website https://physionet.org/content/eegmmidb/1.0.0/
def getEDFFile(url):
    response = requests.get(url)
    local_file_path = "temp_file.edf"
    with open(local_file_path, "wb") as file:
        file.write(response.content)
    raw = mne.io.read_raw_edf(local_file_path, preload=True)
    return raw

url = "https://physionet.org/files/eegmmidb/1.0.0/S010/S010R03.edf"
raw = getEDFFile(url)

# Load the model
maxDaVit = MaxDaViT(num_classes=5,dim=32,depth=(1,1,2,1),dim_conv_stem=3,window_size=(10,4))
maxDaVit.load_state_dict(torch.load("KI/MaxViT/MaxVit11.pt", map_location=device))
maxDaVit.eval()
maxDaVit.to(device)

mean, std = torch.load("KI/Dataset/MotorImgTorch_mean_std.pt")

print(raw.annotations.description)

seq_onsets = np.arange(raw.get_data()[0].shape[0])/160

runs_classes_1 = [5, 6, 9, 10, 13, 14]#different runs correspond to different classes
def get_classes_for_seqs(data_edf,run):
  #get class of run: if the run is not class one, the result is zero
  run_class = np.sum(np.equal(runs_classes_1,run))
  x_one_seq = data_edf.get_data()
  x_seqs = []
  y_seqs = []
  last_index = x_one_seq[0].shape[0]
  ann_len = data_edf.annotations.onset.shape[0]
  for i in range(ann_len-1,-1,-1):#go through ann and save y_data as seperate seqs
    try:
      cur_index = np.where(np.equal(data_edf.annotations.onset[i],seq_onsets))[0][0]#get index where x==ann(start of new_seq)
    except:
      continue

    # pad data with zeros if the seq is too short, if to long, cut it off
    x_seqs_cur = x_one_seq[:,cur_index:last_index]
    if(last_index-cur_index<736):
      if(run<3):
        continue
      x_seqs_cur = np.pad(x_seqs_cur, pad_width=((0,0),(0,736-(last_index-cur_index))), mode='constant', constant_values=0)
    elif(last_index-cur_index>736 and run>2):
      x_seqs_cur = x_seqs_cur[:,:736]

    x_seqs.append(x_seqs_cur)
    annotation_int = int(data_edf.annotations.description[i][1])
    if annotation_int==0:
      y_seqs.append(0)
    elif annotation_int==1:
      y_seqs.append(1+run_class)
    elif annotation_int==2:
      y_seqs.append(3+run_class)
    last_index = cur_index

  x_seqs.reverse()
  y_seqs.reverse()

  x_seqs = torch.tensor(x_seqs,dtype=torch.float32)
  y_seqs = torch.tensor(y_seqs,dtype=torch.int64)
  
  # if run==1 or 2: divide the data in non-overlapping windows with 610 samples: 3.8s
  if(run==1 or run==2):
    x_seqs = x_seqs.unfold(2, 610, 610).squeeze(0).transpose(0,1)
    y_seqs = y_seqs.expand(610)

  # window the data with 75% overlapping windows
  x_seqs = x_seqs.unfold(2, 160, 40).transpose(1,2).transpose(2,3)

  return x_seqs,y_seqs

# test_x, test_y = get_classes_for_seqs(raw,2)
# print(test_x.shape)

def maxDaViTInference(x):
    with torch.no_grad():
        x = (x-mean)/std
        x = x.unsqueeze(1).to(device)
        y_pred = maxDaVit(x)
        
    return (torch.sum(y_pred,0)/x.shape[0]).unsqueeze(0)

print("choose a file: first choose number from 1-109 and then choose a number from 1-14 ")
print("format:\n1-109\n1-14\n")

subject = input()
run = input()

file_path = f"https://physionet.org/files/eegmmidb/1.0.0/S{subject.zfill(3)}/S{subject.zfill(3)}R{run.zfill(2)}.edf"
print(f"file_path: {file_path}")

run = int(run)

raw = getEDFFile(file_path)

x_seqs, y_seqs = get_classes_for_seqs(raw,run)

classes = ["rest", "left fist", "both fists", "right fist", "both feet"]

fig = plt.figure(figsize=(15,8))
fig.suptitle(file_path)
for i in range(1, 11):
    fig.add_subplot(2, 5, i)
    plt.imshow(x_seqs[i][0])
    plt.title(classes[y_seqs[i].cpu().numpy()])
    plt.axis("off")

j = 0
for x,y in zip(x_seqs,y_seqs):
    y_pred = maxDaViTInference(x)
    y = y.unsqueeze(0).to(device)
    y_pred = torch.argmax(y_pred, dim=1)
    print(f"image {j}: ")
    print(f"y_pred: {classes[y_pred.cpu().numpy()[0]]}, y: {classes[y.cpu().numpy()[0]]}")
    print("-----------------------------------")
    j += 1

plt.show()