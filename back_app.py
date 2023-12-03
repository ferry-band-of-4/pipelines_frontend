from fastapi import FastAPI, UploadFile, File
import os
import glob
import mne
import cv2
import meegkit.asr as asr
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from ghostipy.spectral.wavelets import MorseWavelet
import ghostipy
import numpy as np
import subprocess
import gc
from memory_profiler import profile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from io import BytesIO


class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = os.listdir(folder_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.images[idx])
        channel_name = self.images[idx].split(".")[0]   

        image = Image.open(img_name).convert("RGB")
        

        if self.transform:
            image = self.transform(image)

        return image, channel_name
    
class resnet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        self.convolutional_layers = nn.Sequential(resnet.conv1,
                                   resnet.bn1,
                                   resnet.relu,
                                   resnet.maxpool,
                                   resnet.layer1,
                                   resnet.layer2,
                                   resnet.layer3,
                                   resnet.layer4)

        # # Add back the batch norm layer that we removed
        self.avgpool = resnet.avgpool # Global Average pooling layer

        # # # Custom fully connect layer
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes) 

        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.convolutional_layers(x)
    
    def forward(self, images):        
        # Convolutional layers of resnet18
        x = self.convolutional_layers(images) 
               
        # # The layers after the Conv you used the hook on
        h = x.register_hook(self.activations_hook)
        
        # Global average pooling layer from resnet18
        x = self.avgpool(x)
        
        x = x.reshape(x.size(0), -1)
        logits = self.fc(x)
        
        output = F.sigmoid(logits)
        return output
    

@profile
def plot_heatmap(denorm_image, pred, heatmap, channel_name):

    plt.close('all')

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20,20), ncols=3)

    classes = ['schizophrenic', 'healthy']
    ps = F.sigmoid(pred).cpu().detach().numpy()
    ax1.imshow(denorm_image)
    ax1.axis('off')

    ax2.barh(classes, [pred[0].item(), 1-pred[0].item()])
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    ax3.imshow(denorm_image)
    ax3.imshow(heatmap, cmap='jet', alpha=0.7)
    ax3.axis('off')

    plt.savefig(f"grad_cams/{channel_name}.png", bbox_inches='tight', pad_inches=1, format='png')
    

@profile
def get_gradcam(model, image, prediction, size):
    print("prediction...........")
    print(prediction)
    prediction.backward(retain_graph=True)    
    gradients = model.get_gradient()
    pooled_gradients = torch.mean(gradients, dim = [0, 2, 3])
    activations = model.get_activations(image).detach() # A1, A2, ..., Ak
    
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim = 1).squeeze().cpu()
    heatmap = nn.ReLU()(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (size, size))
    
    return heatmap

def mutli_grads(model, images, predictions, channel_names,size):
   # heatmaps = []
   # print("predictionssssssssss...........")
    for i in range(len(images)):
       # print(predictions[i])
        heatmap = get_gradcam(model, images[i].unsqueeze(0), predictions[i], size)
        denorm_image= images[i].permute(1, 2, 0) * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
        plot_heatmap(denorm_image, predictions[i], heatmap,channel_names[i])
        #plt.imsave(f"grad_cams/{channel_names[i]}.png", heatmap, cmap='jet')
        




app = FastAPI()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet(num_classes=1).to(device)
model_num = 20
checkpoint_path = ""
checkpoint = torch.load(f'resnet-finetune-adam-epoch_{model_num}.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

@profile
def plot_figs(path, target_path):
    sfreq = 250
    raw = mne.io.read_raw_edf(path, preload=True)
    train_idx = np.arange(0, 30*sfreq, dtype=int)
    gmw = MorseWavelet(gamma=2, beta=1)
    asr_instance = asr.ASR(method='euclid')
    channels = raw.ch_names
    data = raw._data
    asr_data, _ =  asr_instance.fit(data[:, train_idx])

    dict_eeg = {channels[i]: asr_data[i] for i in range(19)}
    

    i=0
    for channel_name , channel_data in dict_eeg.items():
        Wxh, *_ = ghostipy.spectral.cwt(channel_data, wavelet=gmw, voices_per_octave=10)
        #plt.figure(figsize=(8,6))
        plt.imshow(np.abs(Wxh), aspect='auto', cmap='turbo')
        plt.axis('off')
        image_file_name = channel_name+".png"
        image_path = os.path.join(target_path, image_file_name)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, format='png')
        print(plt.gcf().get_size_inches())
        plt.close()
        i+=1
        subprocess.run(['sync'])
        gc.collect()
    gc.collect()
    plt.close('all')

    print(dict_eeg, asr_data)    
    return True

@profile
def image_transform(image_folder):

    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_file in image_files:
    
        input_path = os.path.join(image_folder, image_file)
        image = Image.open(input_path)

        if image.mode == 'RGBA':
            image = image.convert('RGB')
        

        transformed_image = preprocess(image)

        pil_tranformed_image = transforms.ToPILImage()(transformed_image)

        pil_tranformed_image.save(input_path)
    
    print("transformation to RGB completed")

    return True

@profile
def dir_cleaner(edf_buffer:str, image_buffer:str):
    image_patter_dir = os.path.join(image_buffer, '*.png')
    edf_pattern_dir = os.path.join(edf_buffer, "*.edf")
    grad_cam_pattern_dir = os.path.join("grad_cams", "*.png")
    
    img_files_glob = glob.glob(image_patter_dir)
    efd_files_glob = glob.glob(edf_pattern_dir)
    grad_cam_files_glob = glob.glob(grad_cam_pattern_dir)

    for file_path in img_files_glob:
        try:
            os.remove(file_path)
            #print(f"deleted {file_path}")
        except Exception as e:
            print(e)
            print(f" error deleting image {file_path}")
            return False
    for file_path in efd_files_glob:
        try:
            os.remove(file_path)
        except Exception as e:
            print(e)
            print(f"error deleting edf {file_path}")
            return False
    for file_path in grad_cam_files_glob:
        try:
            os.remove(file_path)
        except Exception as e:
            print(e)
            print(f"error deleting grad_cam {file_path}")
            return False

    return True


@profile
def new_preprocess(image_folder):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  

    ])

    custom_dataset = CustomDataset(image_folder, transform)
    batch_size = 19
    data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    return data_loader


@profile
def plot_predictions(prediction_dict):
    plt.close('all')
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'

    plt.figure(figsize=(20, 10))
    colors = ['tomato' if value > 0.5 else 'skyblue' for value in prediction_dict.values()]
    values = [values if values > 0.5 else 1-values for values in prediction_dict.values()]
    bars = plt.bar(prediction_dict.keys(), values, color=colors)
    plt.yticks(color='white')
    plt.xticks(rotation=90, color='white')  
    plt.bar_label(plt.bar(prediction_dict.keys(), values, color=colors), labels=[round(value, 2) for value in values], padding=3, color='white')
    plt.ylim(0, 1.5)


    # Add legend with custom labels
    if "tomato" not in colors:
        legend_labels = ['Healthy']
    elif "skyblue" not in colors:
        legend_labels = ['Schizophrenic']
    else:
        legend_labels = ['Healthy', 'Schizophrenic']

    
    legend = plt.legend(bars, legend_labels, loc='upper right', fontsize='large', frameon=False, title='Class',  labelcolor='white')

    # Set legend title color
    legend.get_title().set_color('white')

    # Set color of X and Y axes to white
    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    plt.savefig("predictions.png", bbox_inches='tight', pad_inches=0, format='png', facecolor='black')
    plt.close()
    return True


@profile
def model_prediction(data_loader, model):
    model.eval()   
    prediction_dict = {}
    actual_preds ={}
    for images, channel_names in data_loader:
        images = images.to(device)
        outputs = model(images)
        # print("outttttttttpuuuuuuuuuttttttt")
        # print(outputs)
        # print(outputs[0])
        # print("ourputttttt overrrrrrrrrr")
        mutli_grads(model, images, outputs,channel_names, 224)
        round_outputs = torch.round(outputs)
        
        for i in range(len(round_outputs)):
            prediction_dict[channel_names[i]] = round_outputs[i].item()
            actual_preds[channel_names[i]] = outputs[i].item()

    plot_predictions(actual_preds)
    
    return prediction_dict


    

UPLOAD_FOLDER = "edf_buffer"
IMAGE_BUFFER = "image_buffer"

def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    subprocess.run(['sync'])
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file_name = file.filename
    # Create the upload folder if it doesn't exis
    plt.close('all')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    

    #raw = mne.io.read_raw_edf('edf_buffer/h01.edf', preload=True)
    plot_figs(file_path,IMAGE_BUFFER)
    #image_transform(IMAGE_BUFFER)

    #print(raw.__dict__)
    

    # os.remove(file_path)
    # folder_path = "image_buffer"
    # png_files = glob.glob(os.path.join(folder_path, "*.png"))
    # for file in png_files:
    #     os.remove(file)
    return {"filename": file_name, "saved_path": file_path, "removed":True}

@app.get("/clear_dir")
def clear_dir():
    dir_cleaner(UPLOAD_FOLDER,IMAGE_BUFFER)

    return True

@app.get("/predict")
def predict():
    data_loader = new_preprocess(IMAGE_BUFFER)
    prediction_dict = model_prediction(data_loader, model)
    values = list(prediction_dict.values())
    count =0
    for value in values:
        if value == 1:
            count+=1

    if count >= 19 - count:
        prediction_dict["prediction"] = "Schizo positive"
    else:
        prediction_dict["prediction"] = "Schizo negative"
    
    

    return prediction_dict

@app.get("/predict_steamlit")
async def predict_steamlit(name: str):
    print(name)
    file_path = os.path.join(UPLOAD_FOLDER, name)
    print(file_path)


    plot_figs(file_path,IMAGE_BUFFER)
    return {"name": name}
    


if __name__ == "__main__":
    main()
