import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from anomalib.deploy import OpenVINOInferencer
from anomalib.data import InferenceDataset, TaskType
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import read_image
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.models.components import FeatureExtractor
from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from multiprocessing import Process, Queue, freeze_support
from pathlib import Path
import cv2
import glob
import time
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import torch
import h5py
import time
import pickle


def datamodule_activate_func(datamodule):
    datamodule.prepare_data()  # Downloads the dataset if it's not in the specified `root` directory
    datamodule.setup()  # Create train/val/test/prediction sets.
    print('1. func ++++++++++++++++++')
    i, data = next(enumerate(datamodule.val_dataloader()))
    # print("data keys:", data.keys())
    # print(data["image"].shape)


def train_and_val_func(result_queue,model_name,model,datamodule,callbacks):
    
    trainer = Trainer(callbacks=callbacks, **config.trainer)
    model_weight_path = Path.cwd().parent / f"anomalib/results/{model_name}/boyteks/run/weights/lightning" / "model.ckpt"
    if model_weight_path is not None:
        load_model_callback = LoadModelCallback(weights_path=model_weight_path)#model checkpoint file
        trainer.callbacks.insert(0, load_model_callback)  
    else:
        start=time.time()
        trainer.fit(model=model, datamodule=datamodule)
        end=time.time()
        # print(f"training time:{end-start} ms")
        trainer.save_checkpoint(model_weight_path)
    # validation_result=trainer.test(model=model, datamodule=datamodule)
    # print('validation result :',validation_result)
    result_queue.put(trainer)


def openvino_inference_func(result_queue,model_name,model_path,device,trainer,config,images_path):

    pred_score=[]
    elapsed_time=[]
    pred_anomaly_map=[]
    pred_heatmap=[]
    pred_mask=[]
    
    if device=='cpu':
        output_path = Path(config["project"]["path"])
        openvino_model_path = output_path / "weights" / "openvino" / "model.bin"
        metadata = output_path / "weights" / "openvino" / "metadata.json"
        print(openvino_model_path.exists(), metadata.exists())
        
        inferencer = OpenVINOInferencer(
        path=openvino_model_path,  # Path to the OpenVINO IR model.
        metadata =metadata,  # Path to the metadata file.
        device='CPU') # default olarak "CPU" 'dur.
        for i in range(100):
            start=time.time()
            predictions = inferencer.predict(image=images_path[i])
            end=time.time()
            print(f'tahmin zamanı:{(end-start)*1000} ms')
            elapsed_time.append((end-start)*1000)
            pred_anomaly_map.append(predictions.anomaly_map)
            pred_heatmap.append(predictions.heat_map)
            pred_mask.append(predictions.pred_mask)
            pred_score.append(predictions.pred_score)
        mean_time=np.mean(elapsed_time)

    else:
        
        # print("okey mii")
        for i in range(100):
            inference_dataset = InferenceDataset(path=images_path[i],image_size=(256, 256))
            inference_dataloader = DataLoader(dataset=inference_dataset)
            start=time.time()
            predictions = trainer.predict(model=model_path, dataloaders=inference_dataloader)[0]                 
            end=time.time()
            print(f'tahmin zamanı:{(end-start)*1000} ms')
            elapsed_time.append((end-start)*1000)
            pred_anomaly_map.append(-1)
            pred_heatmap.append(-1)
            pred_mask.append(-1)
            pred_score.append(-1)
        mean_time=np.mean(elapsed_time)
    # print("burada sorun yok")
    result_queue.put((pred_score,mean_time,elapsed_time,pred_anomaly_map,pred_heatmap,pred_mask))




def TPR_and_TNR_calc(result_queue,n_ok,n_nok,threshold,tpr,fpr):
    # print("TPR TNR buraya girdi")
    accuracy= np.max(((n_ok*tpr)+(n_nok*(1-fpr)))/(n_ok+n_nok))
    max_accuracy_index = np.argmax(((n_ok*tpr)+(n_nok*(1-fpr)))/(n_ok+n_nok)) #max accuracy index
    selected_threshold = threshold[max_accuracy_index]#max accuracy değerine denk gelen index
    # Maksimum Accuracy değerine karşılık gelen TPR ve TNR değerleri
    selected_tpr = tpr[max_accuracy_index]
    selected_tnr = 1 - fpr[max_accuracy_index]
    result_queue.put((accuracy,selected_threshold,selected_tpr,selected_tnr))

if __name__ == '__main__':
    # train_main=trainMain()
    image_path_bad = "src/datasets/anomalib_boyteks_dataset_gray/test/bad/images/*"
    image_path_good = "src/datasets/anomalib_boyteks_dataset_gray/test/good/images/*"
    good_images_path=glob.glob(image_path_good)
    defected_images_path=glob.glob(image_path_bad)
    n_ok=len(good_images_path)
    n_nok=len(defected_images_path)
    # print(n_ok)
    normal_val=[]
    anomaly_val=[]
    for i in range(len(good_images_path)):
        normal_val.append(0)
        anomaly_val.append(1)
    toplam_val=normal_val+anomaly_val
    result_queue = Queue()
    result_queue2 = Queue()
    result_queue3 = Queue()
    result_queue4 = Queue()
    freeze_support() 

    current_directory = Path.cwd()
    model_name = [  'padim', 'cflow','ganomaly', 'dfkde','patchcore', 'cfa', 'dfm', 'efficient_ad', 'fastflow'] 
    for i in range(len(model_name)):
        CONFIG_PATH = f"src/anomalib/models/{model_name[i]}/custom_boyteks.yaml"
        with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
            print(f"{model_name}:",file.read())
        config = get_configurable_parameters(config_path=CONFIG_PATH)
        # config.optimization.export_mode = "openvino"
        model = get_model(config)
        datamodule = get_datamodule(config)
        callbacks = get_callbacks(config)
        Process(target=datamodule_activate_func(datamodule)).start()
        Process(target=train_and_val_func(result_queue,model_name[i],model,datamodule,callbacks)).start()
        trainer_result=result_queue.get()
        Process(target=openvino_inference_func(result_queue,model_name[i],model,'cpu',trainer_result,config,good_images_path)).start()
        prediction_good_score, mean_time_good,elapsed_time_good,pred_anomaly_map_good,pred_heatmap_good,pred_mask_good = result_queue2.get()
        # print("sorun yokk")
        Process(target=openvino_inference_func(result_queue2,model_name[i],model,'cpu',trainer_result,config,defected_images_path)).start()
        # print("sorun yokk")
        prediction_bad_score, mean_time_bad,elapsed_time_bad,pred_anomaly_map_bad,pred_heatmap_bad,pred_mask_bad = result_queue3.get()
        print("sorun yokk")
            
        mean_time=(mean_time_good+mean_time_bad)/2
        elapsed_time=elapsed_time_bad+elapsed_time_good
        prediction_all_score=prediction_good_score+prediction_bad_score

        fpr,tpr,threshold=roc_curve(toplam_val,prediction_all_score)
        auc_result = roc_auc_score(toplam_val, prediction_all_score)
        Process(target=TPR_and_TNR_calc(result_queue4,n_ok,n_nok,threshold,tpr,fpr)).start()
        accuracy_result,threshold_result,TPR,TNR=result_queue4.get()



        data_lists = [normal_val, anomaly_val, mean_time, elapsed_time, prediction_good_score, pred_anomaly_map_good, pred_heatmap_good,
                    pred_mask_good, prediction_bad_score, pred_anomaly_map_bad, pred_heatmap_bad, pred_mask_bad, auc_result, accuracy_result,
                    TPR, TNR]
        data_names = ["normal_val", "anomaly_val", "mean_time", "elapsed_time", "prediction_good_score", "pred_anomaly_map_good",
                    "pred_heatmap_good", "pred_mask_good", "prediction_bad_score", "pred_anomaly_map_bad", "pred_heatmap_bad",
                    "pred_mask_bad", "auc_result", "accuracy_result", "TPR", "TNR"]


        with h5py.File(f'test_results/{model_name[i]}_result_datas.h5', "w") as h5f:
            for data, name in zip(data_lists, data_names):
                h5f.create_dataset(name, data=np.asarray(data))



        plt.title(f'{model_name[i]} Boyteks Dataset Sonuç')
        plt.xlabel('Prediction Time (ms)')
        plt.ylabel('Accuracy')
        plt.scatter(mean_time, accuracy_result, label=f"{model_name[i]}", marker="^")
        plt.show()
                