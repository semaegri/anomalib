import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning import Trainer
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.data import InferenceDataset, TaskType
from anomalib.models import get_model
from anomalib.pre_processing.transforms import Denormalize
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
from database.db_connection import *

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
    
    
def openvino_inference_func(result_queue,model_path,model_name,device,trainer,config,images_path):
    # pred_score=[]
    # elapsed_time=[]
    # pred_anomaly_map=[]
    # pred_heatmap=[]
    # pred_mask=[]
    if device=='cpu':
        output_path = Path(config["project"]["path"])
        openvino_model_path = output_path / "weights" / "openvino" / "model.bin"
        metadata = output_path / "weights" / "openvino" / "metadata.json"
        print(openvino_model_path.exists(), metadata.exists())
        
        inferencer = OpenVINOInferencer(
        path=openvino_model_path,  # Path to the OpenVINO IR model.
        metadata =metadata,  # Path to the metadata file.
        device='CPU') # default olarak "CPU" 'dur.
        for i in range(len(images_path)):
            image_name=images_path[i].split('/')[-1]
            images_status=images_path[i].split('/')[-2]
            if images_status=='good':
                label=0
            else:
                label=1
            start=time.time()
            predictions = inferencer.predict(image=images_path[i])
            end=time.time()
            prediction_time=round((end-start)*1000,2)
            # print(f'tahmin zamanı:{prediction_time} ms')
            # elapsed_time.append(prediction_time)
            # print('resim durum:',images_status)
            # print('tahmin skoru:',predictions.pred_score)
            if predictions.pred_score>0.50:
                ok_nok='anomaly'
            else:
                ok_nok='good'
            # pred_anomaly_map.append(predictions.anomaly_map)
            # pred_heatmap.append(predictions.heat_map)
            # pred_mask.append(predictions.pred_mask)
            # pred_score.append(predictions.pred_score)
            insert_data(images_path, image_name, model_name, prediction_time,label, predictions.pred_score, ok_nok)
        # mean_time=np.mean(elapsed_time)

    # else:
        
    #     for i in range(len(images_path)):
    #         image_name=images_path[i].split('/')[-1]
    #         label=images_path[i].split('/')[-2]
    #         inference_dataset = InferenceDataset(path=images_path[i],image_size=(256, 256))
    #         inference_dataloader = DataLoader(dataset=inference_dataset)
    #         start=time.time()
    #         predictions = trainer.predict(model=model_path, dataloaders=inference_dataloader)[0]                 
    #         end=time.time()
    #         print(f'tahmin zamanı:{round((end-start)*1000,2)} ms')
    #         prediction_time=round((end-start)*1000,2)
    #         elapsed_time.append(prediction_time)
    #         pred_anomaly_map.append(-1)
    #         pred_heatmap.append(-1)
    #         pred_mask.append(-1)
    #         pred_score.append(-1)
    #         insert_data(images_path, image_name, model_name, prediction_time,label, prediction_score, ok_nok)
    #     mean_time=np.mean(elapsed_time)
        
    # result_queue.put((pred_score,mean_time,elapsed_time,pred_anomaly_map,pred_heatmap,pred_mask))

def TPR_and_TNR_calc(result_queue2,n_ok,n_nok,threshold,tpr,fpr):
    # print("TPR TNR buraya girdi")
    accuracy= np.max(((n_ok*tpr)+(n_nok*(1-fpr)))/(n_ok+n_nok))
    max_accuracy_index = np.argmax(((n_ok*tpr)+(n_nok*(1-fpr)))/(n_ok+n_nok)) #max accuracy index
    selected_threshold = threshold[max_accuracy_index]#max accuracy değerine denk gelen index
    # Maksimum Accuracy değerine karşılık gelen TPR ve TNR değerleri
    selected_tpr = tpr[max_accuracy_index]
    selected_tnr = 1 - fpr[max_accuracy_index]
    result_queue2.put((accuracy,selected_threshold,selected_tpr,selected_tnr))


if __name__ == '__main__':
    
    test_directory="src/datasets/anomalib_boyteks_dataset_gray/test/" 
    TEST_DIR = {'good': f'{test_directory}good/images/*',
            'bad': f'{test_directory}bad/images/*'}
    test_good_path=(TEST_DIR['good'])
    good_images=glob.glob(test_good_path)
    bad_images=glob.glob(TEST_DIR['bad'])
    test_images_path=good_images+bad_images
    n_ok=len(good_images)
    n_nok=len(bad_images)
    
    # print(n_ok)
    
    normal_label=[]
    anomaly_label=[]
    for i in range(len(good_images)):
        normal_label.append(0)
        anomaly_label.append(1)
    
    toplam_label=normal_label+anomaly_label #roc için hazırlanmıştır
    
    result_queue = Queue()
    result_queue2 = Queue()
    result_queue3 = Queue()
    
    freeze_support() 
    model_name = "padim" 
    CONFIG_PATH = f"src/anomalib/models/{model_name}/custom_boyteks.yaml"
    with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
        print(f"{model_name}:",file.read())
    config = get_configurable_parameters(config_path=CONFIG_PATH)
    datamodule = get_datamodule(config)
    config.optimization.export_mode = "openvino"
    model = get_model(config)
    callbacks = get_callbacks(config)
    Process(target=datamodule_activate_func(datamodule)).start()
    Process(target=train_and_val_func(result_queue,model_name,model,datamodule,callbacks)).start()
    trainer_result=result_queue.get()
    Process(target=openvino_inference_func(result_queue,model,model_name,'cpu',trainer_result,config,test_images_path)).start()
    prediction_all_score, mean_time_good,elapsed_time_good,pred_anomaly_map_good,pred_heatmap_good,pred_mask_good = result_queue2.get()
    print("sorun yokk")

    # print(len(prediction_all_score))
    # print(len(toplam_val))

    # print("harikasin işte sonuçlar:",prediction_all_score)
    fpr,tpr,threshold=roc_curve(toplam_label,prediction_all_score)
    auc_result = roc_auc_score(toplam_label, prediction_all_score)
    Process(target=TPR_and_TNR_calc(result_queue3,n_ok,n_nok,threshold,tpr,fpr)).start()
    accuracy_result,threshold_result,TPR,TNR=result_queue3.get()


    # data_lists = [normal_val, anomaly_val, mean_time, elapsed_time, prediction_good_score, pred_anomaly_map_good, pred_heatmap_good,
    #             pred_mask_good, prediction_bad_score, pred_anomaly_map_bad, pred_heatmap_bad, pred_mask_bad, auc_result, accuracy_result,
    #             TPR, TNR]
    



    # with h5py.File(f'{MODEL}_result_datas.h5', "w") as h5f:
    #     for data, name in zip(data_lists, data_names):
    #         h5f.create_dataset(name, data=np.asarray(data))



    # plt.title(f'{MODEL} Boyteks Dataset Sonuç')
    # plt.xlabel('Prediction Time (ms)')
    # plt.ylabel('Accuracy')
    # plt.scatter(mean_time, accuracy_result, label=f"{MODEL}", marker="^")
    # plt.show()
    