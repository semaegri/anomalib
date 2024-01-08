import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
f1 = h5py.File("padim_result_datas.h5", 'r+')

image_bad_path='src/datasets/anomalib_boyteks_dataset_gray/test/bad/images/img_0.png'

data_name = ["normal_val", "anomaly_val", "mean_time", "elapsed_time", "prediction_good_score", "pred_anomaly_map_good",
            "pred_heatmap_good", "pred_mask_good", "prediction_bad_score", "pred_anomaly_map_bad", "pred_heatmap_bad",
            "pred_mask_bad", "auc_result", "accuracy_result", "TPR", "TNR"]

# def print_dataset(dataset):
#     data = dataset[()]  
#     print(data)

# for i in range(len(data_name)):
#     if data_name[i] in f1:
#         dataset = f1[data_name[i]]
#         if isinstance(dataset, h5py.Dataset):
#             print(f"Veri kümesi adı: {data_name[i]}")
#             print_dataset(dataset)
#         else:
#             print(f"{data_name[i]} bir veri kümesi değil.")

# f1.close()

pred_mask_bad = f1["pred_mask_bad"]
pred_anomaly_map_bad = f1["pred_anomaly_map_bad"]
pred_heatmap_bad = f1["pred_heatmap_bad"]


# # H5 veri kümesini bir NumPy dizisine dönüştürün
# # pred_mask_good_array = np.array(pred_mask_good)
# plt.subplot(141)
# plt.imshow(image_bad, cmap='gray')
# plt.title("Original Image", color='red', fontdict={'weight': 'bold'})

# plt.subplot(142)
# plt.imshow(pred_mask_bad[0], cmap='gray')
# plt.title("Prediction Mask Image", color='red', fontdict={'weight': 'bold'})

# plt.subplot(143)
# plt.imshow(pred_anomaly_map_bad[0], cmap='hot')
# plt.title("Prediction Anomaly Map", color='red', fontdict={'weight': 'bold'})

# plt.subplot(144)
# plt.imshow(pred_heatmap_bad[0], cmap='hot')
# plt.title("Prediction Heat Map", color='red', fontdict={'weight': 'bold'})


# plt.tight_layout()  # Subplot'ları düzgünce hizala

# plt.show()
# # H5 dosyasını kapatın
# f1.close()


#ağırlık merkezi, alan, çevre, en, boy değerleri

image_bad = cv2.imread(image_bad_path, cv2.IMREAD_GRAYSCALE)
threshold_value = 100
_, thresholded = cv2.threshold(image_bad, threshold_value, 255, cv2.THRESH_BINARY_INV)
plt.figure()
plt.title("eşik görüntüsü")
plt.imshow(thresholded,cmap='gray')
plt.show()

blobs_contours_list, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# Tüm blob'ları dolaşın ve özellikleri hesaplayın
for blob_contour in blobs_contours_list:
    # Blob'ların alanını hesaplayın
    blob_area = cv2.contourArea(blob_contour)

    # Blob'ların ağırlık merkezini hesaplayın
    moments_dict = cv2.moments(blob_contour)
    if moments_dict["m00"] != 0:
        cx = int(moments_dict["m10"] / moments_dict["m00"])
        cy = int(moments_dict["m01"] / moments_dict["m00"])
    else:
        cx, cy = 0, 0

    # Blob çevresini hesaplayın
    perimeter = cv2.arcLength(blob_contour, True)

    # Blob'un enini ve boyunu hesaplayın
    x, y, w, h = cv2.boundingRect(blob_contour)
    fig, ax = plt.subplots()
    ax.imshow(image_bad, cmap='gray')
    ax.set_title("Manuel Hata Tespiti")

    circle = plt.Circle((cx, cy), max(w, h) / 2 + 5, color='red', fill=False)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    plt.show()

    # Blob özelliklerini yazdırın
    print(f"Blob Alanı: {blob_area}")
    print(f"Blob Ağırlık Merkezi: ({cx}, {cy})")
    print(f"Blob Çevresi: {perimeter}")
    print(f"Blob Genişlik: {w}")
    print(f"Blob Yükseklik: {h}")
    print("-------------------------------------------------")

# blobs_contours_list, _ = cv2.findContours(pred_mask_bad[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # Eğer yalnızca bir blob varsa, onun konturunu alın
# if len(blobs_contours_list) == 1:
#     contour = blobs_contours_list[0]

#     # Blob'ların alanını hesaplayın
#     blob_area = cv2.contourArea(contour)

#     # Blob'ların ağırlık merkezini hesaplayın
#     moments_dict = cv2.moments(contour)
#     if moments_dict["m00"] != 0:
#         cx = int(moments_dict["m10"] / moments_dict["m00"])
#         cy = int(moments_dict["m01"] / moments_dict["m00"])
#     else:
#         cx, cy = 0, 0
#     perimeter = cv2.arcLength(contour, True)
#     x, y, w, h = cv2.boundingRect(contour)
#     # Blob'ları görüntü üzerine çizin
#     image_copy = image_bad.copy()
#     cv2.drawContours(image_copy, [contour], -1, (0, 0, 255), 2)

#     # Blob özelliklerini yazdırın
#     print(f"Blob Alanı: {blob_area}")
#     print(f"Blob Ağırlık Merkezi: ({cx}, {cy})")
#     print(f"Blob Çevresi: {perimeter}")
#     print(f"Blob Genişlik: {w}")
#     print(f"Blob Yükseklik: {h}")
#     # Görüntüyü görselleştirin
#     plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()
# else:
#     print("Yalnızca bir blob olduğundan emin olun.")







