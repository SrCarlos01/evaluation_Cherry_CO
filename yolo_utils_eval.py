
import os
import numpy as np

def scale_boxes(array_boxes:np.ndarray, array_scale:np.ndarray):
    # array_boxes = array_boxes.reshape([-1,4])
    return array_scale*array_boxes


def yolo2coco_box(box):
    box[:2] -= box[2:]/2
    # box[:,2:] += box[:,:2]
    return [float(val) for val in box]

def reformat_yolo2coco(bounding_box,array_scale):

    bboxes = scale_boxes(bounding_box,array_scale)
    bboxes[:,:2] -= bboxes[:,2:] / 2
    bboxes[:,2:] += bboxes[:,:2]

    return bboxes
def create_annotation_from_yolo(file_name, id, bboxes, scores, classes):
    anns = []
    for i,box in enumerate(bboxes):
        ann = {'file_name': file_name}
        ann['bbox'] = yolo2coco_box(box)
        ann['image_id'] = id
        ann['category_id'] = int(classes[i])+1
        ann['score'] = scores[i]
        anns += [ann]
    return anns

def load_bbox_file_dicts(imgs:dict, path_bbox_pred:str):
    # with open(path_bbox_pred) as f:
    #     lines = f.readlines()
    # data = []
    # for l in lines:
    #     data += [json.loads(l)]
    # return data
    files = [name for name in sorted(os.listdir(path_bbox_pred))]
    predictions = []
    # predictions = [[] for _ in range(len(imgs))]
    id_images = {}
    for i, (key,img_info) in enumerate(imgs.items()):
        id_images[img_info['file_name']] = i

    for i,file in enumerate(files):
        info_file = np.loadtxt(os.path.join(path_bbox_pred,file))
        info_file = info_file.reshape([-1,6])
        classes_file = info_file[:,0]
        file_name = file.replace('.txt', '.JPG')
        id_img = id_images[file_name]
        height = imgs[id_img]['height']
        width = imgs[id_img]['width']
        array_scale = np.array([width,height,width,height])
        bboxes = scale_boxes(info_file[:,1:5],array_scale)
        scores = info_file[:,5]

        anns = create_annotation_from_yolo(file_name, id_img, bboxes, scores,classes_file)
        predictions += anns
    return predictions


def load_bbox_file_arrays(imgs:dict, path_bbox_pred:str,n_classes:int):
    # with open(path_bbox_pred) as f:
    #     lines = f.readlines()
    # data = []
    # for l in lines:
    #     data += [json.loads(l)]
    # return data
    files = [name for name in sorted(os.listdir(path_bbox_pred))]
    # predictions = []
    predictions = [[] for _ in range(len(imgs))]
    id_images = {}
    for i, (key,img_info) in enumerate(imgs.items()):
        id_images[img_info['file_name']] = i


    for i,file in enumerate(files):
        info_file = np.loadtxt(os.path.join(path_bbox_pred,file))
        info_file = info_file.reshape([-1,6])
        classes_file = info_file[:,0]
        file_name = file.replace('.txt', '.JPG')
        id_img = id_images[file_name]
        height = imgs[id_img]['height']
        width = imgs[id_img]['width']

        array_scale = np.array([width,height,width,height])
        bboxes = reformat_yolo2coco(info_file[:,1:5],array_scale) # x1y1x2y2
        scores = info_file[:,5]
        arrays_classes = []
        for j in range(n_classes):
            n_items = np.sum(classes_file==j)
            array_class = np.empty([n_items,5])
            array_class[:,:4] = bboxes[classes_file==j]
            array_class[:,4] = scores[classes_file==j]
            arrays_classes += [array_class]



        # id = id_images[file_name]
        # anns = create_annotation_from_yolo(file_name, id, bboxes, scores,classes_file)
        predictions[id_img] = arrays_classes
    return predictions