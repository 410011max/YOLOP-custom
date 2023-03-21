import numpy as np
import json

from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict_Aidea
from tqdm import tqdm


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None, data_percentage = 1.0):
        super().__init__(cfg, is_train, inputsize, transform)
        self.cfg = cfg
        self.data_percentage = data_percentage
        self.db = self._get_db()
        
    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        data_list = list(self.mask_list)
        data_list = data_list[:int(len(data_list)*(self.data_percentage))]
        if self.data_percentage <= 0.0:
            raise ValueError(f"Data percentage should be greater than 0, current value:{self.data_percentage}")
        
        for mask in tqdm(data_list):
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                # if category == "traffic light":
                #     color = obj['attributes']['trafficLightColor']
                #     category = "tl_" + color
                if category in id_dict_Aidea.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = id_dict_Aidea[category]

                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)
                
            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if obj['category'] in id_dict_Aidea.keys():
                    remain.append(obj)
                        
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
