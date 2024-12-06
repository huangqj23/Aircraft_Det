from dataset_tools.hbb.hbb_dataset_converter import HBBDatasetConverter
from dataset_tools.hbb.hbb_dataset_visualizer import HBBDatasetVisualizer
from pathlib import Path


if __name__ == '__main__':

  mar20_aircraft_mapping = {
      'A1': 'SU-35',
      'A2': 'C-130',
      'A3': 'C-17',
      'A4': 'C-5',
      'A5': 'F-16',
      'A6': 'TU160',
      'A7': 'E-3',
      'A8': 'B-52',
      'A9': 'P-3C',
      'A10': 'B-1B',
      'A11': 'E-8',
      'A12': 'TU-22',
      'A13': 'F-15',
      'A14': 'KC-135',
      'A15': 'F-22',
      'A16': 'FA-18',
      'A17': 'TU-95',
      'A18': 'KC-10',
      'A19': 'SU-34',
      'A20': 'SU-24'
  }
  
    # vis
    # visualizer = HBBDatasetVisualizer()
    
    # # 设置类别名称（对YOLO格式必需，其他格式可选）
    # class_names = list(mar20_aircraft_mapping.values())
    # # # visualizer.set_class_names(class_names)

    # coco_path = Path('/data1/DATA_126/hqj/MAR20/val/coco/coco.json')
    # visualizer = HBBDatasetVisualizer()
    # visualizer.set_class_names(class_names)
    # visualizer.visualize(
    #     image_path='/data1/DATA_126/hqj/MAR20/val/images/',
    #     label_path=coco_path,
    #     format='coco',
    #     output_dir='/data1/DATA_126/hqj/MAR20/val/visualize'
    # )

  categories = [
  {
    "id": 1,
    "name": "A1",
    "supercategory": "none"
  },
  {
    "id": 2,
    "name": "A10",
    "supercategory": "none"
  },
  {
    "id": 3,
    "name": "A11",
    "supercategory": "none"
  },
  {
    "id": 4,
    "name": "A12",
    "supercategory": "none"
  },
  {
    "id": 5,
    "name": "A13",
    "supercategory": "none"
  },
  {
    "id": 6,
    "name": "A14",
    "supercategory": "none"
  },
  {
    "id": 7,
    "name": "A15",
    "supercategory": "none"
  },
  {
    "id": 8,
    "name": "A16",
    "supercategory": "none"
  },
  {
    "id": 9,
    "name": "A17",
    "supercategory": "none"
  },
  {
    "id": 10,
    "name": "A18",
    "supercategory": "none"
  },
  {
    "id": 11,
    "name": "A19",
    "supercategory": "none"
  },
  {
    "id": 12,
    "name": "A2",
    "supercategory": "none"
  },
  {
    "id": 13,
    "name": "A20",
    "supercategory": "none"
  },
  {
    "id": 14,
    "name": "A3",
    "supercategory": "none"
  },
  {
    "id": 15,
    "name": "A4",
    "supercategory": "none"
  },
  {
    "id": 16,
    "name": "A5",
    "supercategory": "none"
  },
  {
    "id": 17,
    "name": "A6",
    "supercategory": "none"
  },
  {
    "id": 18,
    "name": "A7",
    "supercategory": "none"
  },
  {
    "id": 19,
    "name": "A8",
    "supercategory": "none"
  },
  {
    "id": 20,
    "name": "A9",
    "supercategory": "none"
  }
  ]
  # 将类别信息安装id装到列表中
  # 
  category_list = [c['name'] for c in categories]
  print(category_list)