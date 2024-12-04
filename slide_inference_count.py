import json
import fiona
import rasterio
import os
import random
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from rasterio import CRS
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import init_default_scope
try:  
    from sahi.slicing import slice_image
except ImportError:
    raise ImportError('Please run "pip install -U sahi" '
                      'to install sahi first for large image inference.')

from mmdet.registry import VISUALIZERS
from mmdet.utils.large_image_center import merge_results_by_nms
from mmdet.utils.misc import get_file_list
from rich import console, print, table
from rich.progress import track
console = console.Console()
table1 = table.Table(title="[red]prediction results: class and bbox num analysis[/red]", show_header=True, header_style="bold magenta")
table2 = table.Table(title="[red]prediction results: class and bbox score analysis[/red]", show_header=True, header_style="bold magenta")

init_default_scope('mmdet')

def parse_args():
    parser = ArgumentParser(
        description='Perform MMDET inference on large images.')
    parser.add_argument( # /data1/DATA115-1/huangqj/PIEAI/to_test/14OCT09032309-M2AS-053934572090_02_P003_truecolor.tif
        '--img', default='test_images/12096.jpg', help='Image path, include image file, dir and URL.')
    parser.add_argument('--config', default='/data1/huangqj/mmlab/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_dior.py', help='Config file')
    parser.add_argument('--checkpoint', default='/data1/huangqj/mmlab/mmdetection/work_dirs/grounding_dino_swin-t_finetune_16xb2_1x_dior/epoch_12.pth', help='Checkpoint file')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument(
        '--out-dir', default='outputs', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument(
        '--patch-size', type=int, default=800, help='The size of patches')
    parser.add_argument(
        '--patch-overlap-ratio',
        type=float,
        default=0.25,
        help='Ratio of overlap between two patches')
    parser.add_argument(
        '--merge-iou-thr',
        type=float,
        default=0.25,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--merge-nms-type',
        type=str,
        default='nms',
        help='NMS type for merging results')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size, must greater than or equal to 1')
    parser.add_argument(
        '--save-patch',
        action='store_true',
        help='Save the results of each patch. '
        'The `--debug` must be enabled.')
    parser.add_argument(
        '--save_txt_results',
        action='store_true',
        help='Save the results of whole image.')
    args = parser.parse_args()
    return args

def WriteVectorFile(file_name, results_bboxes, results_scores, results_labels, crs, transform_tif, name_list):
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'class': 'str',
            'confidence': 'float',
        }
    }

    with fiona.open(file_name, mode='w', driver='ESRI Shapefile', schema=schema, crs=crs, encoding='utf-8') as layer:
        for i, detection in enumerate(results_bboxes):
            xmin, ymin, xmax, ymax = detection[0:4]
            detect_bbox = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
            
            coords = np.array(detect_bbox).reshape(-1, 2)
            coords = [transform_tif * c for c in coords]
            ret = {'type':'Polygon', 'coordinates':[coords]}
            
            confidence = results_scores[i]
            class_name = name_list[int(results_labels[i])]

            detection_feature = {
                'geometry': ret,
                'properties': {
                    'class': class_name,
                    'confidence': confidence,
                },
            }
            layer.write(detection_feature)


def main():
    
    args = parse_args()

    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
                                      " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
                                         "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        test_data_cfg.pipeline = config.tta_pipeline

    # TODO: TTA mode will error if cfg_options is not set.
    #  This is an mmdet issue and needs to be fixed later.
    # build the model from a config file and a checkpoint file
    
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    # using class name in model dataset meta
    name_list = list(model.dataset_meta['classes'])
    if args.texts:
        text_prompt = args.texts
    else:
        text_prompt = model.dataset_meta['classes']

    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    # using rich.console to p
    print(f'[red]Performing inference on {len(files)} images.... '
          'This may take a while.[\red]')
    progress_bar = ProgressBar(len(files))
    for file in files:
        # indicate shapefile name and where to save
        image_name = os.path.basename(file).rsplit('.', 1)[0]
        shp_name = image_name + '.shp'
        shp_path = os.path.join(args.out_dir, shp_name)
        # read the projection and transformation of the image
        with rasterio.open(file, 'r+') as src:
            img = src.read() # read format is numpy array
            img = img.transpose(1, 2, 0) # hwc -> chw
            # crs = src.crs.to_proj4() # get image projection matrix. src.crs is get projection coordinate system .to_proj4() is get projection matrix
            if src.crs is None:
                # 如果不存在，手动定义一个默认的坐标参考系统，这里使用EPSG 4326作为示例
                default_crs = CRS.from_epsg(4326)
                # 将手动创建的坐标参考系统赋给数据源的crs属性
                src.crs = default_crs
                crs_proj4 = src.crs.to_proj4()
            else:
                crs_proj4 = src.crs.to_proj4() # crs = src.crs.to_proj4() 返回坐标参考系统的Proj4字符串表示，而 crs = src.crs 返回坐标参考系统对象本身。如果影像没有坐标参考系统，那么调用src.crs.to_proj4()会报错，而调用src.crs则会返回None。
            transform_tif = src.transform # # get image transformation matrix

        # arrange slices
        height, width = img.shape[:2]
        sliced_image_object = slice_image(
            img,
            slice_height=args.patch_size,
            slice_width=args.patch_size,
            auto_slice_resolution=False,
            overlap_height_ratio=args.patch_overlap_ratio,
            overlap_width_ratio=args.patch_overlap_ratio,
        )
        # perform sliced inference
        slice_results = []
        start = 0
        # get tqdm length
        
        track_length = len(sliced_image_object) // args.batch_size + 1
        # while True:
        for _ in track(range(track_length), description='Sliced inferencing........'):
            slice_inference_results = None
            # prepare batch slices
            end = min(start + args.batch_size, len(sliced_image_object))
            images = []
            for sliced_image in sliced_image_object.images[start:end]:
                images.append(sliced_image)
            slice_inference_results = inference_detector(model, images, text_prompt=text_prompt)
            #TODO: To avoid OOM error, apply nms to each slice, later do this!
            # using score threshold to filter results first now!
            #TODO: Using multiprocess implement, but maybe not work on windows
            for slice_result in slice_inference_results:
                # iterate slice result
                # filter results by score threshold, if score < args.score_thr, result=[]
                slice_result.pred_instances = slice_result.pred_instances[
                    slice_result.pred_instances.scores > args.score_thr]
                
                if len(slice_result.pred_instances) > 0:
                    slice_results.append(slice_result)
        
            # slice_results.extend(slice_inference_results)

            if end >= len(sliced_image_object):
                break
            start += args.batch_size

       
        image_result = merge_results_by_nms(
            slice_results,
            sliced_image_object.starting_pixels,
            src_image_shape=(height, width),
            nms_cfg={
                'type': args.merge_nms_type,
                'iou_threshold': args.merge_iou_thr
            })
        # get pred results------ DetDataSample
        # filter results by score threshold
        # image_result.pred_instances = image_result.pred_instances[
        #             image_result.pred_instances.scores > args.score_thr]
        results = image_result.pred_instances
        results_bboxes, results_scores, results_labels, results_img_id = results.bboxes.data.cpu().numpy().tolist(), results.scores.data.cpu().numpy().tolist(), results.labels.data.cpu().numpy().tolist(), image_result.img_id
        # print(f'Image {str(file.name)} has {len(results_bboxes)} bbox predictions, the score threshold is {args.score_thr} now!')
        # write shapefile
        WriteVectorFile(shp_path, results_bboxes, results_scores, results_labels, crs_proj4, transform_tif, name_list)
        progress_bar.update()
        
        # analyse detection results, and write result to txt
        if len(results_bboxes) > 0:
            class_num_dic = {}
            class_score_dic = {}
            all_bbox_info = []
            # with open(os.path.join(args.out_dir, filename + '.txt'), 'w') as f:
            # iterate results
            for i in range(len(results_bboxes)):
                xmin, ymin, xmax, ymax = results_bboxes[i][0:4]
                confidence = results_scores[i]
                # detr class number start from 1, so need to -1， clas 0 is background
                class_name = name_list[int(results_labels[i]) - 1]
                # for class number analysis
                if class_name not in class_num_dic.keys():
                    class_num_dic[class_name] = 1
                else:
                    class_num_dic[class_name] += 1
                # for score analysis
                if class_name not in class_score_dic.keys():
                    class_score_dic[class_name] = [confidence]
                else:
                    class_score_dic[class_name].append(confidence)
                # for all bbox info
                all_bbox_info.append([xmin, ymin, xmax, ymax, class_name, confidence])
        
            # analyse class, bbox number using table in rich
            table_data = []
            table_data.append(['class', 
                            'bbox number'])
            table1.add_column('class', justify='center', style='cyan', no_wrap=True)
            table1.add_column('bbox number', justify='center', style='magenta')
            for key in class_num_dic.keys():
                table1.add_row(key, 
                            str(class_num_dic[key]))
            # add total bbox number
            table1.add_row('total bbox number',
                        str(len(results_bboxes)))
            # console.print(table1)
            
            # analyse class, bbox score using table in rich, using arange(args.threshold, 1, 0.1) as score range
            table_data = []
            table_data.append(['class', 'score range', 'bbox number'])
            table2.add_column('class', justify='center', style='cyan', no_wrap=True)
            table2.add_column('score range', justify='center', style='magenta')
            table2.add_column('bbox number', justify='center', style='magenta')
            
            class_score_all = []
            for key in class_score_dic.keys():
                score_range_str = ''
                bbox_number_str = ''
                for i in np.arange(args.score_thr, 1, 0.1):
                    # 在第一列中对应的一行添加多行数据
                    # score_range.append(len([score for score in class_score_dic[key] if score >= i]))
                    key = key
                    score_range_str += str(round(i, 2)) + '~' + str(round(i + 0.1, 2)) + '\n'
                    bbox_number_str += str(len([score for score in class_score_dic[key] if (score >= i and score < i + 0.1)])) + '\n'
                    # table2.add_row(key, str(round(i, 2)), str(len([score for score in class_score_dic[key] if (score >= i and score < i + 0.1)])))
                table2.add_row(key, score_range_str, bbox_number_str)
                # add all bbox number
                class_score_all.extend(class_score_dic[key])
            # analyse all bbox score
            score_range_str_all = ''
            bbox_number_str_all = ''
            for i in np.arange(args.score_thr, 1, 0.1):
                score_range_str_all += str(round(i, 2)) + '~' + str(round(i + 0.1, 2)) + '\n'
                bbox_number_str_all += str(len([score for score in class_score_all if (score >= i and score < i + 0.1)])) + '\n'
            table2.add_row('All Classes', score_range_str_all, bbox_number_str_all)
            
            console.print(table1, table2, justify='center')
            
            # 在同一行显示多个表格
            
            # if args.save_txt_results and len(all_bbox_info) > 0:
            #     for bbox in all_bbox_info:
            #         xmin, ymin, xmax, ymax, class_name, confidence = bbox
            #         f.write(str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ','  + class_name + ',' + str(confidence) +'\n')
            
        # vis bbox on image
        # visualizer.add_datasample(
        #     filename,
        #     img,
        #     data_sample=image_result,
        #     draw_gt=False,
        #     show=args.show,
        #     wait_time=0,
        #     out_file=out_file,
        #     pred_score_thr=args.score_thr,
        # )

    if not args.show or (args.debug and args.save_patch):
        print(f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
