from code_base.SSD_synthia_car_detection_fine_tune import calculate_iou, test_ssd512, gt_classification_convert
import pickle
import json
test_gt_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_test_gt-shuffle.pkl'
test_json_file = '/home/public/synthia/ssd_car_test_faster-shuffle.json'


calculate_iou(test_gt_file, test_json_file, POR=2e-3, draw=False)
# test_gt_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_test_gt-shuffle.pkl'
# test_json_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_test-shuffle.json'
# gt = pickle.load(open(test_gt_file, 'rb'))
# gt = gt_classification_convert(gt)
# keys = sorted(gt.keys())
# with open(test_json_file, 'r') as fp:
#     predict_dict = json.load(fp)
# a = sorted(predict_dict.keys())
#
# for key in a:
#     print (key)