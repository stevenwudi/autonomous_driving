from code_base.SSD_synthia_car_detection_fine_tune import calculate_iou


test_gt_file = '/home/public/synthia/ssd_car_fine_tune/ssd_car_test_gt.pkl'
test_json_file = '/home/public/synthia/ssd_car_test_faster.json'


calculate_iou(test_gt_file, test_json_file, POR=1e-3)