import os
import sys
import numpy as np
import cv2
import utils
import csv

trainset_dir = sys.argv[1]
pos_dir = os.path.join(trainset_dir, 'pos')
neg_dir = os.path.join(trainset_dir, 'neg')
output_dir = os.path.join(sys.argv[2], "multipleBackgrounds")
if (not os.path.isdir(output_dir)):
    os.makedirs(output_dir)

total_samples = int(sys.argv[3])
nums_dir = len(os.listdir(pos_dir)) + len(os.listdir(neg_dir))
random_crop_upper = int(total_samples / (4*(nums_dir/2))) + 1
error_samples_path = 'error_samples.log'
error_samples_file = open(error_samples_path, 'a')

with open(os.path.join(output_dir,"gt.csv"), 'w') as gt_csv:
#    spamwriter_1 = csv.writer(csvfile, delimiter=',',
#                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for image in os.listdir(pos_dir):
        if image.endswith("jpg") or image.endswith("JPG"):
            if os.path.isfile(os.path.join(pos_dir,image+".csv")):
                with open(os.path.join(pos_dir,image+ ".csv"), 'r') as csvfile:
                    spamwriter = csv.reader(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    img = cv2.imread(os.path.join(pos_dir, image))
                    print(image)
                    gt= []
                    for row in spamwriter:
                        if len(row) <= 0:
                          continue
                        gt.append(row)

                    gt = np.array(gt).astype(np.float32)
                    gt = gt / (img.shape[1], img.shape[0])
                    gt = gt * (1080, 1080)
                    img = cv2.resize(img, (1080, 1080))


                    # print gt
                    error_image_name = None
                    for angle in range(0,271,90):
                        img_rotate, gt_rotate = utils.rotate(img, gt,angle)
                        for random_crop in range(0,random_crop_upper):
                            try:
                                img_crop, gt_crop = utils.random_crop(img_rotate, gt_rotate)
                                mah_size = img_crop.shape
                                img_crop = cv2.resize(img_crop, (300, 300))
                                gt_crop = np.array(gt_crop)

                                gt_crop = gt_crop*(300.0 / mah_size[1],300.0 / mah_size[0])

                                # for a in range(0,4):
                                no=0
                                # for a in range(0,4):
                                #     no+=1
                                #     img_crop = cv2.circle(img_crop, tuple((gt_crop[a].astype(int))), 2,(255-no*60,no*60,0),9)
                                # cv2.imwrite("asda.jpg", img)
                                # 0/0
                                cv2.imwrite(output_dir + "/" +str(angle)+str(random_crop)+ image, img_crop)
#                                spamwriter_1.writerow((str(angle)+str(random_crop)+ image, gt_crop))
                                gt_crop_list = [str(i) for i in gt_crop.reshape(-1).tolist()]
                                gt_csv.write('{} {} 300 0\n'.format(str(angle)+str(random_crop)+ image, ' '.join(gt_crop_list))) 

                                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                # cl1 = clahe.apply(img_crop)
                                #
                                # cv2.imwrite(output_dir + "/" + dir + "cli" + image, img_crop)
                                # spamwriter_1.writerow((dir + "cli" + image, gt_crop))
                                # for row_counter in range(0,4):
                                #     row=gt_crop[row_counter]
                                #     print row
                                #     img_crop = cv2.circle(img_crop, (int(float(row[0])), int(float(row[1]))), 2, (255, 0, 0), 2)

                                # img_temp = cv2.resize(img_temp, (300, 300))
                            except:
                                error_image_name = image
                    if not error_image_name is None:
                        error_samples_file.write('{}\n'.format(error_image_name))


    for image in os.listdir(neg_dir):
        if image.endswith("jpg") or image.endswith("JPG"):
            img = cv2.imread(os.path.join(neg_dir, image))
            print(image)
            gt = [[img.shape[1]*0.25, img.shape[0]*0.25],
                  [img.shape[1]*0.75, img.shape[0]*0.25],
                  [img.shape[1]*0.75, img.shape[0]*0.75],
                  [img.shape[1]*0.25, img.shape[0]*0.75]]
            gt = np.array(gt).astype(np.float32)
            gt = gt / (img.shape[1], img.shape[0])
            gt = gt * (1080, 1080)
            img = cv2.resize(img, (1080, 1080))

            # print gt
            error_image_name = None
            for angle in range(0,271,90):
                img_rotate, gt_rotate = utils.rotate(img, gt,angle)
                for random_crop in range(0,random_crop_upper):
                    try:
                        img_crop, gt_crop = utils.random_crop(img_rotate, gt_rotate)
                        img_crop = cv2.resize(img_crop, (300, 300))
                        gt_crop = [[0,0],
                                   [300, 0],
                                   [300,300],
                                   [0, 300]]
                        gt_crop = np.array(gt_crop)

                        no=0

                        cv2.imwrite(output_dir + "/" +str(angle)+str(random_crop)+ image, img_crop)
#                        spamwriter_1.writerow((str(angle)+str(random_crop)+ image, gt_crop))
                        gt_crop_list = [str(i) for i in gt_crop.reshape(-1).tolist()]
                        gt_csv.write('{} {} 0 300\n'.format(str(angle)+str(random_crop)+ image, ' '.join(gt_crop_list))) 


                    except:
                        error_image_name = image

                if not error_image_name is None:
                    error_samples_file.write('{}\n'.format(error_image_name))
