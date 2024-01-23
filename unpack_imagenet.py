import json, os, shutil


def unpack(base_dir, 
           target_dir, 
           train='ILSVRC2012_img_train.tar', 
           val='ILSVRC2012_img_val.tar',
           class_json='ImageNet_class_index.json',
           val_label='ImageNet_val_label.txt'):
    
    # path
    train_dir = os.path.join(base_dir, train)
    val_dir   = os.path.join(base_dir, val)
    json_dir  = os.path.join(base_dir, class_json)
    txt_dir   = os.path.join(base_dir, val_label)

    target_train_dir = os.path.join(target_dir, 'train')
    target_val_dir   = os.path.join(target_dir, 'val')

    # dictionary for class to num
    class2num = {}
    with open(json_dir) as json_file:
        json_data = json.load(json_file)
        for num in json_data:
            class2num[json_data[num][0]] = num
    
    # unzip train dataset
    shutil.unpack_archive(train_dir, target_train_dir)
    for class_zip in sorted(os.listdir(target_train_dir)):
        class_, _ = class_zip.split('.')
        shutil.unpack_archive(os.path.join(target_train_dir, class_zip), 
                              os.path.join(target_train_dir, class2num[class_]))
        os.remove(os.path.join(target_train_dir, class_zip))
        
    # unzip val dataset
    shutil.unpack_archive(val_dir, target_val_dir)
    with open(txt_dir, 'r') as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            val_img, class_ = line.split()
            if not os.path.exists(os.path.join(target_val_dir, class2num[class_])):
                os.mkdir(os.path.join(target_val_dir, class2num[class_]))
            
            shutil.move(os.path.join(target_val_dir, val_img),
                        os.path.join(target_val_dir, class2num[class_]))

def check(ImageNet_dir):
    train_dir = os.path.join(ImageNet_dir, 'train')
    val_dir   = os.path.join(ImageNet_dir, 'val')
    
    train_cnt, val_cnt = 0, 0

    for c in os.listdir(train_dir):
        c_dir = os.path.join(train_dir, c)
        train_cnt += len(os.listdir(c_dir))

    for c in os.listdir(val_dir):
        c_dir = os.path.join(val_dir, c)
        val_cnt += len(os.listdir(c_dir))

    ImageNet_train, ImageNet_val = 1281167, 50000

    print('Train Images from ImageNet : {}'.format(ImageNet_train))
    print('Train Images Detected : {}'.format(train_cnt))
    print('Same : {}'.format(ImageNet_train == train_cnt))
    print()

    print('Val Images from ImageNet : {}'.format(ImageNet_val))
    print('Val Images Detected : {}'.format(val_cnt))
    print('Same : {}'.format(ImageNet_val == val_cnt))


if __name__ == '__main__':
    base_dir   = 'C:/Users/onlyb/Documents/연구/simsiam/raw_data'
    target_dir = 'C:/Users/onlyb/Documents/연구/simsiam/data'
    unpack(base_dir, target_dir)

    check(target_dir)