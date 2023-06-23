import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集和测试集中
    split_rate = [0.8, 0.1, 0.1]

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "processed")
    # data_root = cwd
    # origin_empty_path = os.path.join(data_root, "empty")
    # origin_occupied_path = os.path.join(data_root, "occupied")
    # assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    # empty_class = [cla for cla in os.listdir(origin_empty_path) if os.path.isdir(os.path.join(origin_empty_path, cla))]
    # occupied_class = [cla for cla in os.listdir(origin_occupied_path) if os.path.isdir(os.path.join(origin_occupied_path, cla))]
    all_class = [cla for cla in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, cla))]
    
    train_v_t_root = cwd

    # 建立保存训练集的文件夹
    train_root = os.path.join(train_v_t_root, "train")
    mk_file(train_root)
    for cla in all_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(train_v_t_root, "val")
    mk_file(val_root)
    for cla in all_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 建立保存测试集的文件夹
    test_root = os.path.join(train_v_t_root, "test")
    mk_file(test_root)
    for cla in all_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(test_root, cla))

    for cla in all_class:
        cla_path = os.path.join(data_root, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        evaltest_index = random.sample(images, k=int(num*(1.0 - split_rate[0])))
        test_index = random.sample(evaltest_index, k=int(num*(split_rate[-1])))
        for index, image in enumerate(images):
            if image in test_index:
                # 将分配至测试集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(test_root, cla)
                copy(image_path, new_path)
            elif image in evaltest_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()