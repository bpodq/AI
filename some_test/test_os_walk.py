import os


# def test_walk():
#     path = 'D:/Python37/Lib/site-packages/cv2/data/'
#
#     imgpath = '../image/'
#     for root, dirs, files in os.walk(imgpath):  # 递归遍历所有文件
#         print(root, dirs, files)
#         # for file in files:
#         #     print(file)


# def test_listdir():
#     path = '../image/'
#     for file in os.listdir(path):
#         file_path = os.path.join(path, file)
#         if os.path.isdir(file_path):
#             # listdir(file_path, list_name)
#             print()
#         else:
#             # list_name.append(file_path)
#             print(file)


def test_listdir2():
    path = '../image/'
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path):
            print(file)
