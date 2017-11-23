import numpy as np
import scipy.ndimage
import scipy.misc
import sys
import pickle
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------

def find_center(n, br, rr):

    A_nd1 = [i for i in range(rr, n - rr, 2 * br + 1)]

    return(A_nd1)

#----------------------------------------------------------------------------------------------------------------------

def data_one(img_d2, seg_d2, br, rr):

    y_d1 = find_center(n = img_d2.shape[0], br = br, rr = rr)
    x_d1 = find_center(n = img_d2.shape[1], br = br, rr = rr)

    rl = 2 * rr + 1

    n = len(x_d1) * len(y_d1)

    img_0_d3 = np.zeros(shape = (n, rl, rl))
    img_1_d3 = np.zeros(shape = (n, rl, rl))

    img_0_n = 0
    img_1_n = 0

    for y in y_d1:

        for x in x_d1:

            status = np.mean(seg_d2[(y - br):(y + br + 1),(x - br):(x + br + 1)]) >= 0.5

            if(status):

                img_1_d3[img_1_n,] = img_d2[(y - rr):(y + rr + 1),(x - rr):(x + rr + 1)]
                img_1_n += 1

            else:

                img_0_d3[img_0_n,] = img_d2[(y - rr):(y + rr + 1),(x - rr):(x + rr + 1)]
                img_0_n += 1

    img_0_d3 = img_0_d3[0:img_0_n,]
    img_1_d3 = img_1_d3[0:img_1_n,]

    return(img_0_d3, img_1_d3)

#----------------------------------------------------------------------------------------------------------------------

def data_all(img_fld_path, seg_fld_path, fvc, db, f_d1, e_d1, br, rr, x4):

    rl = 2 * rr + 1

    all_f0_d3 = np.zeros(shape = (0, rl, rl))
    all_f1_d3 = np.zeros(shape = (0, rl, rl))

    for f in f_d1:

        for e in e_d1:

            print("create - fvc " + str(fvc) + " db " + str(db) + " f " + str(f) + " e " + str(e))

            path = "fvc/" + str(fvc) + "/db" + str(db) + "_b/" + str(f) + "_" + str(e)

            img_d2 = scipy.misc.imread(img_fld_path + path + ".tif").astype("float") / 255
            seg_d2 = scipy.misc.imread(seg_fld_path + path + ".tif").astype("float") / 255

            one_f0_d3, one_f1_d3 = data_one(img_d2 = img_d2, seg_d2 = seg_d2, br = br, rr = rr)

            all_f0_d3 = np.append(all_f0_d3, one_f0_d3, axis = 0)
            all_f1_d3 = np.append(all_f1_d3, one_f1_d3, axis = 0)

            print("one_f0_d3: " + str(one_f0_d3.shape))
            print("one_f1_d3: " + str(one_f1_d3.shape))

            print("all_f0_d3: " + str(all_f0_d3.shape))
            print("all_f1_d3: " + str(all_f1_d3.shape))

    if(x4 != 0):

        all4_f0_d3 = np.zeros(shape = (4 * all_f0_d3.shape[0], rl, rl))
        all4_f1_d3 = np.zeros(shape = (4 * all_f1_d3.shape[0], rl, rl))

        for i in range(4):

            all4_f0_d3[(i * all_f0_d3.shape[0]):((i + 1) * all_f0_d3.shape[0])] = np.rot90(all_f0_d3,i, axes=(1,2))
            all4_f1_d3[(i * all_f1_d3.shape[0]):((i + 1) * all_f1_d3.shape[0])] = np.rot90(all_f1_d3,i, axes=(1,2))

        all_f0_d3 = all4_f0_d3
        all_f1_d3 = all4_f1_d3

    return(all_f0_d3, all_f1_d3)

#----------------------------------------------------------------------------------------------------------------------

img_fld_path = str(sys.argv[1])
seg_fld_path = str(sys.argv[2])
exm_pkl_path = str(sys.argv[3])

fvc = int(sys.argv[4])
db  = int(sys.argv[5])

f_d1 = np.array(sys.argv[6].split(','))
e_d1 = np.array(sys.argv[7].split(','))

br = int(sys.argv[8])
rr = int(sys.argv[9])
x4 = int(sys.argv[10])

#----------------------------------------------------------------------------------------------------------------------

f0_d3, f1_d3 = data_all(
    img_fld_path = img_fld_path,
    seg_fld_path = seg_fld_path,
    fvc = fvc,
    db = db,
    f_d1 = f_d1,
    e_d1 = e_d1,
    br = br,
    rr = rr,
    x4 = x4)

#----------------------------------------------------------------------------------------------------------------------

X_d3 = np.append(f0_d3, f1_d3, axis = 0)
Y_d1 = np.append(np.full(f0_d3.shape[0], 0), np.full(f1_d3.shape[0], 1), axis = 0)

#----------------------------------------------------------------------------------------------------------------------

exm_pkl_dict = {"X_d3": X_d3, "Y_d1": Y_d1}

exm_pkl_file = open(exm_pkl_path, "wb")

pickle.dump(exm_pkl_dict, exm_pkl_file, protocol = 4)

exm_pkl_file.close()

#----------------------------------------------------------------------------------------------------------------------

print("X_d3")
print(X_d3.shape)

print("Y_d1")
print(Y_d1.shape)