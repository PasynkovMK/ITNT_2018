import numpy as np
import scipy.misc
import keras
import sys

from keras.models import load_model
from keras.models import model_from_json

#----------------------------------------------------------------------------------------------------------------------

def find_center(n, rr, br):

    A_nd1 = [i for i in range(rr, n - rr, 2 * br + 1)]

    return(A_nd1)

#----------------------------------------------------------------------------------------------------------------------

def Nd2_to_Nd3(A_nd2, rr, br):

    y_nd1 = find_center(n = A_nd2.shape[0], rr = rr, br = br)
    x_nd1 = find_center(n = A_nd2.shape[1], rr = rr, br = br)

    rl = 2 * rr + 1

    A_nd3 = np.zeros(shape = (len(y_nd1) * len(x_nd1), rl, rl), dtype = A_nd2.dtype)

    n = 0

    for y in y_nd1:
        
        for x in x_nd1:

            A_nd3[n,] = A_nd2[(y - rr):(y + rr + 1),(x - rr):(x + rr + 1)]

            n += 1

    return(A_nd3)

#----------------------------------------------------------------------------------------------------------------------

def Nd3_to_Nd2(A_nd3, nc, nr, rr, br):

    y_nd1 = find_center(n = nr, rr = rr, br = br)
    x_nd1 = find_center(n = nc, rr = rr, br = br)

    bl = 2 * br + 1

    A_nd2 = np.zeros(shape = (nr, nc), dtype = A_nd3.dtype)

    i = 0

    for y in y_nd1:
        
        for x in x_nd1:

            A_nd2[(y - br):(y + br + 1),(x - br):(x + br + 1)] = 1 if (A_nd3[i,1] >= 0.5) else 0

            i += 1

    return(A_nd2)

#----------------------------------------------------------------------------------------------------------------------

img_data_fld_path = str(sys.argv[1])
seg_data_fld_path = str(sys.argv[2])
seg_view_fld_path = str(sys.argv[3])

models_path = str(sys.argv[4])
weight_path = str(sys.argv[5])

fvc = int(sys.argv[6])
db  = int(sys.argv[7])
br  = int(sys.argv[8])
rr  = int(sys.argv[9])

f_nd1 = np.array(sys.argv[10].split(','))
e_nd1 = np.array(sys.argv[11].split(','))

batch_size = int(sys.argv[12])

#----------------------------------------------------------------------------------------------------------------------

model_file = open(models_path, "r")

model_json = model_file.read()

model_file.close()

model = model_from_json(model_json)

model.load_weights(weight_path)

print("Loaded model from disk")

#----------------------------------------------------------------------------------------------------------------------

for f in f_nd1:

    for e in e_nd1:

        #--------------------------------------------------------------------------------------------------------------

        print("apply - fvc " + str(fvc) + " db " + str(db) + " f " + str(f) + " e " + str(e))

        #--------------------------------------------------------------------------------------------------------------

        path = "fvc/" + str(fvc) + "/db" + str(db) + "_b/" + str(f) + "_" + str(e) + ".tif"

        img_nd2 = scipy.misc.imread(img_data_fld_path + path) / 255

        img_nd3 = Nd2_to_Nd3(A_nd2 = img_nd2, rr = rr, br = br)

        img_nd3 = img_nd3.reshape(img_nd3.shape[0], 4 * (rr * rr + rr) + 1)

        seg_nd3 = model.predict(img_nd3, batch_size = batch_size, verbose = 0)

        seg_nd2 = Nd3_to_Nd2(A_nd3 = seg_nd3, nc = img_nd2.shape[1], nr = img_nd2.shape[0], rr = rr, br = br)

        scipy.misc.toimage(seg_nd2, cmin = 0, cmax = 1).save(seg_data_fld_path + path)

        scipy.misc.toimage(img_nd2 * seg_nd2, cmin = 0, cmax = 1).save(seg_view_fld_path + path)