import numpy as np
import scipy.ndimage
import scipy.misc
import sys

#----------------------------------------------------------------------------------------------------------------------

from skimage.measure import label, regionprops
from skimage.morphology import erosion, dilation

#----------------------------------------------------------------------------------------------------------------------

img_data_fld_path = str(sys.argv[1])
seg_data_fld_path = str(sys.argv[2])
seg_view_fld_path = str(sys.argv[3])

fvc = int(sys.argv[4])
db  = int(sys.argv[5])

f_nd1 = np.array(sys.argv[6].split(','))
e_nd1 = np.array(sys.argv[7].split(','))

close = int(sys.argv[8])

#----------------------------------------------------------------------------------------------------------------------

for f in f_nd1:

    for e in e_nd1:

        #--------------------------------------------------------------------------------------------------------------

        print("label - fvc " + str(fvc) + " db " + str(db) + " f " + str(f) + " e " + str(e))

        #--------------------------------------------------------------------------------------------------------------

        path = "fvc/" + str(fvc) + '/db' + str(db) + '_b/' + str(f) + '_' + str(e) + ".tif"

        img_nd2 = scipy.misc.imread(img_data_fld_path + path) / 255

        seg_nd2 = scipy.misc.imread(seg_data_fld_path + path) / 255

        lab_nd2 = label(seg_nd2)

        lab_nd2[lab_nd2 != np.argmax(np.array([i.area for i in regionprops(lab_nd2)])) + 1] = 0

        for i in range(close):

            lab_nd2 = dilation(lab_nd2)

        for i in range(close):

            lab_nd2 = erosion(lab_nd2)

        scipy.misc.toimage(lab_nd2, cmin = 0, cmax = 1).save(seg_data_fld_path + path)

        scipy.misc.toimage(lab_nd2 * img_nd2, cmin = 0, cmax = 1).save(seg_view_fld_path + path)

