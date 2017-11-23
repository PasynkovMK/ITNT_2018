import numpy as np
import scipy.misc
import sys

#----------------------------------------------------------------------------------------------------------------------

def errors(seg_nd2, ref_nd2):

    FAR = np.sum(np.logical_and(np.logical_not(ref_nd2), seg_nd2))
    FRR = np.sum(np.logical_and(ref_nd2, np.logical_not(seg_nd2)))

    return(FAR, FRR)

#----------------------------------------------------------------------------------------------------------------------

seg_data_fld_path  = str(sys.argv[1])
ref_data_fld_path  = str(sys.argv[2])

report_path = str(sys.argv[3])

fvc = int(sys.argv[4])
db  = int(sys.argv[5])

f_nd1 = np.array(sys.argv[6].split(','))
e_nd1 = np.array(sys.argv[7].split(','))

#----------------------------------------------------------------------------------------------------------------------

FAR_nd1 = np.zeros(shape = (len(f_nd1) * len(e_nd1)))
FRR_nd1 = np.zeros(shape = (len(f_nd1) * len(e_nd1)))

#----------------------------------------------------------------------------------------------------------------------

i = 0

for f in f_nd1:

    for e in e_nd1:

        #--------------------------------------------------------------------------------------------------------------

        print("valid - fvc " + str(fvc) + " db " + str(db) + " f " + str(f) + " e " + str(e))

        #--------------------------------------------------------------------------------------------------------------

        path = "fvc/" + str(fvc) + "/db" + str(db) + "_b/" + str(f) + "_" + str(e) + ".tif"

        seg_nd2 = scipy.misc.imread(seg_data_fld_path + path) / 255
        ref_nd2 = scipy.misc.imread(ref_data_fld_path + path) / 255

        seg_nd2[seg_nd2 >= 0.5] = 1
        ref_nd2[ref_nd2 >= 0.5] = 1

        seg_nd2[seg_nd2 <= 0.5] = 0
        ref_nd2[ref_nd2 <= 0.5] = 0

        FAR, FRR = errors(seg_nd2 = seg_nd2, ref_nd2 = ref_nd2)

        FAR_nd1[i] = (FAR / (ref_nd2.shape[0] * ref_nd2.shape[1]))
        FRR_nd1[i] = (FRR / (ref_nd2.shape[0] * ref_nd2.shape[1]))

        i += 1

mean_FAR = np.mean(FAR_nd1)
mean_FRR = np.mean(FRR_nd1)

mean_sum = np.mean(FAR_nd1) + np.mean(FRR_nd1)

print("FAR: " + str(np.around(100 * mean_FAR, 3)) + '%')
print("FRR: " + str(np.around(100 * mean_FRR, 3)) + '%')
print("SUM: " + str(np.around(100 * mean_sum, 3)) + '%')

#----------------------------------------------------------------------------------------------------------------------

report_file = open(report_path, "w")

report_file.write("FAR: " + str(np.around(100 * mean_FAR, 3)) + '%' + '\n')
report_file.write("FRR: " + str(np.around(100 * mean_FRR, 3)) + '%' + '\n')
report_file.write("SUM: " + str(np.around(100 * mean_sum, 3)) + '%' + '\n')

report_file.close()