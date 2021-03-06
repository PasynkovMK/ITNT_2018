import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------
# Параметры
#----------------------------------------------------------------------------------------------------------------------

br = 3
rr = 5
MS = np.arange(1, 32 + 1, 1)

bl = 2 * br + 1
rl = 2 * rr + 1

net = 22
FVC = [2002, 2004]

#----------------------------------------------------------------------------------------------------------------------
# Таблицы данных
#----------------------------------------------------------------------------------------------------------------------

FAR = np.zeros(shape = (len(FVC), len(MS)))
FRR = np.zeros(shape = (len(FVC), len(MS)))
SUM = np.zeros(shape = (len(FVC), len(MS)))

#----------------------------------------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------------------------------------

for fvc in range(0, len(FVC)):

    for ms in range(0, len(MS)):

        #----------------------------------------------------------------------------------------------------------
        # Считывание данных из report
        #----------------------------------------------------------------------------------------------------------

        var_path = "[br_" + str(br) + "_rr_" + str(rr) + "_net_" + str(net) + "_ms_" + str(MS[ms]) + "]"

        net_path = "report_text/fvc_" + str(FVC[fvc]) + "-" + var_path + ".txt"

        net_file = open(net_path, "r")

        FAR[fvc, ms] = float(net_file.readline()[5:-1]) / 100
        FRR[fvc, ms] = float(net_file.readline()[5:-1]) / 100
        SUM[fvc, ms] = float(net_file.readline()[5:-1]) / 100

        #----------------------------------------------------------------------------------------------------------
        # Вывод в консоль
        #----------------------------------------------------------------------------------------------------------

        print("file: " + net_path)

        print("FAR: " + str(FAR[fvc, ms]))
        print("FRR: " + str(FRR[fvc, ms]))
        print("SUM: " + str(SUM[fvc, ms]))

        net_file.close()

    #------------------------------------------------------------------------------------------------------------------
    # Графики
    #------------------------------------------------------------------------------------------------------------------

    minMS = MS[list(SUM[fvc,]).index(min(SUM[fvc,]))]
    minSUM = np.around(SUM[fvc, minMS - 1] * 100, 2)

    plt.title("FVC: " + str(FVC[fvc]) + " - min MS: " + str(minMS) + " - min SUM: " + str(minSUM) + "%")
    plt.plot(MS, FAR[fvc,], '-o', label = "FAR")
    plt.plot(MS, FRR[fvc,], '-o', label = "FRR")
    plt.plot(MS, SUM[fvc,], '-o', label = "SUM")
    plt.legend(loc = "upper right", numpoints = 1)
    plt.show()