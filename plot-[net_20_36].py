import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------
# Параметры
#----------------------------------------------------------------------------------------------------------------------

br = 3
rr = 5
ms = 32

bl = 2 * br + 1
rl = 2 * rr + 1

NET = np.arange(20, 36 + 1, 1)
FVC = [2002, 2004]
VAR = [((rl * rl) * x + x + 2 * x + 2) for x in range(4, 20 + 1, 1)]

#----------------------------------------------------------------------------------------------------------------------
# Таблицы данных
#----------------------------------------------------------------------------------------------------------------------

FAR = np.zeros(shape = (len(FVC), len(NET)))
FRR = np.zeros(shape = (len(FVC), len(NET)))
SUM = np.zeros(shape = (len(FVC), len(NET)))
AIC = np.zeros(shape = (len(FVC), len(NET)))

#----------------------------------------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------------------------------------

for fvc in range(0, len(FVC)):

    for net in range(0, len(NET)):

        #----------------------------------------------------------------------------------------------------------
        # Считывание данных из report
        #----------------------------------------------------------------------------------------------------------

        var_path = "[br_" + str(br) + "_rr_" + str(rr) + "_net_" + str(NET[net]) + "_ms_" + str(ms) + "]"

        net_path = "report_text/fvc_" + str(FVC[fvc]) + "-" + var_path + ".txt"

        net_file = open(net_path, "r")

        FAR[fvc, net] = float(net_file.readline()[5:-1]) / 100
        FRR[fvc, net] = float(net_file.readline()[5:-1]) / 100
        SUM[fvc, net] = float(net_file.readline()[5:-1]) / 100

        #----------------------------------------------------------------------------------------------------------
        # Вычисление оценок критериев
        #----------------------------------------------------------------------------------------------------------

        AIC[fvc, net] = np.log10((SUM[fvc, net] ** 2)) + 2 * VAR[net] / max(VAR)

        #----------------------------------------------------------------------------------------------------------
        # Вывод в консоль
        #----------------------------------------------------------------------------------------------------------

        print("file: " + net_path)

        print("FAR: " + str(FAR[fvc, net]))
        print("FRR: " + str(FRR[fvc, net]))
        print("SUM: " + str(SUM[fvc, net]))
        print("VAR: " + str(VAR[net]))

        net_file.close()

    #------------------------------------------------------------------------------------------------------------------
    # Графики
    #------------------------------------------------------------------------------------------------------------------

    # plt.title("FVC: " + str(FVC[fvc]))
    # plt.plot(VAR, FAR[fvc,], '-o', label = "FAR")
    # plt.plot(VAR, FRR[fvc,], '-o', label = "FRR")
    # plt.plot(VAR, SUM[fvc,], '-o', label = "SUM")
    # plt.legend(loc = "upper right", numpoints = 1)
    # plt.show()

    # plt.title("FVC: " + str(FVC[fvc]))
    # plt.plot(NET, AIC[fvc,], '-o', label = "AIC")
    # plt.legend(loc = "upper right", numpoints = 1)
    # plt.show()

    plt.title("FVC: " + str(FVC[fvc]))
    plt.plot(range(4, 21), FAR[fvc,], '-o', label = "FAR")
    plt.plot(range(4, 21), FRR[fvc,], '-o', label = "FRR")
    plt.plot(range(4, 21), SUM[fvc,], '-o', label = "SUM")
    plt.legend(loc = "upper right", numpoints = 1)
    plt.show()

    plt.title("FVC: " + str(FVC[fvc]))
    plt.plot(range(4, 21), AIC[fvc,], '-o', label = "AIC")
    plt.legend(loc = "upper right", numpoints = 1)
    plt.show()