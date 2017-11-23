#!/bin/bash

#----------------------------------------------------------------------------------------------------------------------
# img_data_fld_path - папка с входными изображениями
# seg_data_fld_path - папка с выходной маской сегментации
# seg_view_fld_path - папка с выходной маской сегментации наложенной на входное изобржение (иллюстрация работы)
# ref_data_fld_path - папка с вручную размеченной сегментацией
#
# learn_x4 - увеличение данных в 4 раза путем разворота
# check_x4 - увеличение данных в 4 раза путем разворота
#
# br - радиус блока
# rr - радиус области
# ms - параметер мультистарта
#
# net - номер используемой сети
# epochs - количество эпох обучения
# batch_size - batch_size используемый с нейронных сетях
#
# close - параметр морфологической обработки для устранения разрывов
# e_d1 - используемые оттиски каждого пальца
#----------------------------------------------------------------------------------------------------------------------

img_data_fld_path="data/img/"
seg_data_fld_path="data/seg/"
seg_view_fld_path="data/view/"
ref_data_fld_path="data/ref/"

learn_x4=1
check_x4=0

br=3
rr=5
ms=9

net=38
epochs=100
batch_size=32

close=10

e_d1=1,2,3,4,5,6,7,8

#----------------------------------------------------------------------------------------------------------------------

# for net in $(seq 16 36)
# do

    #------------------------------------------------------------------------------------------------------------------

    for fvc in 2002 2004
    do
        fvc_FAR=0
        fvc_FRR=0
        fvc_SUM=0

        i=0

        #--------------------------------------------------------------------------------------------------------------

        for db in 1 2 3 4
        do
            let ++i

            db_FAR=0
            db_FRR=0
            db_SUM=0

            j=0

            #----------------------------------------------------------------------------------------------------------

            for k in $(seq 1 9)
            do
                let ++j

                k_FAR=0
                k_FRR=0
                k_SUM=0

                check_f_d1=$((${k} + 100 + 0))
                valid_f_d1=$((${k} + 100 + 1))

                learn_f_d1="101,102,103,104,105,106,107,108,109,110,"

                learn_f_d1="${learn_f_d1/${check_f_d1},}"
                learn_f_d1="${learn_f_d1/${valid_f_d1},}"

                if [ ${learn_f_d1:0:1} == "," ]; then
                    learn_f_d1=${learn_f_d1:1}
                fi

                if [ ${learn_f_d1:(${#learn_f_d1} - 1)} == "," ]; then
                    learn_f_d1=${learn_f_d1::-1}
                fi

                echo "check: ${check_f_d1}"
                echo "valid: ${valid_f_d1}"
                echo "learn: ${learn_f_d1}"

                learn_path="examples_learn_pkl/fvc_${fvc}_db_${db}_k_${k}-[br_${br}_rr_${rr}].pkl"
                check_path="examples_check_pkl/fvc_${fvc}_db_${db}_k_${k}-[br_${br}_rr_${rr}].pkl"

                models_path="models_json/fvc_${fvc}_db_${db}_k_${k}-[br_${br}_rr_${rr}_net_${net}_ms_${ms}].json"
                weight_path="models_hdf5/fvc_${fvc}_db_${db}_k_${k}-[br_${br}_rr_${rr}_net_${net}_ms_${ms}].hdf5"
                report_path="report_text/fvc_${fvc}_db_${db}_k_${k}-[br_${br}_rr_${rr}_net_${net}_ms_${ms}]+l.txt"

                #------------------------------------------------------------------------------------------------------

                create_l_args[1]="${img_data_fld_path}"
                create_l_args[2]="${ref_data_fld_path}"
                create_l_args[3]="${learn_path}"
                create_l_args[4]="${fvc}"
                create_l_args[5]="${db}"
                create_l_args[6]="${learn_f_d1}"
                create_l_args[7]="${e_d1}"
                create_l_args[8]="${br}"
                create_l_args[9]="${rr}"
                create_l_args[10]="${learn_x4}"

                if [ ! -f ${learn_path} ]; then
                    python lib/create.py "${create_l_args[@]}"
                fi

                #------------------------------------------------------------------------------------------------------

                create_c_args[1]="${img_data_fld_path}"
                create_c_args[2]="${ref_data_fld_path}"
                create_c_args[3]="${check_path}"
                create_c_args[4]="${fvc}"
                create_c_args[5]="${db}"
                create_c_args[6]="${check_f_d1}"
                create_c_args[7]="${e_d1}"
                create_c_args[8]="${br}"
                create_c_args[9]="${rr}"
                create_c_args[10]="${check_x4}"

                if [ ! -f ${check_path} ]; then
                    python lib/create.py "${create_c_args[@]}"
                fi

                #------------------------------------------------------------------------------------------------------

                net_args[1]="${learn_path}"
                net_args[2]="${check_path}"
                net_args[3]="${models_path}"
                net_args[4]="${weight_path}"
                net_args[5]="${rr}"
                net_args[6]="${ms}"
                net_args[7]="${epochs}"
                net_args[8]="${batch_size}"

                if [[ ! -f ${models_path} || ! -f ${weight_path} ]]; then
                    python lib/net_${net}.py "${net_args[@]}"
                fi

                #------------------------------------------------------------------------------------------------------

                apply_args[1]="${img_data_fld_path}"
                apply_args[2]="${seg_data_fld_path}"
                apply_args[3]="${seg_view_fld_path}"
                apply_args[4]="${models_path}"
                apply_args[5]="${weight_path}"
                apply_args[6]="${fvc}"
                apply_args[7]="${db}"
                apply_args[8]="${br}"
                apply_args[9]="${rr}"
                apply_args[10]="${valid_f_d1}"
                apply_args[11]="${e_d1}"
                apply_args[12]="${batch_size}"

                python lib/apply.py "${apply_args[@]}"

                #------------------------------------------------------------------------------------------------------

                label_args[1]="${img_data_fld_path}"
                label_args[2]="${seg_data_fld_path}"
                label_args[3]="${seg_view_fld_path}"
                label_args[4]="${fvc}"
                label_args[5]="${db}"
                label_args[6]="${valid_f_d1}"
                label_args[7]="${e_d1}"
                label_args[8]="${close}"

                python lib/label.py "${label_args[@]}"

                #------------------------------------------------------------------------------------------------------

                valid_agrs[1]="${seg_data_fld_path}"
                valid_agrs[2]="${ref_data_fld_path}"
                valid_agrs[3]="${report_path}"
                valid_agrs[4]="${fvc}"
                valid_agrs[5]="${db}"
                valid_agrs[6]="${valid_f_d1}"
                valid_agrs[7]="${e_d1}"

                python lib/valid.py "${valid_agrs[@]}"

                #------------------------------------------------------------------------------------------------------

                k_report_path="report_text/fvc_${fvc}_db_${db}_k_${k}-[br_${br}_rr_${rr}_net_${net}_ms_${ms}]+l.txt"

                k_FAR=$(sed -n '1p' < ${k_report_path})
                k_FRR=$(sed -n '2p' < ${k_report_path})
                k_SUM=$(sed -n '3p' < ${k_report_path})

                k_FAR=${k_FAR:5:-1}
                k_FRR=${k_FRR:5:-1}
                k_SUM=${k_SUM:5:-1}

                db_FAR=$(bc -l <<< "${db_FAR} + ${k_FAR}")
                db_FRR=$(bc -l <<< "${db_FRR} + ${k_FRR}")
                db_SUM=$(bc -l <<< "${db_SUM} + ${k_SUM}")

            #----------------------------------------------------------------------------------------------------------

            done

            db_report_path="report_text/fvc_${fvc}_db_${db}-[br_${br}_rr_${rr}_net_${net}_ms_${ms}]+l.txt"

            db_FAR=$(bc -l <<< "scale=3; ${db_FAR} / ${j}")
            db_FRR=$(bc -l <<< "scale=3; ${db_FRR} / ${j}")
            db_SUM=$(bc -l <<< "scale=3; ${db_SUM} / ${j}")

            rm -f ${db_report_path}

            echo ${db_FAR} | bc | awk '{printf "FAR: %.3f\n", $0}' >> "${db_report_path}"
            echo ${db_FRR} | bc | awk '{printf "FRR: %.3f\n", $0}' >> "${db_report_path}"
            echo ${db_SUM} | bc | awk '{printf "SUM: %.3f\n", $0}' >> "${db_report_path}"

            fvc_FAR=$(bc -l <<< "${fvc_FAR} + ${db_FAR}")
            fvc_FRR=$(bc -l <<< "${fvc_FRR} + ${db_FRR}")
            fvc_SUM=$(bc -l <<< "${fvc_SUM} + ${db_SUM}")

        #--------------------------------------------------------------------------------------------------------------

        done

        fvc_report_path="report_text/fvc_${fvc}-[br_${br}_rr_${rr}_net_${net}_ms_${ms}]+l.txt"

        fvc_FAR=$(bc -l <<< "scale=3; ${fvc_FAR} / ${i}")
        fvc_FRR=$(bc -l <<< "scale=3; ${fvc_FRR} / ${i}")
        fvc_SUM=$(bc -l <<< "scale=3; ${fvc_SUM} / ${i}")

        rm -f ${fvc_report_path}

        echo ${fvc_FAR} | bc | awk '{printf "FAR: %.3f\n", $0}' >> "${fvc_report_path}"
        echo ${fvc_FRR} | bc | awk '{printf "FRR: %.3f\n", $0}' >> "${fvc_report_path}"
        echo ${fvc_SUM} | bc | awk '{printf "SUM: %.3f\n", $0}' >> "${fvc_report_path}"

        #--------------------------------------------------------------------------------------------------------------

    done

# done