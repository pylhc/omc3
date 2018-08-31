import os
import numpy as np
import tfs


def merge_data(working_directory, magnet1, circuit1, magnet2, circuit2, beam, ip, tunemeasprecision):

    if ip in ['ip1', 'ip2', 'ip5', 'ip8', 'IP1', 'IP2', 'IP5', 'IP8']:
        IR = ip.lower() + beam.lower()
        sides = ['L', 'R']
        for side, magnet in zip(sides, [magnet1, magnet2]):

            tdatax = tfs.read(os.path.join(working_directory, IR + side + 'X.tfs'))
            tdatay = tfs.read(os.path.join(working_directory, IR + side + 'Y.tfs'))
            kdata = tfs.read(os.path.join(working_directory, ip + side + 'K.tfs'))
            K, Qx, Qxrms, Qy, Qyrms = pair(tdatax, tdatay, kdata, tunemeasprecision)

            write_tfs_files(K, Qx, Qxrms, Qy, Qyrms, working_directory, magnet, beam)

    else:
        for magnet, circuit in zip([magnet1,magnet2],[circuit1, circuit2]):
            tdatax = tfs.read(os.path.join(working_directory, circuit+'_tune_x_'+str(beam).lower()+'.tfs'))
            tdatay = tfs.read(os.path.join(working_directory, circuit+'_tune_y_'+str(beam).lower()+'.tfs'))
            kdata = tfs.read(os.path.join(working_directory, circuit + '_k.tfs'))
            K, Qx, Qxrms, Qy, Qyrms = pair(tdatax, tdatay, kdata, tunemeasprecision)

            write_tfs_files(K, Qx, Qxrms, Qy, Qyrms, working_directory, magnet, beam)


def write_tfs_files(K, Qx, Qxrms, Qy, Qyrms, working_directory, magnet, beam):
    # TODO replace with tfs.TFsDataFrame and tfs.write
    result = tfs_file_writer.TfsFileWriter.open(os.path.join(working_directory, magnet + '.' + beam + '.dat'))
    result.set_column_width(20)
    result.add_column_names(['K',    'TUNEX',     'TUNEX_ERR',     'TUNEY',     'TUNEY_ERR'])
    result.add_column_datatypes(['%le', '%le', '%le', '%le', '%le'])

    for i in range(len(K)):
        result.add_table_row([K[i], Qx[i], Qxrms[i], Qy[i], Qyrms[i] ])
    result.write_to_file()  


def pair(tdatax,tdatay,kdata, tunemeasprecision):
    Qx    = []
    Qxrms = []
    Qy    = []
    Qyrms = []
    K = []
    
    if len(tdatax.TIME) > len(kdata.TIME):
        step = 300
        for i in range(len(kdata.TIME)):
            if kdata.TIME[i] > tdatax.TIME[0] and kdata.TIME[i] < tdatax.TIME[len(tdatax.TIME)-1]:
                new_timex = tdatax.TIME - kdata.TIME[i]
                maskx  = (new_timex**2<step**2)
                tunex_mask= tdatax.TUNE[maskx]
                aveQx = np.average(tunex_mask)
                Qrmsx = np.std(tunex_mask)
                
                new_timey = tdatay.TIME - kdata.TIME[i]
                masky  = (new_timey**2<step**2)
                tuney_mask= tdatay.TUNE[masky]
                aveQy = np.average(tuney_mask)
                Qrmsy = np.std(tuney_mask)

                if len(tunex_mask)>0 and len(tuney_mask)>0:
                    Qx.append(aveQx)
                    Qxrms.append(Qrmsx)
                    Qy.append(aveQy)
                    Qyrms.append(Qrmsy)
                    K.append(kdata.K[i])

    Qxrms = np.sqrt(np.array(Qxrms) ** 2 + tunemeasprecision ** 2)
    Qyrms = np.sqrt(np.array(Qyrms) ** 2 + tunemeasprecision ** 2)
    return K, Qx, Qxrms, Qy, Qyrms



