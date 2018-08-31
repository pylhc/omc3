import numpy as np
import re


def findQuadrupoleType(searchstring, beam, twiss):

    magnet = re.findall('MQ\w+' + searchstring, ''.join(list(twiss.NAME)))

    return magnet[0]


def MagnetSpecs(magnetname, beam, twiss):

    if magnetname in twiss.NAME:
        index = twiss.NAME.index(magnetname)
        position = twiss.S[index]
        k = twiss.K1L[index]
        length = twiss.L[index]
        Polarity = twiss.POLARITY[index]

    elif magnetname+'.'+beam in twiss.NAME:
        index = twiss.NAME.index(magnetname+'.'+beam)
        position = twiss.S[index]
        k = twiss.K1L[index]
        length = twiss.L[index]
        Polarity = twiss.POLARITY[index]


    return position, k, length, Polarity


def MagnetPolarity(magnetname, beam, twiss):

    if magnetname in twiss.NAME:
        index = twiss.NAME.index(magnetname)

        Polarity = np.sign(twiss.K1L[index])

    elif magnetname+'.'+beam in twiss.NAME:
        index = twiss.NAME.index(magnetname+'.'+beam)

        Polarity = np.sign(twiss.K1L[index])

    

    return Polarity


def MagnetLength(magnetname, beam, twiss):

    if magnetname in twiss.NAME:
        index = twiss.NAME.index(magnetname)

        Length = twiss.L[index]

    elif magnetname+'.'+beam in twiss.NAME:
        index = twiss.NAME.index(magnetname+'.'+beam)

        Length = twiss.L[index]


    return Length


def MagnetPosition(magnetname, beam, twiss):

    if magnetname in twiss.NAME:
        index = twiss.NAME.index(magnetname)

        position = twiss.S[index]

    elif magnetname+'.'+beam in twiss.NAME:
        index = twiss.NAME.index(magnetname+'.'+beam)

        position = twiss.S[index]

    return position


def Lstar(magnetname1,magnetname2, beam, twiss):

    position1 = MagnetPosition(magnetname1, beam, twiss)
    position2 = MagnetPosition(magnetname2, beam, twiss)

    l1 = MagnetLength(magnetname1, beam, twiss)
    l2 = MagnetLength(magnetname2, beam, twiss)

    if position1 < position2:
        L_star = (abs(position1 - position2) - l2) / 2.
    elif position2 < position1:
        L_star = (abs(position1 - position2) - l1) / 2.

    # assuming that positions in twiss file are at the end of the element
    return L_star


def LstarPosition(magnetname1, magnetname2, beam, twiss):

    L_star = Lstar(magnetname1, magnetname2, beam, twiss)

    Magnet1Pos = MagnetPosition(magnetname1, beam, twiss)
    Magnet2Pos = MagnetPosition(magnetname2, beam, twiss)
    if Magnet1Pos < Magnet2Pos:
        L_star_pos = Magnet1Pos + L_star
    elif Magnet2Pos < Magnet1Pos:
        L_star_pos = Magnet2Pos + L_star
    else:
        L_star_pos = 0

    return L_star_pos

def IsInTwiss(name,beam,twiss):

    if name in twiss.NAME:
        return True
    else:
        return False


def FindParentBetweenMagnets(magnetname1, magnetname2, name, beam, twiss):

    if IsInTwiss(magnetname1, beam, twiss):
        index1 = twiss.NAME.index(magnetname1)
    elif IsInTwiss(magnetname1 + '.' + beam, beam, twiss):
        index1 = twiss.NAME.index(magnetname1+'.'+beam)

    if IsInTwiss(magnetname2, beam, twiss):
        index2 = twiss.NAME.index(magnetname2)
    elif IsInTwiss(magnetname2 + '.' + beam, beam, twiss):
        index2 = twiss.NAME.index(magnetname2 + '.' + beam)

    if name in twiss.PARENT[min(index1, index2):max(index1, index2)]:
        return True
    else:
        return False


def FindKeywordBetweenMagnets(magnetname1, magnetname2, name, beam, twiss):

    if IsInTwiss(magnetname1, beam, twiss):
        index1 = twiss.NAME.index(magnetname1)
    elif IsInTwiss(magnetname1 + '.' + beam, beam, twiss):
        index1 = twiss.NAME.index(magnetname1+'.'+beam)

    if IsInTwiss(magnetname2, beam, twiss):
        index2 = twiss.NAME.index(magnetname2)
    elif IsInTwiss(magnetname2 + '.' + beam, beam, twiss):
        index2 = twiss.NAME.index(magnetname2 + '.' + beam)



    if name in twiss.KEYWORD[min(index1, index2):max(index1, index2)]:
        return True
    else:
        return False


def ReturnDataofBPMinBetweenMagnets(magnetname1, magnetname2, BPMKeyword, beam, twiss):

    results_name = []
    results_pos = []

    if IsInTwiss(magnetname1, beam, twiss):
        index1 = twiss.NAME.index(magnetname1)
    elif IsInTwiss(magnetname1 + '.' + beam, beam, twiss):
        index1 = twiss.NAME.index(magnetname1+'.'+beam)

    if IsInTwiss(magnetname2, beam, twiss):
        index2 = twiss.NAME.index(magnetname2)
    elif IsInTwiss(magnetname2 + '.' + beam, beam, twiss):
        index2 = twiss.NAME.index(magnetname2 + '.' + beam)

    for key, name, pos in zip(twiss.KEYWORD[min(index1, index2): max(index1, index2)], twiss.NAME[min(index1, index2): max(index1, index2)], twiss.S[min(index1, index2): max(index1, index2)]) :
        if key == BPMKeyword:
            results_name.append(name)
            results_pos.append(pos)

    return results_name, results_pos


if __name__ == '__main__':
    MagnetSpecs('MQXA.1L1', 'B1')

    Lstar('MQXA.1L1', 'MQXA.1R1', 'B1')

    # print(FindParentBetweenMagnets('MQXA.1L1', 'MQXA.1R1', 'OMK', 'B1'))
    print(FindKeywordBetweenMagnets('MQXA.1L1', 'MQXA.1R1', 'MONITOR', 'B1'))
    ReturnDataofBPMinBetweenMagnets('MQXA.1L1', 'MQXA.1R1', 'MONITOR', 'B1')



