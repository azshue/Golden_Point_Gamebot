# -*- coding: UTF-8 -*-
import numpy as np
import sys
import itertools
from math import floor 
from numpy import random

def LineToNums(line, type=float):
    return (type(cell) for cell in line.split('\t'))

def stage1(input1, input2):# input1,last golden point; input2:last-2 golden point
    if abs(input1 - input2) > 2.5:
        input = (input1 + input2)/2.0
    else:
        input = input1
    if(input >= 20):
        output = '20'
    elif(input >= 5 and input < 20):
        output = '%.0f' % input 
    elif(input < 5):
        input *= 100
        input = floor(input)/100.0
        output = '%.2f' % input
    return output

def stage2(history):# assert history has more than 5 lines
    flag_up = 0
    flag_down = 0
    for i in range(4):
        if history[-i-2][0] - history[-i-1][0] > 0:
            flag_down += 1
        elif history[-i-2][0] - history[-i-1][0] < 0:
            flag_up += 1
    if flag_down == 5:
        output = 'down'
    elif flag_up == 5:
        output = 'up'
    else:
        output = 'osci'
    return output
    
def stage3(history):#variance of all the input
    his_array = np.array(history)
    std_list = np.zeros(len(history))
    for i in range(len(history)):
        std_list[i] = np.std(his_array[i][1:])
    mean_std = np.mean(std_list[5:])
    if((std_list[-1] - mean_std) < 0 and abs(std_list[-1] - mean_std) >= 0.3 * mean_std):
        output = 'smallstep'
    elif((std_list[-1] - mean_std) > 0 and abs(std_list[-1] - mean_std) >= 0.3 * mean_std):
        output = 'bigstep'
    else:
        output = 'normalstep'
    
    return output 

def get_s(history):
    inter1 = stage1(history[-1][0], history[-2][0])# inter1: first level of index of the state_dict: decide the base value
    inter2 = stage2(history)# inter2: second level: decide the tendency(increase or decrease)
    inter3 = stage3(history)# inter3: third lecel: decide the step size
    return inter1, inter2, inter3

def get_base(idx1):
    base = float(idx1)
    if base >= 5:
        base += random.uniform(0.3, 0.7)
    else:
        base += random.uniform(0, 0.5)
    return base

def get_ratio(history):
    ratio = 0
    distance = 0
    for i in range(3):
        for j in range(len(history[0])-1):
            distance += abs(history[-i-1][j+1] - history[-i-1][0])
        distance = distance / (len(history[0])-1)
        ratio += distance/history[-i-1][0]
    ratio = ratio / 3.0
    return ratio

def get_a(idx, base, ratio):
    small = 0.6
    big = 1.0
    mid = 0.8
    coef = 0.34
    bias =  3.4

    if(idx == 0):#increase small step with the given base
        return base + small*np.exp(coef*base-bias)*ratio
    elif(idx == 1):#increase big step with the given base
        return base + big*np.exp(coef*base-bias)*ratio
    elif(idx == 2):#increse normal step with the given base
        return base + mid*np.exp(coef*base-bias)*ratio
    elif(idx == 3):#decrease small
        return base - small*np.exp(coef*base-bias)*ratio
    elif(idx == 4):#decrease big
        return base - big*np.exp(coef*base-bias)*ratio
    elif(idx == 5):#derease normal
        return base - mid*np.exp(coef*base-bias)*ratio
    elif(idx == 6):#around small
        return random.uniform(base - small*np.exp(coef*base-bias)*ratio, base + small*np.exp(coef*base-bias)*ratio)
    elif(idx == 7):#around big
        return random.uniform(base - big*np.exp(coef*base-bias)*ratio, base + big*np.exp(coef*base-bias)*ratio)
    elif(idx == 8):#around normal 
        return random.uniform(base - mid*np.exp(coef*base-bias)*ratio, base + mid*np.exp(coef*base-bias)*ratio)

def create_state_dict(list_1,list_2,list_3, num_act):
    
    state_dict ={}
    for i in range(len(list_1)):
        name_1 = list_1[i]
        state_1_dict = {}
        for j in range(len(list_2)):
            name_2 = list_2[j]
            state_2_dict = {}
            for k in range(len(list_3)):
                name_3 = list_3[k]
                state_2_dict[name_3] = [1/num_act for i in range(num_act)]
            state_1_dict[name_2] = state_2_dict
        state_dict[name_1] = state_1_dict
    return state_dict

def main():
    # fo = open('FirstData.txt')
    # metaLine = fo.readline()
    metaLine = sys.stdin.readline()
    lineNum, columnNum = LineToNums(metaLine, int)

    # read the training data  
    data = []
    for line in map(lambda _: sys.stdin.readline(), range(lineNum)):
        gnum, *nums = LineToNums(line)
        data.append((gnum, nums))
    # modify the data format
    data_1 = []
    for i in range(lineNum):
        temp = []
        temp.append(data[i][0])
        for j in range(columnNum-1):
            temp.append(data[i][1][j])
        data_1.append(temp)
    data =data_1

    # #actions
    num_act = 9
    # state_dict
    tmp_2 = ['%.2f' % x for x in np.arange(0, 5, 0.01)]
    tmp_2 = list(tmp_2)
    list_1 = ['20','19','18','17','16','15','14','13','12','11','10','9','8','7','6','5' ]
    list_1.extend(tmp_2)
    list_2 = ['down', 'up', 'osci']
    list_3 = ['smallstep', 'bigstep', 'normalstep']
    
    s_list = create_state_dict(list_1, list_2, list_3, num_act)
    
    # record the number of past rounds
    rd = 0

    # using all the past rounds data for training
    while(rd < lineNum):
        #print('round = ', rd)
        # compute candidates
        if rd == 0:
            candidate1 = 17 + random.uniform(0, 1)
            candidate2 = 17 - random.uniform(0, 1) 
            #print('candidate1, candidate2:', candidate1, candidate2)
            #print("%f\t%f" % (candidate1, candidate2))
            result1 = candidate1
            result2 = candidate2
            rd += 1
            continue
        if rd > 0 and rd < 5:
            ddata = np.array(data)
            candidate1 = np.mean(ddata[:,0]) * 0.8
            candidate2 = np.mean(ddata[:,0]) * 0.7
            result1 = candidate1
            result2 = candidate2
            #print('candidate1, candidate2:', candidate1, candidate2)
            rd += 1
            continue

        else:
            history = data[:rd]
            idx = []
            idx1, idx2, idx3 = get_s(history)
            probs = np.array(s_list[idx1][idx2][idx3])
            idxx = np.where(probs == np.max(probs))
            if(len(idxx) > 1):
                ids = random.randint(0, len(idxx), 2)
                idx.append(idxx[0][ids[0]])
                idx.append(idxx[0][ids[1]])
            if(idx == []):
                idx.append(idxx[0][0])
                idx_ = s_list[idx1][idx2][idx3].index(sorted(s_list[idx1][idx2][idx3])[-2])
                idx.append(idx_)

            base = get_base(idx1)
            #base = history[-1][0]
            ratio = get_ratio(history)
            #print('ratio:', np.exp(0.34*base-2.4)*ratio)
            candidate1 = get_a(idx[0], base, ratio)* 0.8
            candidate2 = get_a(idx[1], base, ratio)* 0.9
            result1 = candidate1
            result2 = candidate2
            #print("%f\t%f" % (candidate1, candidate2))
            #print('candidate1, candidate2:', candidate1, candidate2)

        # compute rewards 
        sigma = 0
        for i in range(1,len(data[rd])):
            sigma += data[rd][i]
        sigma += candidate1
        sigma += candidate2
        g_curr = (sigma / (len(data[rd])-1+2))*0.618
        #print('golden point', g_curr)

        gain = 0.1

        # good 
        # if candidate1 == g_curr or candidate2 == g_curr:
        array = np.array(data[rd])
        array -= g_curr
        array = abs(array)

        # if our choice is best
        if abs(candidate1 - g_curr) <= min(array[1:]):
            # update action probs
            s_list[idx1][idx2][idx3][idx[0]] += gain
            for i in range(len(s_list[idx1][idx2][idx3])):
                if i != idx[0]:
                    s_list[idx1][idx2][idx3][i] -= gain/(len(s_list[idx1][idx2][idx3])-1)
        if abs(candidate2 - g_curr) <= min(array[1:]):
            # update action probs
            s_list[idx1][idx2][idx3][idx[1]] += gain
            for i in range(len(s_list[idx1][idx2][idx3])):
                if i != idx[1]:
                    s_list[idx1][idx2][idx3][i] -= gain/(len(s_list[idx1][idx2][idx3])-1)

        # if our choice 40%
        imedia = array[1:]
        imedia = np.sort(imedia)
        tmp1 = imedia[int(0.4*len(imedia))]
        if abs(candidate1 - g_curr) < tmp1 and abs(candidate1 - g_curr) > min(array[1:]):
            #update
            idx0 = -1
            for u in range(len(imedia)):
                if imedia[u] >= abs(candidate1-g_curr):
                    idx0 = u+1
                    break
            pro = gain/idx0
            s_list[idx1][idx2][idx3][idx[0]] += pro
            for i in range(len(s_list[idx1][idx2][idx3])):
                if i != idx[0]:
                    s_list[idx1][idx2][idx3][i] -= pro/(len(s_list[idx1][idx2][idx3])-1)
        if abs(candidate2 - g_curr) < tmp1 and abs(candidate2 - g_curr) > min(array[1:]):
            #update
            idx0 = -1
            for u in range(len(imedia)):
                if imedia[u] >= abs(candidate2-g_curr):
                    idx0 = u+1
                    break
            pro = gain/idx0
            s_list[idx1][idx2][idx3][idx[1]] += pro
            for i in range(len(s_list[idx1][idx2][idx3])):
                if i != idx[0]:
                    s_list[idx1][idx2][idx3][i] -= pro/(len(s_list[idx1][idx2][idx3])-1)
            

        # if our choice -30% 
        imedia = array[1:]
        imedia = np.sort(imedia)
        tmp1 = imedia[int(0.7*len(imedia))]
        if abs(candidate1 - g_curr) > tmp1 and abs(candidate1 - g_curr) < max(abs(max(data[int(rd)][1:])-g_curr), abs(min(data[int(rd)][1:])-g_curr)):
            #update
            idx0 = -1
            for u in range(len(imedia)):
                if imedia[u] >= abs(candidate1-g_curr):
                    idx0 = u+1
                    break
            pro = gain/idx0
            s_list[idx1][idx2][idx3][idx[0]] -= pro
            for i in range(len(s_list[idx1][idx2][idx3])):
                if i != idx[0]:
                    s_list[idx1][idx2][idx3][i] += pro/(len(s_list[idx1][idx2][idx3])-1)
        if abs(candidate2 - g_curr) < tmp1 and abs(candidate2 - g_curr) < max(abs(max(data[int(rd)][1:])-g_curr), abs(min(data[int(rd)][1:])-g_curr)):
            #update
            idx0 = -1
            for u in range(len(imedia)):
                if imedia[u] >= abs(candidate2-g_curr):
                    idx0 = u+1
                    break
            pro = gain/idx0
            s_list[idx1][idx2][idx3][idx[1]] -= pro
            for i in range(len(s_list[idx1][idx2][idx3])):
                if i != idx[0]:
                    s_list[idx1][idx2][idx3][i] += pro/(len(s_list[idx1][idx2][idx3])-1)
        

        # if our choice is worst
        if abs(candidate1 - g_curr) >= max(abs(max(data[int(rd)][1:])-g_curr), abs(min(data[int(rd)][1:])-g_curr)):
            # update action probs
            s_list[idx1][idx2][idx3][idx[0]] -= gain
            for i in range(len(s_list[idx1][idx2][idx3])):
                if i != idx[0]:
                    s_list[idx1][idx2][idx3][i] += gain/(len(s_list[idx1][idx2][idx3])-1)
        if abs(candidate2 - g_curr) >= max(abs(max(data[int(rd)][1:])-g_curr), abs(min(data[int(rd)][1:])-g_curr)):
            # update action probs
            s_list[idx1][idx2][idx3][idx[1]] -= gain
            for i in range(len(s_list[idx1][idx2][idx3])):
                if i != idx[1]:
                    s_list[idx1][idx2][idx3][i] += gain/(len(s_list[idx1][idx2][idx3])-1)
        rd += 1
          

    # for k,v in s_list.items():
    #     
    #     for k_1, v_1 in v.items():
    #      
    #         for k_2, v_2 in v_1.items():
    #             #print(k,' ',k_1,' ', k_2)
    #             #print(v_2)
    print("%f\t%f" % (result1, result2))

if __name__ == '__main__':
    main()

        
            



