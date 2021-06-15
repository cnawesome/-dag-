import numpy
import random
import DAG
import sys



def Compound_G(taskset1, taskset2, t_c, t_w, t1_c, t1_w):
    #将taskset2的所有节点重新标号，更改c矩阵
    #合成后的列表
    com_c = [0]
    t1 = taskset1
    t2 = taskset2
    #合成后的通信矩阵
    com = []
    com_w = []

    #合成任务列表
    t1_len = len(t1)
    for i in range(0, t1_len):
        #len(t1) = t2在合成图中的开始坐标
        com_c.append(i+1)
    com_c_len = len(com_c)
    t2_len = len(t2)
    for i in range(0, t2_len):
        #len(t2) = t2在合成图中的开始坐标
        com_c.append(com_c_len+i)
    print("initial:", com_c)

    #创建一个几行几列的零矩阵
    for x in range (len(com_c)):
        sublist = []
        com.append(sublist)
        for y in range(len(com_c)):
            sublist.append(0)

    for x in range (len(com_c)):
        sublist = []
        com_w.append(sublist)
        for y in range(3):
            sublist.append(0) 

    #print("inital c:", com)

    #创建一个合成图的通信开销矩阵
    '''
    #    实现了合并两个子矩阵通信开销
    '''
    for i in range(0, len(com_c)):
        for j in range(0, len(com_c)):
            if i == 0:
                if j == 1 or j == t1_len+1:
                    com[i][j] = 1
                else:
                    com[i][j] = 99
            if(0 < i < t1_len+1 and j>= t1_len+1 and i != 0):
                com[i][j] = 99
            if(i >= t1_len+1 and 0 <j < t1_len+1):
                com[i][j] = 99
            if(0 < i < t1_len+1 and 0 < j < t1_len+1):
                com[i][j] = t_c[i-1][j-1]   
            if(i >= t1_len+1 and j >= t1_len+1):
                com[i][j] = t1_c[i-(t1_len+1)][j-(t1_len+1)]
            if(i > 0 and j == 0):
                com[i][j] = 99
    print("compound c:")
    for i in com:
        print(i[:])
        print('\n')
    #创建一个合成图的处理器开销矩阵
    for j in range(0, len(com_c)):
        for k in range(0, 3):
            if j == 0:
                com_w[j][k] = 0 
            if 0 < j < t1_len+1:
                com_w[j][k] = t_w[j-1][k]
            if j>= t1_len+1:
                com_w[j][k] = t1_w[j-(t1_len+1)][k] 
    print("com_w:", )
    for i in com_w:
        print(i[:])
        print('\n')
             
    return com_c, com, com_w, com_c_len

def depth(set, c):
    de = []
    for i in set:
        sublist = []
        de.append(sublist)
    sub_l = []
    for i in set:
        DAG.get_depth(c, i, de[i])

    print(de)
    node_depth = []
    for i in range(len(de)):           
        node_depth.append(DAG.max_depth(de[i]))
    print(node_depth)
    return node_depth 

def generate_g(num):
    t_list = []
    for i in range(num):
        t_list.append(i)
    #for i in range(0,num):
    #    t_list.append(random.randint(0,num))
    print(t_list)
    t_c = generate_c(t_list, num)
    t_w = generate_w(t_list, num)
    
    return t_list, t_c, t_w

def generate_c(list, num):
    c = []
    #创建一个几行几列的零矩阵
    for x in range (0,num):
        sublist = []
        c.append(sublist)
        for y in range(0, num):
            sublist.append(0)
    #print("c:",c)
    for i in range(0, len(list)):
        randomNext = random.randint(1,num-i)
        for j in range(0, randomNext):
            postion = random.randint(i, num-1)
            if(postion != i):
                c[i][postion] = random.randint(10,40)
            #print('p:', postion)
            #print('c:',c)
        #重新循环给未选择为随机后继的位置赋值
        for k in range(0, len(c[i])):
            if c[i][k] == 0 and k != i:
                c[i][k] = 99
    return c    

def generate_w(list, num):
    w = []
    #创建一个几行几列的零矩阵
    for x in range (num):
        sublist = []
        w.append(sublist)
        for y in range(3):
            sublist.append(0)
            
    for i in range(len(list)):
        for j in range(len(w[i])):
            w[i][j] = random.randint(10, 30)
    return w

def task_generate(num):
    for i in range(int(num/2)):
        #设置任务个数
        t_num = random.randint(1,8)
        t_num1 = random.randint(1,8)
        #问题：自身会有一个值
        t_list, t_c, t_w = generate_g(t_num)
        t1_list, t1_c, t1_w = generate_g(t_num1)
        print("list:", t_list)
        print("com1:", t_c)
        print("w:", t_w)
        print("list:", t1_list)
        print("com2:", t1_c)
        print("w:", t1_w)
        c_list, c_g, w_g, dif_loc = Compound_G(t_list, t1_list, t_c, t_w, t1_c, t1_w)
        print("limit point:", dif_loc)
        for i in range(0, len(c_list)):
            DAG.SortList(c_list, c_g, c_list[i])
        print("sortlist:", DAG.sortlist)
        #获取节点深度
        d = depth(DAG.sortlist, c_g)
        c_tasks, uc_tasks, c_com, uc_com, w_com, uw_com = DAG.cut_comG(DAG.sortlist, dif_loc, d, c_g, w_g)
        
        DAG.random_Schedule(DAG.sortlist, c_g, w_g)
        #schedule_2:FCFS调度
        DAG.FCFS_m_schedule(DAG.sortlist, c_g, w_g)
        #schedule_3:HEFT调度
        DAG.HEFT_schedule(DAG.sortlist, w_g)
        #schedule_4:CHEFT调度
        DAG.CHEFT_schedule(c_tasks, c_com, w_com)
        DAG.CHEFT_schedule(uc_tasks, uc_com, uw_com)
 
if __name__ == '__main__':

    task_generate(6)
    
