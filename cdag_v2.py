import numpy
import math 
import matplotlib.pyplot as plt
from functools import partial
import copy
import random

'''
    
task_set1:
               (0)
             /  |  \
            /   |   \
           /    |    \
          /     |     \
        (1)    (2)—— ——(3)
         \      /  \   /
          \    /    \ /
           \  /     (5)   
           (4)     /
            \     /
             \   / 
              (6)

task_set2:
              (0)
             /  \
            /    \
           /      \    
          /        \     
        (1)—— —— —— (2)
        / \       
       /   \      
      /     \     
    (3)      (4)   
           
'''

task_set = [0, 1, 2, 3, 4, 5, 6]

task_set1 = [0, 1, 2, 3, 4]
#计算开销矩阵
w = [[14, 16, 9],
    [13, 19, 18],
    [11, 13, 19],
    [13, 8, 17],
    [12, 13, 10],
    [13, 16, 9],
    [7, 15, 11]]

w1 = [[15, 18, 8],
    [12, 18, 16],
    [11, 15, 18],
    [12, 8, 20],
    [14, 16, 12]]

#通信开销矩阵
c = [[0, 18, 9, 14, 99, 99 , 99],
    [99, 0, 99, 99, 99, 12, 99],
    [99, 99, 0, 99, 18, 22, 99],
    [99, 99, 11, 0, 99, 15, 99],
    [99, 99, 99, 99, 0, 99, 7],
    [99, 99, 99, 99, 99, 0, 9],
    [99, 99, 99, 99, 99, 99, 0]
    ]

c1 = [[0, 8, 12, 99, 99],
    [99, 0, 99, 15, 20],
    [99, 16, 0, 99, 99],
    [99, 99, 99, 0, 99],
    [99, 99, 99, 99, 0]
    ]

#深度开销矩阵
d = []

def max_depth():
    #遍历图，找到每一个最后结点，比较出最大的depth
    pass

def succ(c, node_id):
    succ_list = []
    for i in range(len(c[node_id])):
            if (c[node_id][i]!= 0 and c[node_id][i]!= 99):
                    succ_list.append(i)
    return succ_list

def pre(c, node_id):
    pre_list = []
    for i in range(len(c)):
            if (c[i][node_id]!= 0 and c[i][node_id]!= 99):
                    pre_list.append(i)
    return pre_list


def get_depth(c, ni, de, d=0):
    pres = pre(c, ni)
    if(pre(c, ni)):
        d += 1
        for i in pres:
            get_depth(c, i, de, d)    
    else:
        de.append(d)

def max_depth(ls):
    max = 0
    for i in ls:
        if i > max:
            max = i 
    return max

#获取节点的深度
'''
    find the most depth road is difficult
    so，wo try find depth of all node in road and find the max depth  
'''
def depth(set, c):
    de = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    sub_l = []
    for i in set:
        get_depth(c, i, de[i])

    print(de)
    node_depth = []
    for i in range(len(de)):           
        node_depth.append(max_depth(de[i]))
    print(node_depth)
    return node_depth 

#DAG可靠性模型
def Nerror(n_i, p_u):
    #任务i发生故障的均值
    lamda = 1
    nerror = math.exp(-lamda *w[n_i][p_u])
    return nerror

def compound_G(taskset1, taskset2):
    #将taskset2的所有节点重新标号，更改c矩阵
    com_c = [0]
    t1 = taskset1
    t2 = taskset2
    com = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    com_w = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

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

    for i in range(0, len(com_c)):
        for j in range(0, len(com_c)):
            #13*13
            if i == 0 :
                if j == 1 or j == t1_len+1 :
                    com[i].append(1)
                else:
                    com[i].append(99)
            if(0 < i < t1_len+1 and j>= t1_len+1 and i != 0):
                com[i].append(99)
            if(i >= t1_len+1 and 0 <j< t1_len+1):
                com[i].append(99)
            if(0 < i < t1_len+1 and 0 < j < t1_len+1):
                com[i].append(c[i-1][j-1])   
            if(i >= t1_len+1 and j >= t1_len+1):
                com[i].append(c1[i-(t1_len+1)][j-(t1_len+1)])
            if(i > 0 and j == 0):
               com[i].append(99)
    
    for i in com:
        print(i[:])
        print('\n')
    
    #创建一个合成图的处理器开销矩阵
    for j in range(0, len(com_c)):
        for k in range(0, 3):
            if j == 0:
                com_w[j].append(0)
            if 0 < j < t1_len+1:
                com_w[j].append(w[j-1][k])
            if j>= t1_len+1:
                com_w[j].append(w1[j-(t1_len+1)][k])
             
    return com_c, com, com_w, com_c_len

def cut_comG(task, loc, dep, c, w):
    #合成图的深度矩阵重新分为两个图的深度矩阵
    task1 = task[:loc]
    task2 = task[loc:]
    dep1 = dep[:loc]
    dep2 = dep[loc:]
    print("tasks and depths:", task1, task2, dep1, dep2)
    max_d1 = max(dep1)
    max_d2 = max(dep2)
    print("max depth:",max_d1, max_d2)
    
    #条件队列
    con_tasks1 = []
    con_tasks2 = []
    ucon_tasks = []

    for i in task:
        if dep[i] <= max_d2 and i in task1:
            con_tasks1.append(i)
        elif dep[i] <= max_d2 and i in task2:
            con_tasks2.append(i)
        else:
            ucon_tasks.append(i)
    
    con_tasks = con_tasks1 + con_tasks2

    print(con_tasks1)
    print(con_tasks2)
    print(con_tasks)
    tasks_len = len(con_tasks1) + len(con_tasks2)

    comcom = copy.deepcopy(c)
    ucomcom = copy.deepcopy(c)

    comp = copy.deepcopy(w)
    ucomp = copy.deepcopy(w)

    for t in task:
        if t not in con_tasks:
            for i in range(len(comcom[t])):
                comcom[t][i] = 99
        else:
            for j in task:
                if j not in con_tasks:
                    comcom[t][j] = 99


    #不满足深度的条件队列的通信开销矩阵
    for t in task:
        if t not in ucon_tasks:
            for i in range(len(comcom[t])):
                ucomcom[t][i] = 99
        else:
            for j in task:
                if j not in ucon_tasks:
                    ucomcom[t][j] = 99
    
    #处理器开销矩阵
    for t in task:
        if t not in con_tasks:
            for i in range(len(comp[t])):
                comp[t][i] = 99

    for t in task:
        if t not in ucon_tasks:
            for i in range(len(comp[t])):
                ucomp[t][i] = 99   

    
    print("com:")
    for i in comcom:
        print(i[:])
        print('\n')
    print("ucom:")    
    for i in ucomcom:
        print(i[:])
        print('\n')
    print("com:")
    for i in comp:
        print(i[:])
        print('\n')
    print("ucom:")    
    for i in ucomp:
        print(i[:])
        print('\n')  
    return con_tasks, ucon_tasks, comcom, ucomcom, comp, ucomp
    


def FCFS_schedule(set):
    task_set = set

    #最早开始时间
    EST = 0
    #最晚结束时间
    EFT = 0 

    per_task = task_set
    per_task_all_cost = [] 
    sum_cost = 0 
    task_cost_sum = [] 
    for i in range(len(task_set)):          
        print("process:")
        max_cpu_cost = 0
        for k in w[task_set[i]]:        
            if ( k > max_cpu_cost):
                max_cpu_cost = k 
        if(i < len(task_set)-1):
            if(c[task_set[i]][task_set[i+1]]!= 0 and c[task_set[i]][task_set[i+1]]!= 99):                  
                cost = c[task_set[i]][task_set[i+1]] + max_cpu_cost
            else:
                cost = max_cpu_cost
        else:
            cost = max_cpu_cost
    
        sum_cost = cost + sum_cost    
        per_task_all_cost.append(cost)
        task_cost_sum.append(sum_cost)
    allcost = sum(per_task_all_cost[:5])
    print("all cost:",allcost)
    fig_FCFS_bar(per_task, per_task_all_cost)
    fig_FCFS(per_task, task_cost_sum)



def random_Schedule(set, c, w):
    '''
        random number appear not only once
    '''
    seq = []
    cost = 0
    step = 0
    while step < len(set):
        r = random.randint(0,12)
        if r not in seq:
            seq.append(r)
            step += 1
            if seq:
                cost += max(w[r]) + c[seq[-1]][r]
            else:
                cost += max(w[r])
    print("initial list:", set)
    print("seq:", seq)
    print("cost:", cost)
    return seq

def FCFS_m_schedule(set, c, w):
    task_set = set

    #最早开始时间
    EST = 0
    #最晚结束时间
    EFT = 0 

    per_task = task_set
    per_task_all_cost = [] 
    sum_cost = 0 
    task_cost_sum = [] 
    for i in range(len(task_set)):          
        print("process:")
        max_cpu_cost = 0
        for k in w[task_set[i]]:        
            if ( k > max_cpu_cost):
                max_cpu_cost = k 
        if(i < len(task_set)-1):
            if(c[task_set[i]][task_set[i+1]]!= 0 and c[task_set[i]][task_set[i+1]]!= 99):                  
                cost = c[task_set[i]][task_set[i+1]] + max_cpu_cost
            else:
                cost = max_cpu_cost
        else:
            cost = max_cpu_cost
    
        sum_cost = cost + sum_cost    
        per_task_all_cost.append(cost)
        task_cost_sum.append(sum_cost)
    allcost = sum(per_task_all_cost[:5])
    print("all cost:",allcost)
    fig_FCFS_bar(per_task, per_task_all_cost)
    fig_FCFS(per_task, task_cost_sum)



#sortlist：符合任务优先级的任务队列
sortlist = []
def SortList(set, set_g, nodei):
    set = set
    preset = pre(set_g, nodei)
    if (len(preset)):
        for i in preset:
            SortList(set,set_g, i)
            if i not in sortlist:
                sortlist.append(i)  
    else:
        if nodei not in sortlist:
            sortlist.append(nodei)
    if nodei not in sortlist:
          sortlist.append(nodei)                    

def wbar(ni, ps=w):
    """ Average computation cost """
    return sum(p for p in ps[ni]) / len(ps[ni])

def cbar(ni, nj, ps=w):
    """ Average communication cost """
    n = len(ps)
    comsum = 0
    if n == 1:
        return 0
    npairs = n * (n-1)
    print("npairs:",npairs)
    return 1. * sum(c[ni][nj] for a1 in ps[ni] for a2 in ps[nj] 
                                        if a1 != a2 and c[ni][nj] != 99) / npairs

job_v = []
rank_v = []
def ranku(ni, ps=w):
    rank = partial(ranku, ps=w)
    wf = partial(wbar, ps=w)
    cf = partial(cbar, ps=w)
    rank_value = 0 

    if ni in c and c[ni]:
        rank_value = wf(ni) + max(cf(ni, nj) + rank(nj) for nj in c[i] if c[i][nj] !=0 and c[i][nj]!= 99)
        print("task prior ->", ni)
        print("rank_value:",rank_value)
        job_v.append(ni)
        rank_v.append(rank_value)
        return rank_value 
    else:
        print("task prior ->", ni)
        rank_value = wf(ni)
        print("rank_value:",rank_value)
        job_v.append(ni)
        rank_v.append(rank_value)
        return rank_value


def cwbar(ni, ps):
    """ Average computation cost """
    return sum(p for p in ps[ni]) / len(ps[ni])

def ccbar(ni, nj, c, ps):
    """ Average communication cost """
    n = len(ps[ni])
    comsum = 0
    if n == 1:
        return 0
    npairs = n * (n-1)
    return 1. * sum(c[ni][nj] for a1 in range(0,len(ps[ni])) for a2 in range(0,len(ps[nj])) 
                                        if a1 != a2 and c[ni][nj] != 99) / npairs


#深度相关的任务调度的任务优先级排序
def cranku(ni, c, crps):
    crank = partial(cranku, c=c, crps=crps)
    wf = partial(cwbar, ps=crps)
    cf = partial(ccbar, c=c, ps=crps)
    rank_value = 0

    if len(pre(c, ni)):    
        rank_value = wf(ni) + max(cf(ni, nj) + crank(nj) for nj in pre(c, ni) )
        print("task prior ->", ni)
        print("rank_value:",rank_value)
        if ni not in job_v:
            job_v.append(ni)    
            rank_v.append(rank_value)
        return rank_value 
    else:
        print("task prior ->", ni)
        rank_value = wf(ni)
        print("rank_value:", rank_value)
        if ni not in job_v:
            job_v.append(ni)
            rank_v.append(rank_value)
        return rank_value    

def HEFT_schedule(jobs, w):
    rank = partial(ranku, ps=w)
    print("initial jobs:", jobs)
    sort_jobs = sorted(jobs, key=rank)
    print("sort jobs:",sort_jobs) 
    fig_HEFT_bar(sort_jobs, rank_v) 
    print("job_v:", job_v)
    print("rank_v:", rank_v)       

#CHEFT 深度相关的任务调度
def CHEFT_schedule(jobs, cg, wg):
    rank = partial(cranku, c=cg, crps=wg)
    flag = 0
    for i in jobs:
        for j in wg[i]:
            if jobs == 99:
                flag += 1
        if flag == len(jobs):
            del jobs[i]
    print("initial jobs:", jobs)
    sort_jobs = sorted(jobs, key=rank)
    print("sort jobs:",sort_jobs) 
    fig_HEFT_bar(sort_jobs, rank_v) 
    print("job_v:", job_v)
    print("rank_v:", rank_v)

def fig_FCFS_bar(x, y):
    fig = plt.Figure()
    #plt.ylim(0.0,1.0)
    plt.bar(x, y)
    #plt.plot(x, y)
    plt.xlabel('task number')
    plt.ylabel('cost ')
    plt.show()

def fig_FCFS(x, y):
    fig = plt.Figure()
    #plt.xlim(0,500)
    #plt.ylim(0.0,1.0)
    plt.plot(x, y)
    plt.xlabel('task number')
    plt.ylabel('cost ')
    plt.show()                  

def fig_HEFT_bar(x, y):
    fig = plt.Figure()
    #plt.ylim(0.0,1.0)
    plt.bar(x, y)
    #plt.plot(x, y)
    plt.xlabel('task number')
    plt.ylabel('priority ')
    plt.show()

def unfairness(Gm, G):
    pass

#DAG调度完成时间
def Makespan(order, t, c, w):
    """ Finish time of last job """
        
    '''
        order: 任务序列
        t:  特定任务i
        c:  该任务序列的通信开销矩阵
        w:  该任务序列的处理器开销矩阵
    
        调用完成时间 = 前一个任务的结束时间 + 任务的执行时间
        还需要改善
    '''
    #任务最早开始时间
    EST = 0
    #任务最晚结束时间
    EFT = 0 
    T = dict()
    for i in range(len(order)):
        EFT = EST + max(j for j in w[order[i]]) 
        if (i == t):
            T[i] = (EST, EFT)
            return T
        elif(c[i][i+1] != 99 ):
            EFT = EFT + c[i][i+1]
            EST = EFT
            T[i] = (EST, EFT)
        #T.append((EST, EFT))

def average_Cost(set, i, c_g, w_g):
    '''
        未完成
        关于sortlist内部的字典存储问题
        #调度算法调度长度/总的调度长度
        #平均调度长度比貌似只有在多DAG调度中才有意义

    '''
    i_slen = 0
    fin_slen = 0
    average_s = 0

    t = Makespan(set,i, c_g, w_g)
    for i in t.keys():
        if(i == t):
            i_slen = t[i][1]
        fin_slen = t[i][1]
    print("s_time:",i_slen, fin_slen)

    average_s = i_slen/fin_slen
    print("average schedule length:",average_s)
    return average_s
'''
待完成：
    本代码中没有考虑通信开销，在处理器选择上没有考虑在同一个处理器的情况。
'''
