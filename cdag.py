import numpy
import math 
import matplotlib.pyplot as plt
from functools import partial
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


d = []

def max_depth():
    
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
    #若考虑通信开销，设通信开销发生故障的概率也符合泊松分布
    #nerror = math.exp(-lamda *(w[n_i][p_u]+c[n_i][])



def compound_G(taskset1, taskset2):
    
    com_c = [0]
    t1 = taskset1
    t2 = taskset2
    com = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    com_w = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

    #合成任务列表
    t1_len = len(t1)
    for i in range(0, t1_len):
        com_c.append(i+1)
    com_c_len = len(com_c)
    t2_len = len(t2)
    for i in range(0, t2_len):
        com_c.append(com_c_len+i)
    print("initial:", com_c)

    #创建一个合成图的通信开销矩阵
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
             
    return com_c, com, com_w

def FCFS_schedule(set):
    task_set = set
    EST = 0
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

def FCFS_m_schedule(set, c, w):
    task_set = set
    EST = 0
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


sortlist = []
def SortList(set, set_g, nodei):
    set = set
    preset = pre(set_g, nodei)
    if (len(preset)):
        for i in preset:
            SortList(set,set_g, i)
            if i not in sortlist:
                for j in c[nodei]: 
                    if c[nodei][i] < j:
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
    rank = partial(ranku)
    wf = partial(wbar)
    cf = partial(cbar)
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
       
    

def HEFT_schedule(jobs):
    rank = partial(ranku, ps=w)
    print("initial jobs:", jobs)
    sort_jobs = sorted(jobs, key=rank)
    print("sort jobs:",sort_jobs) 
    fig_HEFT_bar(sort_jobs, rank_v) 
    print(rank_v)       

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
    EST = 0
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
