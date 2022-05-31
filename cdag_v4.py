import numpy as np
import math 
import matplotlib.pyplot as plt
from functools import partial
import copy
import random
from itertools import chain

# import DAG model
'''
#尝试用oop思想来构造DAG，过程复杂
    
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
    [8, 10, 12]]

#处理器选择
x = [[1, 1, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1]]

x1 = [[0, 1, 1, 0, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 1, 1, 0]]

#通信开销矩阵

c = [[0, 18, 9, 14, 99, 99 , 99],
    [99, 0, 99, 99, 16, 99, 99],
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

#调度时间长度
T = dict()

def max_depth():
    #遍历图，找到每一个最后结点，比较出最大的depth
    pass

#仅求直接后继
def succ(c, node_id):
    succ_list = []
    for i in range(len(c[node_id])):
            if (c[node_id][i]!= 0 and c[node_id][i]!= 99):
                    succ_list.append(i)
    return succ_list

#仅求直接前驱
def pre(c, node_id):
    pre_list = []
    for i in range(len(c)):
            if (c[i][node_id]!= 0 and c[i][node_id]!= 99):
                    pre_list.append(i)
    return pre_list

#用于计算排好序中的任务节点的前驱

all_pre = []

def preb(list, c, node_post, task_order, node):
  
    prelist = []
    thrownlist = []

    #只有第一次迭代的是完整的路径
    for i in range(len(c)):
    # if c[i][node_id] != 0 and c[i][node_id] != 99 and i in list[:node_id] and i != preblist[:][1]:  
        if  c[i][node] != 0 and c[i][node] != 99:           
            if  i in list[:node_post]:
                    #向上追溯        
                    task_order.append((i,node))
                    prelist.append(i)
                    tasko = copy.deepcopy(task_order)
                    preb(list, c, node_post, task_order, i)
                    #向下追溯
                    #在每个分支时，将前面pre的顺序保存下来
                    del_front(tasko)
                    exe_branch(task_order, c, tasko, prelist, node)
            else:
                prelist.append(i)
                thrownlist.append(i)
            


    return task_order, thrownlist
   
def preb_1(list, c, node_id, task_order, node):
    
    prelist = []
    thrownlist = []
    '''
    if  c[:][node] == 0 or c[:][node] == 99 :           
            task_order.append((0,node))
            #return task_order
    '''
    for i in range(len(c)):  
        if  c[i][node] != 0 and c[i][node] != 99:           
            task_order.append((i,node))
            prelist.append(i)
            tasko = copy.deepcopy(task_order)
            #prebcost = prebcost + c[i][node_id]
            preb_1(list, c, node_id, task_order, i) 
            #向下追溯
            #在每个分支时，将前面pre的顺序保存下来
            del_front(tasko)
            exe_branch(task_order, c, tasko, prelist, node) 
        #遇到是前驱但不在已执行结点队列中，该前驱结点依然需要记下来    

    return task_order

def del_front(tasko):
    '''
        在该点位置上保留下未递归完成的若干个后继结点
    '''
    for t in range(0, len(tasko)):
        if t < len(tasko) and tasko[t][0] == 0:
            del tasko[:t+1]
            t = 0
    

def exe_branch(task_order, c, tasko, prelist, node):
    '''
        在寻找前继的递归过程中，当遇到该点后的某一前继分支递归结束，回到该点，
        这时重新建立另一条前继路径，需要储存该点之前的若递归过的后继
    '''
    #约束应该加上所有的pre是否都遍历了
    prel = pre(c, node)
    if len(prel) > 1:
        f = 0
        for i in prel:
            if i in prelist:
                f = f + 1    
        if f != len(prel):
            #记下之前的结束位置,递归中前面几个符合的结点先加入进去
            for e in tasko:
                #该结点还有其他前继
                if e[1] != node :
                    task_order.append(e)
                else:
                    break  
    


def preb_2(list, c, node_id, task_order, node):
    '''
        version 3
    '''
    #print("结点:", node,"正在处理...")
    #tasko = []
    prelist = []

    for i in range(len(c)):
        if  c[i][node] != 0 and c[i][node] != 99:           
            if  i in list[:node_id]:
                    task_order.append((i,node))
                    prelist.append(i)
                    #print("添加结点进入taskorder：", task_order)
                    tasko = copy.deepcopy(task_order)
                    preb_2(list, c, node_id, task_order, i)
                    #在每个分支时，将前面pre的顺序保存下来
                    #print("分支判断的结点：", node)
                    #print("tasko:", tasko)
                    #处理一下tasko                 
                    for t in range(len(tasko)):
                        if t < len(tasko) and tasko[t][0] == 0:
                            del tasko[:t+1]
                            t = 0
                    #约束应该加上所有的pre是否都遍历了
                    prel = pre(c, node)
                    if len(prel) >1:
                        f = 0
                        for i in prel:
                            if i in prelist:
                                f = f + 1
                        if f == len(prel):
                            pass 
                        else:
                            #记下之前的结束位置
                            for e in tasko:
                                if e[1] != node :
                                    task_order.append(e)
                                else:
                                    break  
                    #print("通过分支判断后taskorder：", task_order) 
        
    return task_order    

#获取结点ni的深度
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
    #若考虑通信开销，设通信开销发生故障的概率也符合泊松分布
    #nerror = math.exp(-lamda *(w[n_i][p_u]+c[n_i][])


'''
    合成两个任务图
'''
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
             
    return com_c, com, com_w, com_c_len

'''
    根据深度分割任务图
'''
def cut_comG(task, loc, dep, c, w):
    #合成图的深度矩阵重新分为两个图的深度矩阵
    task1 = task[:loc]
    task2 = task[loc:]
    longlist = 0  if len(task1) > len(task2) else 1
    dep1 = dep[:loc]
    dep2 = dep[loc:]
    print("tasks and depths:", task1, task2, dep1, dep2)
    max_d1 = max(dep1)
    max_d2 = max(dep2)
    max_d =  max_d1 if max_d1<max_d2 else max_d2
    print("max depth:",max_d1, max_d2)
    
    #条件队列
    con_tasks1 = []
    con_tasks2 = []
    ucon_tasks = []

    for i in task:
        if dep[i] <= max_d and i in task1:
            #创建一个子图满足深度的条件队列
            con_tasks1.append(i)
        elif dep[i] <= max_d and i in task2:
            con_tasks2.append(i)
        else:
            #创建一个子图不满足深度的条件队列
            ucon_tasks.append(i)
    #根据两个列表创建通信矩阵
    con_tasks = con_tasks1 + con_tasks2

    print("con_task:", con_tasks1)
    print("con_task2:", con_tasks2)
    print(con_tasks)
    #tasks_len = len(con_tasks1) + len(con_tasks2)

    comcom = copy.deepcopy(c)
    ucomcom = copy.deepcopy(c)

    comp = copy.deepcopy(w)
    ucomp = copy.deepcopy(w)

    #com = [[0 for i in range(tasks_len)] for i in range(tasks_len)] 
    #uncom = [[0 for i in range(len(ucon_tasks))] for i in range(len(ucon_tasks))]
    
    #满足深度的条件队列的通信开销矩阵
  
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
            for i in range(len(ucomcom[t])):
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
    print("comp:")
    for i in comp:
        print(i[:])
        print('\n')
    print("ucomp:")    
    for i in ucomp:
        print(i[:])
        print('\n')
    

    return con_tasks, ucon_tasks, comcom, ucomcom, comp, ucomp, longlist
    

def FCFS_schedule(set):
    #先根据依赖关系排好序
    task_set = set

    #最早开始时间
    EST = 0
    #最晚结束时间
    EFT = 0 

    per_task = task_set
    per_task_all_cost = [] 
    sum_cost = 0 
    task_cost_sum = [] 
    #if (len(task_set) != 0 ):
    #for i in range(len(self.task_pool)):
    for i in range(len(task_set)):          
        print("process:")
        max_cpu_cost = 0
        for k in w[task_set[i]]:        
            if ( k > max_cpu_cost):
                max_cpu_cost = k 
        if(i < len(task_set)-1):
            if(c[task_set[i]][task_set[i+1]]!= 0 and c[task_set[i]][task_set[i+1]]!= 99): 
                #任务i的通信消耗+处理器消耗                 
                cost = c[task_set[i]][task_set[i+1]] + max_cpu_cost
            else:
                cost = max_cpu_cost
        else:
            cost = max_cpu_cost
    
        sum_cost = cost + sum_cost    
        per_task_all_cost.append(cost)
        task_cost_sum.append(sum_cost)
    #处理器开销最大的
    #任务i的完成时间
    allcost = sum(per_task_all_cost[:5])
    print("all cost:",allcost)
    #绘制cost，task线图
    fig_FCFS_bar(per_task, per_task_all_cost)
    fig_FCFS(per_task, task_cost_sum)
    #print("task all cost:", task_all_cost)


def FCFS_schedule_1(set):
    '''
        添加了处理器选择，判断是否前后两个任务是否在同一处理器上
    '''
    #先根据依赖关系排好序
    task_set = set

    

    #最早开始时间
    EST = 0
    #最晚结束时间
    EFT = 0 

    per_task = task_set
    per_task_all_cost = [] 
    sum_cost = 0 
    task_cost_sum = [] 
    #if (len(task_set) != 0 ):
    #for i in range(len(self.task_pool)):
    for i in range(len(task_set)):          
        print("process:")
        #随机选择一个处理器
        k = random.randint(0,len(w[task_set[i]])-1) 
        print("k:", k)
        if(i < len(task_set)-1):
            if(c[task_set[i]][task_set[i+1]]!= 0 and c[task_set[i]][task_set[i+1]]!= 99 and x[task_set[i]][task_set[i+1]] == 1): 
                #任务i的通信消耗+处理器消耗  
                print("更换处理器", task_set[i],"到", task_set[i+1]) 
                #当选择不同处理器时，通过左移或右移一个处理器
                if k < len(w[task_set[i+1]])-1:          
                    cost = c[task_set[i]][task_set[i+1]]*x[task_set[i]][task_set[i+1]] + w[task_set[i+1]][k+1]
                else:
                    cost = c[task_set[i]][task_set[i+1]]*x[i][i+1] + w[task_set[i+1]][k-1]
                print(cost)
            else:
                print("无需更换处理器", task_set[i],"到", task_set[i+1])  
                cost = w[task_set[i+1]][k]
                print(cost)
        else:
            cost = w[task_set[i]][k]
    
        sum_cost = cost + sum_cost    
        per_task_all_cost.append(cost)
        task_cost_sum.append(sum_cost)
    #处理器开销最大的
    #任务i的完成时间
    allcost = sum(per_task_all_cost[:5])
    print("all cost:",allcost)
    #绘制cost，task线图
    fig_FCFS_bar(per_task, per_task_all_cost)
    fig_FCFS(per_task, task_cost_sum)
    #print("task all cost:", task_all_cost)


def random_Schedule(set, c, w):
    '''
        random number appear not only once
    '''
    seq = []
    #cost = 0
    step = 0
    while step < len(set):
        r = random.randint(0, len(set)-1)
        if r not in seq:
            seq.append(r)
            step += 1
            #if seq:
            #    cost += max(w[r]) + c[seq[-1]][r]
            #elif step ==1 :
            #    cost += max(w[r])
    print("initial list:", set)
    print("random seq:", seq)
    #print("cost:", cost)

    return seq

def FCFS_m_schedule(set, c, w):
    '''
    多DAG FCFS
        set： 任务列表
        c：通信开销矩阵
        w：处理器开销矩阵

        cost equal 调度时间 
    '''
    #先根据依赖关系排好序
    task_set = set

    FCFS_task = []
    per_task = task_set
    per_task_all_cost = [] 
    sum_cost = 0 
    task_cost_sum = [] 
    #if (len(task_set) != 0 ):
    #for i in range(len(self.task_pool)):
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
    #处理器开销最大的
    #任务i的完成时间
    allcost = sum(per_task_all_cost[:5])
    print("all cost:",allcost)
    #显示调度时间
    #Makerspan(set,c, w)
    #绘制cost，task线图
    fig_FCFS_bar(per_task, per_task_all_cost)
    fig_FCFS(per_task, task_cost_sum)
    #print("task all cost:", task_all_cost)


'''
#   将图中结点按前驱排序
#   只保证前驱在前，后继在后
'''
#sortlist：符合任务优先级的任务队列
sortlist = []
def SortList(set, set_g, nodei):
    '''
        set: initial set
        set_g: 该任务列表的通信开销矩阵
        nodei：指定节点

    '''
    set = set
    preset = pre(set_g, nodei)
    if (len(preset)):
        '''
        if(len(preset) == 1): 
            nodeid = pre(nodei)        
            SortList(set, nodeid[0])
        else:
        '''    
        for i in preset:
            SortList(set,set_g, i)
            if i not in sortlist:
                '''
                #忘了什么意思了
                #for j in c[nodei]: 
                    #
                    #if c[nodei][i] < j:
                    sortlist.append(i)
                '''  
                sortlist.append(i)  
    else:
        if nodei not in sortlist:
            sortlist.append(nodei)
    if nodei not in sortlist:
          sortlist.append(nodei)                    

def wbar(ni, ps):
    """ Average computation cost """
    return sum(p for p in ps[ni]) / len(ps[ni])

def cbar(ni, nj, c, ps):
    """ Average communication cost """
    n = len(ps)
    comsum = 0
    if n == 1:
        return 0
    npairs = n * (n-1)
    print("npairs:",npairs)
    return 1. * sum(c[ni][nj] for a1 in range(0,len(ps[ni])) for a2 in range(0,len(ps[nj])) 
                                        if a1 != a2 and c[ni][nj] != 99) / npairs


job_v = []
rank_v = []
def ranku(ni, cc, ps):
    '''
        ni： 指定节点
    '''
    
    #偏函数，类似于java中的多态
    rank = partial(ranku, cc=cc, ps=ps)
    wf = partial(wbar, ps=ps)
    cf = partial(cbar,c =cc, ps=ps)
    rank_value = 0 

    if len(pre(cc, ni)):
    
        rank_value = wf(ni) + max(cf(ni, nj) + rank(nj) for nj in pre(cc, ni))
        print("task prior ->", ni)
        print("rank_value:",rank_value)
        job_v.append(ni)
        rank_v.append(rank_value)
        return rank_value
    else:
        print("/start//")
        print("task prior ->", ni)
        rank_value = wf(ni)
        print("rank_value:",rank_value)
        job_v.append(ni)
        rank_v.append(rank_value)
        return wf(ni)

cjob_v = []
crank_v = []

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
    #a1与a2表示处理器的选择
    return 1. * sum(c[ni][nj] for a1 in range(0,len(ps[ni])) for a2 in range(0,len(ps[nj])) 
                                        if a1 != a2 and c[ni][nj] != 99) / npairs


#深度相关的任务调度的任务优先级排序
def cranku(ni, c, crps):
    '''
        ni： 指定节点
    '''
    
    #偏函数，类似于java中的多态
    crank = partial(cranku, c=c, crps=crps)
    wf = partial(cwbar, ps=crps)
    cf = partial(ccbar, c=c, ps=crps)
    rank_value = 0

    if len(pre(c, ni)):    
        rank_value = wf(ni) + max(cf(ni, nj) + crank(nj) for nj in pre(c, ni) )
        print("task prior ->", ni)
        print("rank_value:",rank_value)
        if ni not in cjob_v:
            cjob_v.append(ni)    
            crank_v.append(rank_value)
        return rank_value 
    else:
        print("/start/")
        print("task prior ->", ni)
        rank_value = wf(ni)
        print("rank_value:", rank_value)
        if ni not in cjob_v:
            cjob_v.append(ni)
            crank_v.append(rank_value)
        return rank_value    

def HEFT_schedule(jobs, c, w):
    rank = partial(ranku, cc=c, ps=w)
    print("initial jobs:", jobs)
    sort_jobs = sorted(jobs, key=rank)
    #sort_jobs.reverse()
    print("heft sort jobs:",sort_jobs)
    fig_HEFT_bar(job_v, rank_v) 
    print("job_v:", job_v)
    print("rank_v:", rank_v)  
    return sort_jobs     

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
    print("cheft sort jobs:",sort_jobs)
    #fig_HEFT_bar(sort_jobs, crank_v) 
    print("job_v:", cjob_v)
    print("rank_v:", crank_v)
    return sort_jobs

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
    if x and y:
        plt.bar(x, y)
    #plt.plot(x, y)
    plt.xlabel('task number')
    plt.ylabel('priority ')
    plt.show()

def eft_gent(eft_time):
 
    #处理eft_time 
    eft = []
    #temp相当于一个栈
    temp = []
    #用来判断初次序号不同的标记
    flag = 0
    for i in range(len(eft_time)):
        if i != 0 and i < len(eft_time):
            if eft_time[i-1][0] == eft_time[i][0]:
                if eft_time[i-1] in eft:
                    flag = 1
                temp.append(eft_time[i])               
            elif eft_time[i-1][0] != eft_time[i][0] and temp:
                max = 0
                #选择路径最长的作为该结点调度长度
                for j in range(len(temp)):
                    if j == 0:
                        max = temp[j][1]
                    if temp[j-1][1] > temp[j][1]:
                        max = temp[j-1][1]
                    else:
                        max = temp[j][1]
                if flag == 0:
                    eft.append((eft_time[i-1][0], max))
                eft.append(eft_time[i])
                del temp[:]
                flag = 0
            elif eft_time[i-1][0] != eft_time[i][0]:
                eft.append(eft_time[i])
        elif i == 0:
            if (eft_time[i][0] == eft_time[i+1][0]):
                temp.append(eft_time[i])
            else:
                eft.append(eft_time[i])
    return eft

#公平性
def Speedrate(node_i, Gm, gm_c, gm_w, G, g_c, g_w):
    
    speedrate = 0

    time1 = 0
    time1_1 = 0
    time2 = 0
    time2_1 = 0
    
    eft_time1, eft_time1_1 = Makerspan(Gm, gm_c, gm_w)
    eft_time2, eft_time2_1 = Makerspan(G, g_c, g_w)

    #print("eft_time1:", eft_time1)
    #print("eft_time1_1:", eft_time1_1)
    #print("eft_time2:", eft_time2)
    #print("eft_time2_1:", eft_time2_1)

    eft1 = eft_gent(eft_time1)
    eft1_1 = eft_gent(eft_time1_1)
    eft2 = eft_gent(eft_time2)
    eft2_1 = eft_gent(eft_time2_1)

    #print("eft1:", eft1)
    #print("eft1_1:", eft1_1)
    #print("eft2:", eft2)
    #print("eft2_1:", eft2_1)

    #需要按list顺序执行
    for x in eft1:
        if x[0] == node_i:
            break
        time1 = time1 + x[1]
    for x1 in eft1_1:
        if x1[0] == node_i:
            break
        time1_1 = time1_1 + x1[1]
    for y in eft2:
        if y[0] == node_i:
            break
        time2 = time2 + y[1]
    for y1 in eft2_1:
        if y1[0] == node_i:
            break
        time2_1 = time2_1 + y1[1]

    #print("time1:", time1)
    #print("time2:", time2)
    #slowdown = t1_eft[-1] / t2_eft[-1]

    time1_diff = time1_1 - time1
    time2_diff = time2_1 - time2
    
    if time2 != 0:
        speedrate = time1  / time2 
    
    print("node:", node_i)
    print("speedrate:", speedrate)
    return speedrate

def Speedrate_1(node_i, Gm, gm_c, gm_w, G, g_c, g_w):
    '''
        一般算法使用
        
        思想:对一个DAG生成的执行队列计算调度长度，该队列的每个节点都根据前驱计算调度长度，然后计算总和。
            队列中有前驱相关性的，可以在计算后减去那部分调度长度。
    '''
    speedrate = 0
    
    time1 = 0
    time1_1 = 0
    time2 = 0
    time2_2 = 0

    #t1_st, t1_eft = Makespan(Gm, node_i, gm_c, gm_w)
    #t2_st, t2_eft = Makespan(G, node_i, g_c, g_w)
    
    eft_time1, eft_time1_1 = Makerspan_1(Gm, gm_c, gm_w)
    eft_time2, eft_time2_2 = Makerspan_1(G, g_c, g_w)

    #print("slowndown(eft_time1:", eft_time1)
    #print("slowndown(eft_time1_1:", eft_time1_1)
    #print("slowndown(eft_time2:", eft_time2)
    #print("slowndown(eft_time2_2:", eft_time2_2)

    eft1 = eft_gent(eft_time1)
    eft1_1 = eft_gent(eft_time1_1)
    eft2 = eft_gent(eft_time2)
    eft2_2 = eft_gent(eft_time2_2)

    #print("eft1:", eft1)
    #print("eft1:", eft1_1)
    #print("eft2:", eft2)
    #print("eft2:", eft2_2)

    for x in eft1:
        if x[0] == node_i:
            break
        time1 = time1 + x[1]
    for x1 in eft1_1:
        if x1[0] == node_i:
            break
        time1_1 = time1_1 + x1[1]
    for y in eft2:
        if y[0] == node_i:
            break
        time2 = time2 + y[1]
    for y1 in eft2_2:
        if y1[0] == node_i:
            break
        time2_2 = time2_2 + y1[1]

    time1 = time1_1 - time1
    time2 = time2_2 - time2
    #print("time1:", time1)
    #print("time2:", time2)
    
    if time2 != 0:
        speedrate = time1 / time2
    print("node:", node_i)
    print("speedrate:", speedrate)
    return speedrate

def Makerspan(order, c, w):
   
    #最晚执行时间的中间结果
    eft_median = []
    #最晚任务执行时间
    eft_time = []
    
    #最晚执行时间的中间结果
    eft_median1 = []
    #最晚任务执行时间
    eft_time1 = []

    for i in range(len(order)):
        task_order = []
        task_order1 = []
        costsum = 0
        #求出任务前驱
        #print("preb入口结点：", order[i])
        task_order, thrownlist = preb(order, c, i, task_order, order[i])
        #求出理想的前驱集合，没有约束条件，集合长度>=task_ord
        #这里好像有问题，第四个参数,改了
        task_order1 = preb_1(order, c, i, task_order1, order[i])
        
        task_ord = [x[0] for x in task_order ]
        task_ord = list(task_ord)
        #包含执行任务节点的整个前驱列表，可以用来直接与通信开销矩阵使用
        task_ord.insert(0, order[i])
        #为每个路径添加头结点      
        for j in range(len(task_ord)):
            if j<len(task_ord)-1 and task_ord[j] == 0:
                task_ord.insert(j+1, order[i])
        
        task_ord1 = [x[0] for x in task_order1 ]
        task_ord1 = list(task_ord1)
        task_ord1.insert(0, order[i])

        for j1 in range(len(task_ord1)):
            if j1<len(task_ord1)-1 and task_ord1[j1] == 0:
                task_ord1.insert(j1+1, order[i])

        print("t_order:", task_order)
        print("order:", task_ord) 
        print("thrown:", thrownlist)       
        print("order1:", task_ord1) 

        front = 0
        rear = 0

        #排除不符合格式的路径
        #例如会有的路径没有遍历完
        errorloc = []
        for t1 in range(0, len(task_ord)):
            if task_ord[t1] == order[i]:
                errorloc.append(t1)
                for t2 in range(t1, len(task_ord)):
                    if task_ord[t2] == 0:
                        break
                    elif task_ord[t2] == order[i]:
                        errorloc.append(t2)
                        break
        print("errorloc：", errorloc)
        
        if len(task_ord) > 1:
            for k in range(0, len(task_ord)):
                if k<len(task_ord) and task_ord[k] == order[i]:
                    front = k
                    rear = front
                    for k1 in range(front, len(task_ord)):
                        if task_ord[k1] == 0:
                            rear = k1
                            break      
                    for l in task_ord[front:rear]:
                        if l in thrownlist:
                                del task_ord[front:rear]
                                break                                   
        '''
        #将一个列表按指定值所在的位置划分为多个列表
        posi = []
        for i in task_order:
            if task_ord[i] == 0:
               posi.append(i)
        for i in range(len(posi)):
            task_i = []
            task_i = task_ord[:i]
            print(task_i)
        '''
        print("changed order:", task_ord)        
        print("order1:", task_ord1) 

        task_ord, eft_median, eft_time= divid_task_ord(i, task_ord, task_ord1, eft_median, eft_time, order, c, w)
        task_ord1, eft_median1, eft_time1 = divid_task_ord(i, task_ord1, task_ord1, eft_median1, eft_time1, order, c, w)


    #print("eft median:", eft_median)
    #print("eft time:", eft_time)
    #print("eft median1:", eft_median1)
    #print("eft time1:", eft_time1)
    
    return eft_time, eft_time1

def Makerspan_1(order, c, w):
    
    #最晚执行时间的中间结果
    eft_median = []
    #最晚任务执行时间
    eft_time = []

    #最晚执行时间的中间结果
    eft_median1 = []
    #最晚任务执行时间
    eft_time1 = []

    for i in range(len(order)):
        task_order = []
        
        #求出任务前驱
        task_order, thrownlist = preb(order, c, i, task_order, order[i])
        #求出理想的前驱集合，没有约束条件，集合长度>=task_ord
        task_order1 = preb_1(order, c, i, task_order, order[i])
        #利用前驱，计算时间
         
        task_ord = [x[0] for x in task_order ]
        #task_ord表示该结点符合执行顺序的前继集合
        task_ord = list(task_ord)
        #print("qian task_ord:", task_ord)
        #包含执行任务节点的整个前驱列表，可以用来直接与通信开销矩阵使用
        task_ord.insert(0, order[i])
        #每条路径的结尾，就是另一条路径的开端，每个路径都缺少头结点
        for j in range(len(task_ord)-1):
            if j<len(task_ord)-1 and task_ord[j] == 0:
                task_ord.insert(j+1, order[i])
        
        task_ord1 = [x[0] for x in task_order1 ]
        task_ord1 = list(task_ord1)
        task_ord1.insert(0, order[i])

        for j1 in range(len(task_ord1)):
            if j1<len(task_ord1)-1 and task_ord1[j1] == 0:
                task_ord1.insert(j1+1, order[i])

        #print("t_order:", task_order)
        #print("order:", task_ord)
        #print("thrown:", thrownlist)
        #print("t_order1:", task_order)
        #print("order1:", task_ord1)

        front = 0
        rear = 0
        #排除不符合的路径
        #list中0在不同DAG，代表不同含义，在合成DAG中0表示头结点。
        if len(task_ord) > 1:
            for k in range(0, len(task_ord)):
                if k<len(task_ord) and task_ord[k] == order[i]:
                    front = k
                    rear = front
                    for k1 in range(front, len(task_ord)):
                        if task_ord[k1] == 0:
                            rear = k1
                            break      
                    for l in task_ord[front:rear]:
                        if l in thrownlist:
                                del task_ord[front:rear]
                                break                                   
          
        #print("t_order:", task_order)
        #print("order:", task_ord)
        #print("thrown:", thrownlist)
        #print("t_order1:", task_order1)
        #print("order1:", task_ord1)

        #计算受约束任务调度长度
        #i1, task_ord1, eft_median1, eft_time1, order1, c, w = divid_task_ord(i, task_ord, eft_median, eft_time, order, c, w, costsum)
        #i1, task_ord1_1, eft_median_1, eft_time1_1, order1, c, w  = divid_task_ord(i, task_ord1, eft_median1, eft_time1, order, c, w, costsum)

        i, task_ord, eft_median, eft_time, order, c, w = divid_task_ord_1(i, task_ord, eft_median, eft_time, order, c, w)
        i, task_ord1, eft_median1, eft_time1, order, c, w  = divid_task_ord_1(i, task_ord1, eft_median1, eft_time1, order, c, w)
    
    #print("eft median:", eft_median)
    #print("eft time:", eft_time)
    #print("eft median1:", eft_median1)
    #print("eft time1:", eft_time1)
    
    return eft_time, eft_time1

def divid_task_ord(i, task_ord, task_ord1, eft_median, eft_time, order, c, w):
    
    #直接不考虑切分，只是不考虑0的情况，就其他算法而言，不考虑0是因为0是作为分隔作用
    costsum = 0
    eft_m = []
    if len(task_ord) > 1:
        for j in range(0, len(task_ord)):
            #print("index:", j)
            if(j < len(task_ord) and task_ord[j] != 0):
            #选择最大消耗的处理器
                costsum = costsum + max(w[task_ord[j]]) + c[task_ord[j+1]][task_ord[j]]
                eft_median.append(costsum)
                eft_m.append(costsum)
                #eft_time.append((order[i], eft_m[-1]))
                #print("add cost:", costsum)
            elif(j < len(task_ord) and task_ord[j] == 0):
                if eft_median:
                    costsum = costsum + max(w[task_ord[j]])
                    eft_median.append(costsum)
                    eft_m.append(costsum)
                    eft_time.append((order[i], eft_m[-1]))
                    eft_median.append(0)
                    del eft_m[:]
                #如果是0的操作            
                costsum = 0
            elif(j == len(task_ord)):
                if eft_median:
                    costsum = costsum + max(w[task_ord[j]])
                    eft_median.append(costsum)
                    eft_m.append(costsum)
                    eft_time.append((order[i], eft_m[-1]))
                    eft_median.append(0)
                    del eft_m[:]
                    costsum = 0
                else:
                    costsum = costsum + max(w[task_ord[j]])
                    eft_time.append((order[i], eft_m[-1]))
                    costsum = 0
        
    elif len(task_ord) == 1 and task_ord[0] == 0:  
        costsum = costsum + max(w[task_ord[0]])
        #print("add cost:", costsum)
        eft_median.append(costsum)
        eft_m.append(costsum)       
        eft_time.append((order[i], eft_m[-1]))
        del eft_m[:]
        #eft_median.append(0)
        costsum = 0
    
    elif len(task_ord) == 1 and task_ord[0] != 0:
        for k in range(0, len(task_ord1)-1):
            if(k < len(task_ord1) and task_ord1[k] != 0):
                #选择最大消耗的处理器
                    costsum = costsum + max(w[task_ord1[k]]) + c[task_ord1[k+1]][task_ord1[k]]
                    eft_median.append(costsum)
                    eft_m.append(costsum)
                    #eft_time.append((order[i], eft_m[-1]))
                    #print("add cost:", costsum)
            elif(k < len(task_ord1) and task_ord1[k] == 0):
                if eft_median:
                    costsum = costsum + max(w[task_ord1[k]])
                    eft_median.append(costsum)
                    eft_m.append(costsum)
                    eft_time.append((order[i], eft_m[-1]))
                    eft_median.append(0)
                    del eft_m[:]
                #如果是0的操作            
                costsum = 0


    #print("eft_time:", eft_time)
    return task_ord, eft_median, eft_time

#考虑处理器选择
def divid_task_ord_1(i, task_ord, eft_median, eft_time, order, c, w ):
    #直接不考虑切分，只是不考虑0的情况，就其他算法而言，不考虑0是因为0是作为分隔作用
    costsum = 0
    k = random.randint(0, len(w[order[i]])-1) 
    if len(task_ord) > 1:
        for j in range(0, len(task_ord)-1):
            #print("index:", j)
            if(task_ord[j] != 0):
            #选择处理器
                #这个约束不能体现处理器不能选择同一个
                if k < len(w[order[i]])-1:
                    costsum = costsum + w[task_ord[j]][k+1] + c[task_ord[j+1]][task_ord[j]]
                    #print("add cost:", costsum)
                else:
                    costsum = costsum + w[task_ord[j]][k-1] + c[task_ord[j+1]][task_ord[j]]
                    #print("add cost:", costsum)
                eft_median.append(costsum)
            elif(task_ord[j] == 0):
                if eft_median:
                    eft_time.append((order[i], eft_median[-1]))
                #如果是0的操作            
                eft_median.append(0)
                costsum = 0
    elif len(task_ord) == 1:  
        costsum = costsum + w[task_ord[0]][k]
        #print("add cost:", costsum)
        eft_median.append(costsum)       
        eft_time.append((order[i], eft_median[-1]))
        eft_median.append(0)
        

    return i, task_ord, eft_median, eft_time, order, c, w 
  
#DAG调度完成时间
def Makespan(order, t, c, w):
    """ Finish time of last job """
        

    #任务开始时间
    ST = 0
    #任务最晚结束时间
    EFT = 0 
    st_time = []
    eft_time = []
    for i in range(0, len(order)):
        EFT = ST + max(j for j in w[order[i]]) 
        if (order[i] == t):
            st_time.append(ST)
            eft_time.append(EFT)
            T[i] = (order[i] , ST, EFT)
            return st_time, eft_time
        elif(c[order[i]][order[i+1]] != 99):
            EFT = EFT + c[order[i]][order[i+1]]
            eft_time.append(EFT)
            ST = EFT
            st_time.append(EFT)
            T[i] = (order[i], ST, EFT)
        
        #T.append((ST, FT))
    return  st_time, eft_time

#提示：单DAG/多DAG
def SlowdownRate(mlist, m_c, m_w):
  
    slowdown = 0

    slist = []

    time1 = 0
    time2 = 0

    eft_time2, eft_time2_1 = Makerspan(mlist, m_c, m_w)

    del sortlist[:]
    #if longls == 0:
    for i in range(0, len(task_set1)):
        SortList(task_set1, c1, task_set1[i])
    slist = copy.deepcopy(sortlist)
    
    #让sortlist恢复原状，为后面合成DAG使用
    #del sortlist[:]

    eft_time1, eft_time1_1 = Makerspan(slist, c1, w1)


    #print("eft_time1：", eft_time1)
    #print("eft_time2：", eft_time2)

    eft1 = eft_gent(eft_time1)
    eft2 = eft_gent(eft_time2)

    print("eft_1：", eft1)
    print("eft_2：", eft2)

    for node in task_set1:
        for x in eft1:
            if x[0] == node:
                break
            time1 = time1 + x[1]
        for x in eft2:
            if x[0] == node:
                break
            time2 = time2 + x[1]
    
    print("node:", node)
    if time2 != 0:
        slowdown = time1 / time2
    
    print("slowndown:", slowdown)
    pass

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
博弈框架
       
slr:调度长度比
spu：公平性
                  slr                          spu             
        --------------------------------------------------------
slr     |      (         )         |       (          )        |
        |--------------------------|---------------------------|
spu     |      (         )         |       (          )        |
        --------------------------------------------------------
'''

def game_ss(task_1_slr, task_2_slr,task_1_spu, task_2_spu):
    
    leftTop = 0
    rightTop = 0
    leftDown = 0
    rightDown = 0

    if(task_1_slr > task_2_slr):
        leftTop = 1
    elif(task_1_slr < task_2_slr):
        leftTop = -1
        
    if(task_1_slr > task_2_spu): 
        rightTop = 1
    elif(task_1_slr < task_2_spu):
        rightTop = -1
    
    if(task_1_spu > task_2_slr): 
        leftDown = 1
    elif(task_1_spu < task_2_slr):
        leftDown = -1

    if(task_1_spu > task_2_slr): 
        rightDown = 1
    elif(task_1_spu < task_2_slr):
        rightDown = -1   
   


'''
待完成：
    本代码中没有考虑通信开销，在处理器选择上没有考虑在同一个处理器的情况。

    需要处理一下efttime的代码
    heft的节点的调度长度需要处理

    出现一个问题：heft与chef的排序是一样的
        未排除两个DAG左后的结点有差不多的开销
'''
if __name__ == '__main__':
    #单任务图
    #print(succ(task_set[2]))
    #print(pre(task_set[2]))
    '''
    for i in range(0, len(task_set)):
        SortList(task_set, c, task_set[i])
    print(sortlist)
    FCFS_schedule(sortlist)
    HEFT_schedule(sortlist, w)
    for i in sortlist:
        print(Makespan(sortlist, i, c, w))
    '''
    
    for i in range(0, len(task_set)):
        SortList(task_set, c, task_set[i])
    print(sortlist)
    FCFS_schedule(sortlist)
    FCFS_schedule_1(sortlist)
    
    del sortlist[:]
    #处理合成图
    c_list, c_g, w_g, dif_loc=compound_G(task_set,task_set1)

    print("compund graph:", c_g)
    print("compund w:", w_g)
    print("dif-loc:", dif_loc)
    if len(sortlist) == 0:
        for i in range(0, len(c_list)):
            SortList(c_list, c_g, c_list[i])
    print("sortlist:", sortlist)
    
    #for i in sortlist:
    #   print(Makespan(sortlist,i, c_g, w_g))
    #获取节点深度
    d = depth(sortlist, c_g)
    #c_tasks:满足深度
    #uc_tasks:不满足深度
    c_tasks, uc_tasks, c_com, uc_com, w_com, uw_com, longls = cut_comG(sortlist, dif_loc, d, c_g, w_g)
    #print(c_tasks)
 
    #schedule_1:随机调度
    randomtask = random_Schedule(sortlist, c_g, w_g)
    #schedule_2:FCFS调度
    FCFS_m_schedule(sortlist, c_g, w_g)
    #schedule_3:HEFT调度
    HEFT_tasks = HEFT_schedule(sortlist, c_g, w_g)
    #schedule_4:CHEFT调度
    CHEFTtask1 = CHEFT_schedule(c_tasks, c_com, w_com)
    CHEFTtask2 = CHEFT_schedule(uc_tasks, uc_com, uw_com)
    CHEFTtask = CHEFTtask1 + CHEFTtask2
    #指定节点，在使用调度的队列与为使用调度的队列中比较

    '''
        为什么只是传入一个c_com，本来要求是传入CHEFtask的通信开销矩阵，因为我要求得结点属于的DAG全部包含在了
        c_com，就可以利用现存的。
    '''
    '''
    print("FCFS schedule:")
    print("fcfstask:", sortlist)
    print("sorttask:", sortlist)
    s = 0
    epoch = 0
    for j in sortlist:
        if j != 0 and j in sortlist[8:]:
            epoch = epoch +1
            s = s + Speedrate(j, sortlist, c_g, w_g, sortlist, c_g, w_g)
    sd = s / epoch
    print("average speedrate:", sd)

    #还是有点问题
    print("select cpu version")
    s = 0
    epoch = 0
    j = 0
    for j in sortlist:
        if j != 0 and j in sortlist[8:]:
            epoch = epoch +1
            s = s + Speedrate(j, sortlist, c_g, w_g, sortlist, c_g, w_g)
    sd = s / epoch
    print("average speedrate:", sd)

    
    print("HEFT schedule:")
    print("HEFTtask:", HEFT_tasks)
    print("sorttask:", sortlist)
    s = 0
    epoch3 = 0
    j = 0
    for j in HEFT_tasks:
        if j != 0 and j in sortlist[8:]:
            epoch3 = epoch3 +1
            s = s + Speedrate(j, HEFT_tasks, c_g, w_g, sortlist, c_g, w_g)
    sd = s / epoch3 
    print("average speedrate:", sd)
    
    '''
    '''
        
        #random在使用slowdown时会报错
        #在于Cheft 所有的结点的前驱最终都会指向结点0，在preb中会适时的插入结点来区分不同路径
        解决random schedule是在解决什么？对于preb，是在解决迭代前继结点时，有些前继结点不在该结点的调度队列之前
    '''
    
    #经常出错
    print("Random schedule:")
    print("random task:", randomtask)
    print("sortlist:", sortlist)
    s2 = 0 
    epoch2 = 0
    j = 0
    for j in randomtask:
        if j != 0 and j in sortlist[dif_loc:]:
            epoch2 = epoch2 + 1
            s2 = s2 + Speedrate(j, randomtask, c_g, w_g, sortlist, c_g, w_g)
    sd2 = s2 / epoch2  
    print("average speedrate:", sd2)
    
    print("CHEFT schedule:")
    print("CHEFTtask:", CHEFTtask)
    print("sorttask:", sortlist)
    s1 = 0
    epoch1 = 0
    j = 0
    for j in CHEFTtask:
        if j != 0 and j in sortlist[dif_loc:]:
            epoch1 = epoch1 + 1 
            s1 = s1 + Speedrate(j, CHEFTtask, c_com, w_com, sortlist, c_g, w_g)
    sd1 = s1 / epoch1
    print("average speedrate:", sd1)
    
    SlowdownRate(sortlist, c_g, w_g)
    
    
