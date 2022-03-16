def succ(c, node_id):
    succ_list = []
    for i in range(len(c[node_id])):
            if (c[node_id][i]!= 0 and c[node_id][i]!= 99):
                    succ_list.append(i)
    return succ_list


def ngenerate_c(list, num, layer):
    list_layer = []
    for  i in range(len(list)):
        list_layer.append(0)

    nodenum = int((len(list)-2)/(layer-2))
    id = 1
    for i in range(0, layer):
        if i == 0:
            list_layer[0] = i
        elif i == layer-1:
            list_layer[-1] = i
        else:
            for j in range(id, id + nodenum):
                list_layer[j] = i
            id += nodenum
    #避免在分层时，有些结点没有赋值层数
    for k in range(0, len(list_layer)):
        if k != 0 and list_layer[k] == 0:
            list_layer[k] = list_layer[k-1] 
    print("layer", list_layer)
    c = []
    # 创建一个几行几列的零矩阵
    for x in range(0, num):
        sublist = []
        c.append(sublist)
        for y in range(0, num):
            sublist.append(0)

    for numi in range(0, (layer-1)):
        for index in range(len(list_layer)):
            if list_layer[index] == numi:
                for layindex in range(len(list_layer)):
                    if list_layer[layindex] == numi + 1:
                        c[index][layindex] = random.randint(10,20)
                    elif index != list_layer:
                        c[index][layindex] = 99
    for i in range(len(c)):
        if i == len(c) -1:
            for j in range(len(c[i])):
                c[i][j] = 99
        for j in range(len(c[i])):
            if i== j:
                c[i][j] = 0

    print("c:", c)

    return c

def short_generate_w(list, num):
    w = []
    # 创建一个几行几列的零矩阵
    for x in range(num):
        sublist = []
        w.append(sublist)
        for y in range(3):
            sublist.append(0)

    for i in range(len(list)):
        for j in range(len(w[i])):
            #显示短DAG更长的时间
            w[i][j] = random.randint(20, 40)
    return w

def short_generate_g(num):
    t_list = []
    for i in range(num):
        t_list.append(i)
        
    # for i in range(0,num):
    #    t_list.append(random.randint(0,num))
    print(t_list)
    t_c = ngenerate_c(t_list, num, layer= 3)
    t_w = short_generate_w(t_list, num)

    return t_list, t_c, t_w

# 设置任务个数
t_num = random.randint(5, 8)
t_list, t_c, t_w = short_generate_g(t_num)