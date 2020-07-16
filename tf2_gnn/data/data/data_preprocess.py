# F:\Github\CCS19\Curre\GNN\GraphNNCode\GraphNNCode\New\result\result\test
import gzip
import random
import warnings
import jsonlines
import jsonlines
from gensim.models import word2vec, Word2Vec
import re
import numpy as np
import os
import json
import sys
warnings.filterwarnings("ignore")

# 存储边信息
adj_zc = []

# cdfg边信息
adj_cdfg_a = []
adj_cdfg_c = []
adj_cdfg_p = []


# 获得AST或CDFG的图信息
def w2v(filename,cwetype):
    if os.path.exists(os.path.split(filename)[0]+"/" + "word2vec_" + str(cwetype) + ".pkl"):
        print("Word2Vec already !")
        return
    words = []
    for root, dirs, files in os.walk(filename):  # os.walk()返回一个三元组,
        # 随机打乱文件
        np.random.shuffle(files)
        for file in files:
            flag_vec = 0
            word = []
            with open(os.path.join(root + "/" + file), "r") as f:
                for line in f.readlines():
                    if "-" * 20 in line:
                        flag_vec = 1

                    if flag_vec == 1:  # 处理特征矩阵
                        if "^" * 20 in line:
                            for i in word:
                                words.append(i)
                            # continue
                        if re.search('(?<=,).*', line):
                            # print(a.group().split(')')[0])
                            # 去除空的情况
                            # if a.group().split(')')[0] != '':
                            word.append(re.search('(?<=,).*', line).group().split(')')[0].split(" "))  # 将token加入数组
    model = Word2Vec(words, min_count=1, size=100, sg=1, window=5,
                     negative=3, sample=0.001, hs=1, workers=4)

    # a=model.wv.vocab
    # for k in a:
    #     print(k)
    model.save(os.path.split(filename)[0]+"/"+ "word2vec_" + str(cwetype) + ".pkl")
    print("Word2Vec ready")

def GetInfor(filename):
    import gzip
    for root, dirs, files in os.walk(filename):  # os.walk()返回一个三元组,
        np.random.shuffle(files)
        filenum = len(files)
        count = 0
        save = 0
        for fname in files:
            # ast边信息
            adj_ast_total = []
            adj_child = []
            adj_com = []
            adj_next = []
            adj_guared = []
            adj_lexical = []
            adj_jump = []
            # cdfg边信息
            adj_cdfg_a = []
            adj_cdfg_c = []
            adj_cdfg_p = []
            # 针对每个文件的初始化
            label_label = False
            label_child = False  # child
            label_from = False  # compute_form
            label_next = False  # next_token
            label_by = False  # garde_by
            label_negation = False  # garde_negation
            label_att = False  # node_attribute
            label_use = False  # Lexical_use
            label_jump = False  # jump
            label_joern = False  # cdfg
            label_joern_word = False # joern_word
            count += 1
            joern_word = []
            with open(root +'/'+ fname, "r") as f:
                data = f.readlines()
                for line in data:
                    if (line.strip().find("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^") >= 0):  # 跳出文件处理
                        break
                    # if (line.find("-----ast_node-----")>=0):  #此处不处理ast 源码内容
                    #     break
                    nodes = line.split('\n')[0].split(',')
                    if (line.find("-----label-----") >= 0):
                        label_label = True
                        continue
                    if (label_label):
                        label_single = line.strip()
                        label_label = False
                        continue

                    if (line.find("-----children-----") >= 0):
                        label_child = True
                        continue
                    if (label_child):
                        if line.find("-----nextToken-----") >= 0:
                            label_child = False
                        else:
                            adj_child.append((int(nodes[0]), int(nodes[1])))

                    if (line.find("-----nextToken-----") >= 0):
                        label_next = True
                        continue
                    if (label_next):
                        if line.find("-----computeFrom-----") >= 0:
                            label_next = False
                        else:
                            # nodes = line.split('\n')[0].split(',')
                            for i in range(len(nodes)):
                                if i != len(nodes) - 1:
                                    adj_next.append((int(nodes[i]), int(nodes[i + 1])))

                    if (line.find("-----computeFrom-----") >= 0):
                        label_from = True
                        continue
                    if (label_from):
                        if line.find("-----guardedBy-----") >= 0:
                            label_from = False
                        else:
                            adj_com.append((int(nodes[0]), int(nodes[1])))

                    if (line.find("-----guardedBy-----") >= 0):
                        label_by = True
                        continue
                    if (label_by):
                        if line.find("-----guardedByNegation-----") >= 0:
                            label_by = False
                        else:
                            adj_guared.append((int(nodes[0]), int(nodes[1])))

                    if (line.find("-----guardedByNegation-----") >= 0):
                        label_negation = True
                        continue
                    if (label_negation):
                        if line.find("-----lastLexicalUse-----") >= 0:
                            label_negation = False
                        else:
                            adj_guared.append((int(nodes[0]) , int(nodes[1])))

                    if (line.find("-----lastLexicalUse-----") >= 0):
                        label_use = True
                        continue
                    if (label_use):
                        if line.find("-----jump-----") >= 0:
                            label_use = False
                        else:
                            adj_lexical.append((int(nodes[0]), int(nodes[1])))

                    if (line.find("-----jump-----") >= 0):
                        label_jump = True
                        continue
                    if (label_jump):
                        if line.find("-----attribute-----") >= 0:
                            label_jump = False
                        else:
                            adj_jump.append((int(nodes[0]), int(nodes[1])))

                    if (line.find("-----attribute-----") >= 0):
                        label_att = True
                        continue
                    if (label_att):
                        if line.find("-----ast_node-----") >= 0:
                            label_att = False
                        else:
                            num = line.split(';')
                            for x in num:
                                if (x == "" or x == '\n'):
                                    num.remove(x)
                            nodeNum = len(num)
                            label_adj = False
                            # print(nodeNum)
                    if (line.find("-----joern-----") >= 0):
                        label_joern = True
                        continue
                    if (label_joern):
                        if line.find("-----------------------------------") >= 0:
                            label_joern = False
                        else:
                            threedot = line.strip().split("(")[1].split(")")[0].split(",")  # 分割成三元组
                            # print(len(threedot))
                            # print(line)
                            try:
                                adj_single = (int(threedot[0]), int(threedot[1]))

                                if int(threedot[2]) == 0:
                                    adj_cdfg_a.append(adj_single)
                                    continue
                                if int(threedot[2]) == 1:
                                    adj_cdfg_c.append(adj_single)
                                    continue
                                if int(threedot[2]) == 2:
                                    adj_cdfg_p.append(adj_single)
                                    continue
                            except:
                                continue  # 大于100直接跳过
                    if line.find("-----------------------------------") >= 0:
                        label_joern_word = True
                        continue
                    if (label_joern_word):
                        if re.search('(?<=,).*', line):
                            joern_word.append(re.search('(?<=,).*', line).group().split(')')[0])


            # len(num) >= 各边节点。
            i = 0
            while i < len(adj_child):
                l1 = adj_child[i][0]
                l2 = adj_child[i][1]
                if l1 >= len(num) or l2 >= len(num):
                    adj_child.remove((l1, l2))
                    continue
                i = i + 1

            i = 0
            while i < len(adj_com):
                l1 = adj_com[i][0]
                l2 = adj_com[i][1]
                if l1 >= len(num) or l2 >= len(num):
                    adj_com.remove((l1, l2))
                    continue
                i = i + 1

            i = 0
            while i < len(adj_next):
                l1 = adj_next[i][0]
                l2 = adj_next[i][1]
                if l1 >= len(num) or l2 >= len(num):
                    adj_next.remove((l1, l2))
                    continue
                i = i + 1

            i = 0
            while i < len(adj_guared):
                l1 = adj_guared[i][0]
                l2 = adj_guared[i][1]
                if l1 >= len(num) or l2 >= len(num):
                    adj_guared.remove((l1, l2))
                    continue
                i = i + 1

            i = 0
            while i < len(adj_lexical):
                l1 = adj_lexical[i][0]
                l2 = adj_lexical[i][1]
                if l1 >= len(num) or l2 >= len(num):
                    adj_lexical.remove((l1, l2))
                    continue
                i = i + 1

            i = 0
            while i < len(adj_jump):
                l1 = adj_jump[i][0]
                l2 = adj_jump[i][1]
                if l1 >= len(num) or l2 >= len(num):
                    adj_jump.remove((l1, l2))
                    continue
                i = i + 1


            #处理CDFG的特征向量
            random_use = np.random.rand(100)
            vectors_cdfg = []
            model = word2vec.Word2Vec.load(os.path.split(filename)[0]+"/word2vec_" + str(os.path.split(filename)[1]) + '.pkl')
            for i in range(len(joern_word)):
                word_single1 = joern_word[i].split(" ")
                try:
                    vector = model.wv[word_single1]  # 查词
                except:
                    vector = np.reshape(random_use, (1, 100))

                vector = vector[-1, 0:100]
                #
                vectors_cdfg.append(vector.tolist())
            # if len(vectors_cdfg) < len(num):
            #     vectors_cdfg.append(vector.tolist())
            # 处理AST的特征向量
            vectors = []
            # model = word2vec.Word2Vec.load \
            #     (r'C:\Users\Orange\Desktop\RGIN/word2vec_test_369.pkl')
            # a = model.wv.vocab

            # a=os.path.split(sys.argv[0])
            with open(os.path.split(filename)[0]+"/our_map_all.txt", "r") as f:
                s = f.read()
                s = s.replace('\'', '\"')
                mapping = json.loads(json.dumps(eval(s)))
            for i in range(len(num)):
                vector = []
                word_single = num[i]
                # for j in word_single:
                # vector = model.wv[word_single]  # 查词
                vector = mapping[word_single]  # 查词
                # vector = vector[-1, 0:99]
                vectors.append(vector)
            label = int(label_single) + 0.0

            adj_singel = [adj_child, adj_com, adj_next, adj_jump, adj_guared, adj_lexical]
            adj_ast_total.append(adj_singel)
            adj_singel = [adj_cdfg_a, adj_cdfg_c, adj_cdfg_p]
            adj_cdfg_total = []
            adj_cdfg_total.append(adj_singel)
            # 存储AST文件
            with jsonlines.open(os.path.split(filename)[0]+'/'+str(os.path.split(filename)[1])+'_ast.jsonl', mode='a') as writer:
                writer.write(
                            {"Property": label, "graph": {"node_features": vectors, "adjacency_lists": adj_ast_total[0]}})
            with jsonlines.open(os.path.split(filename)[0]+'/'+str(os.path.split(filename)[1])+'_cdfg.jsonl', mode='a') as writer:
                writer.write(
                            {"Property": label, "graph": {"node_features": vectors_cdfg, "adjacency_lists": adj_cdfg_total[0]}})

            # with jsonlines.open(r'middle/ast/data.jsonl', mode='a') as writer:
                # print("fine")
                # writer.write(
                #     {"Property": label, "graph": {"node_features": vectors, "adjacency_lists": adj_ast_total[0]}})
            # with jsonlines.open(r'middle/cdfg/data.jsonl', mode='a') as writer:
                # print("fine")
                # writer.write(
                #     {"Property": label, "graph": {"node_features": vectors_cdfg, "adjacency_lists": adj_cdfg_total[0]}})
            # 训练集0.8

            # if count >= int(filenum * trp) and save == 0:
            #     f_in = open(r'middle/ast/data.jsonl', 'rb')
            #     # print(os.getcwd())
            #     f_out = gzip.open(r"middle/ast/train.jsonl.gz", 'wb')
            #     f_out.writelines(f_in)
            #     f_out.close()
            #     f_in.close()
            #     os.remove(r'middle/ast/data.jsonl')
            #     f_in = open(r'middle/cdfg/data.jsonl', 'rb')
            #     f_out = gzip.open(r"middle/cdfg/train.jsonl.gz", 'wb')
            #     f_out.writelines(f_in)
            #     f_out.close()
            #     f_in.close()
            #     os.remove(r'middle/cdfg/data.jsonl')
            #     save += 1
            # # 验证集 0.1
            # elif count >= int(filenum * vap) and save == 1:
            #     f_in = open(r'middle/ast/data.jsonl', 'rb')
            #     f_out = gzip.open(r"middle/ast/valid.jsonl.gz", 'wb')
            #     f_out.writelines(f_in)
            #     f_out.close()
            #     f_in.close()
            #     os.remove(r'middle/ast/data.jsonl')
            #     f_in = open(r'middle/cdfg/data.jsonl', 'rb')
            #     f_out = gzip.open(r"middle/cdfg/valid.jsonl.gz", 'wb')
            #     f_out.writelines(f_in)
            #     f_out.close()
            #     f_in.close()
            #     os.remove(r'middle/cdfg/data.jsonl')
            #     save += 1
        # 测试集 0.1
        # f_in = open(r'middle/ast/data.jsonl', 'rb')
        # f_out = gzip.open(r"middle/ast/test.jsonl.gz", 'wb')
        # f_out.writelines(f_in)
        # f_out.close()
        # f_in.close()
        # os.remove(r'middle/ast/data.jsonl')
        # f_in = open(r'middle/cdfg/data.jsonl', 'rb')
        # f_out = gzip.open(r"middle/cdfg/test.jsonl.gz", 'wb')
        # f_out.writelines(f_in)
        # f_out.close()
        # f_in.close()
        # os.remove(r'middle/cdfg/data.jsonl')
        # print("done")
def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' ready now')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' already')
        return False
#spilt and zip
def Split(train,valid,test,path):
    # print("path")
    # print(os.path.split(path)[0])
    tem_ast=[]
    tem_cdfg=[]
    mkdir(os.path.split(path)[0] + '/' + 'tem/ast')
    mkdir(os.path.split(path)[0] + '/' + 'tem/cdfg')
    mkdir(os.path.split(path)[0]+'/'+'tem_'+os.path.split(path)[1]+'/ast')
    mkdir(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg')


    #ast+cdfg
    with open(os.path.split(path)[0]+'/'+str(os.path.split(path)[1])+'_ast.jsonl', "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            tem_ast.append(item)
    with open(os.path.split(path)[0]+'/'+str(os.path.split(path)[1])+'_cdfg.jsonl', "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            tem_cdfg.append(item)
    randomtem=np.arange(len(tem_ast))
    np.random.shuffle(randomtem)
    for step,i in enumerate(randomtem):
        ast=tem_ast[i]
        cdfg=tem_cdfg[i]
        # a=len(tem_ast)*train
        if step+1<int(len(tem_ast)*train):
            with jsonlines.open(os.path.split(path)[0]+'/'+'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl', mode='a') as writer:
                writer.write(ast)
            with jsonlines.open(os.path.split(path)[0]+'/'+'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl', mode='a') as writer:
                writer.write(cdfg)
            continue
        if step+1 == int(len(tem_ast)*train):
            # zip
            f_in = open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl', 'rb')
            # print(os.getcwd())
            # f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/train.jsonl.gz', 'wb')
            f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem' + '/ast/train.jsonl.gz',
                              'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl')
            f_in = open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl', 'rb')
            # print(os.getcwd())
            # f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/train.jsonl.gz', 'wb')
            f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem' + '/cdfg/train.jsonl.gz',
                              'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl')
            print("train ready")
            continue
        if step+1>int(len(tem_ast)*train) and step+1<int(len(tem_ast)*(train+valid)):
            with jsonlines.open(os.path.split(path)[0]+'/'+'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl', mode='a') as writer:
                writer.write(ast)
            with jsonlines.open(os.path.split(path)[0]+'/'+'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl', mode='a') as writer:
                writer.write(cdfg)
            continue
        if step+1 == int(len(tem_ast)*(train+valid)):
            # zip
            f_in = open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl', 'rb')
            # print(os.getcwd())
            # f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/valid.jsonl.gz', 'wb')
            f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem' +  '/ast/valid.jsonl.gz',
                              'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl')
            f_in = open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl', 'rb')
            # print(os.getcwd())
            # f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/valid.jsonl.gz', 'wb')
            f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem' + '/cdfg/valid.jsonl.gz',
                              'wb')

            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl')
            print("valid ready")
            continue
        if step + 1 > int(len(tem_ast) * (train + valid)) and step + 1 < int(len(tem_ast)) :
            with jsonlines.open(os.path.split(path)[0]+'/'+'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl', mode='a') as writer:
                writer.write(ast)
            with jsonlines.open(os.path.split(path)[0]+'/'+'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl', mode='a') as writer:
                writer.write(cdfg)
            continue
        if step + 1 == int(len(tem_ast)) :
            # zip
            f_in = open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl', 'rb')
            # print(os.getcwd())
            # f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/test.jsonl.gz', 'wb')
            f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem' +  '/ast/test.jsonl.gz',
                              'wb')

            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/ast/ast.jsonl')
            f_in = open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl', 'rb')
            # print(os.getcwd())
            # f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/test.jsonl.gz', 'wb')
            f_out = gzip.open(os.path.split(path)[0] + '/' + 'tem'  + '/cdfg/test.jsonl.gz',
                              'wb')

            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(os.path.split(path)[0] + '/' + 'tem_'+os.path.split(path)[1]+'/cdfg/cdfg.jsonl')
            print("test ready")

def Preprocess(path):
    cwetype = os.path.split(path)[1]
    #检测是否有字典如果没有就训练
    w2v(path,cwetype)
    #进行jsonal的提取
    if os.path.exists(os.path.split(path)[0]+'/'+str(os.path.split(path)[1])+'_ast.jsonl') and os.path.exists(os.path.split(path)[0]+'/'+str(os.path.split(path)[1])+"_cdfg.jsonl"):
        Split(0.8,0.1,0.1,path)
    else:
        GetInfor(path)
        Split(0.8, 0.1, 0.1, path)
# Preprocess(r"E:\pythonProject\adanet\data\CWE-200")