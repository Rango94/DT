#最小二乘
import random as rd
import numpy as np
import json
import math

class DessionTree:

    def __init__(self,data_cate='con',cate='val',max_deep=6):
        self.max_deep=max_deep  #最大深度
        self.data_cate=data_cate    #数据类型是连续还是离散
        self.cate=cate  #决策树类型，方差 还是信息增益 还是信息增益率
        self.tree = Tree(0) #初始化一棵树
        self.Dic=0  #一个辅助字典用于把连续的值切分为离散的值

    def fit(self,X,Y):
        self.feature_len = len(X[0])
        print('建立分裂点')
        self.split_the_X(X)
        print('建立完毕')
        if self.data_cate=='con' and self.cate!='val':
            print('将回归问题离散化')
            Y=self.dis_the_con(Y)
            print('离散化完毕')
        self.Split(self.tree,X,Y,'#')

    #将连续的y值离散化，因为id3和c4.5只能做分类不能做回归。
    def dis_the_con(self,Y):
        max_ = np.max(Y)
        min_ = np.min(Y)
        step = (max_ - min_) / 200
        # print(max_,min_)
        self.Dic = {}
        for i in range(200):
            self.Dic[i] = [min_ + step * i, min_ + step * (i + 1)]
        for idx, i in enumerate(Y):
            if i <= self.Dic[0][0]:
                Y[idx] = self.Dic[0][0]
            elif i > self.Dic[len(self.Dic) - 1][1]:
                Y[idx] = self.Dic[len(self.Dic) - 1][1]
            else:
                for key in self.Dic:
                    if i <= self.Dic[key][1] and i > self.Dic[key][0]:
                        # print(Y[idx],end='\t')
                        Y[idx] = self.Dic[key][0]
                        # print(Y[idx])
        # for i in Y:
            # print(i)
        return Y

    def Split_bak(self,tree,X,Y):
        if tree.deep<=self.max_deep and np.var(Y)!=0 and X!=[]:
            best_idx = 0
            best_point = 0
            best_value = 9999999999999
            x_left = []
            y_left = []
            x_right = []
            y_right = []
            for i in range(self.feature_len):
                # feature_sort = [np.min(X[:, i]),np.max(X[:,i])]
                # split_point=[feature_sort[0]+k*(feature_sort[1]-feature_sort[0])/50 for k in range(1,49)]
                for p in self.X_point_dic[i]:
                    value ,x1,y1,x2,y2= self.select_the_point(X, Y, i, p)
                    if best_value > value:
                        best_value = value
                        best_idx = i
                        best_point = p
                        x_left=x1
                        y_left=y1
                        x_right=x2
                        y_right=y2

            if y_right==[] or y_left==[]:
                tree.set_value(np.mean(Y))
                tree.value_x = X
                tree.value_y = Y
                return tree

            tree.idx=best_idx
            tree.point=best_point
            l_tree=Tree(tree.deep+1)
            r_tree=Tree(tree.deep+1)
            tree.set_node('left',self.Split(l_tree,x_left,y_left))
            tree.set_node('right', self.Split(r_tree, x_right, y_right))
        else:
            tree.set_value(np.mean(Y))
            tree.value_x=X
            tree.value_y=Y
        return tree

    def Split(self, tree, X, Y,route):
        if tree.deep <= self.max_deep:
            best_idx = 0
            best_point = 0
            best_value = 9999999999999
            x_left = []
            y_left = []
            x_right = []
            y_right = []
            for i in range(self.feature_len):
                # feature_sort = [np.min(X[:, i]),np.max(X[:,i])]
                # split_point=[feature_sort[0]+k*(feature_sort[1]-feature_sort[0])/50 for k in range(1,49)]
                for p in self.X_point_dic[i]:
                    value, x1, y1, x2, y2 = self.select_the_point(X, Y, i, p)
                    if best_value > value:
                        best_value = value
                        best_idx = i
                        best_point = p
                        x_left = x1
                        y_left = y1
                        x_right = x2
                        y_right = y2
            if y_right.size==0 or y_left.size==0:
                tree.set_value(np.mean(Y))
                tree.value_x = X
                tree.value_y = Y
                return tree
            tree.idx = best_idx
            tree.point = best_point
            l_tree = Tree(tree.deep + 1)
            r_tree = Tree(tree.deep + 1)
            print('节点深度%d'%tree.deep,'节点路径%s'%route,'分裂特征索引%d'%best_idx,'分裂值%0.6f'%best_point)
            tree.set_node('left', self.Split(l_tree, x_left, y_left,route+'0'))
            tree.set_node('right', self.Split(r_tree, x_right, y_right,route+'1'))
        else:
            tree.set_value(np.mean(Y))
            tree.value_x = X
            tree.value_y = Y
        return tree

    def select_the_point(self, x, y, IDX, point):
        left_y=[]
        right_y=[]
        left_x=[]
        right_x=[]
        for idx,i in enumerate(x):
            if x[idx][IDX]<point:
                left_y.append(y[idx])
                left_x.append(i)
            else:
                right_y.append(y[idx])
                right_x.append(i)
        if right_y==[] or left_y==[]:
            return 9999999999999,np.array([]),np.array([]),np.array([]),np.array([])
        if self.cate=='val':
            l_mean = np.mean(left_y)
            r_mean = np.mean(right_y)
            return sum([(i-l_mean)*(i-l_mean) for i in left_y])+sum([(i-r_mean)*(i-r_mean) for i in right_y]),\
                   np.array(left_x),\
                   np.array(left_y),\
                   np.array(right_x),\
                   np.array(right_y)
        if self.cate=='id3':
            return -(self.Entropy(y)-((len(left_y)/len(y))*self.Entropy(left_y)+(len(right_y)/len(y))*self.Entropy(right_y))),\
                   np.array(left_x),\
                   np.array(left_y),\
                   np.array(right_x),\
                   np.array(right_y)
        if self.cate=='c4.5':
            E=self.Entropy(y)
            return -((E-(len(left_y)/len(y)*self.Entropy(left_y)+len(right_y)/len(y)*self.Entropy(right_y)))/E),\
                   np.array(left_x),\
                   np.array(left_y),\
                   np.array(right_x),\
                   np.array(right_y)

    def Entropy(self,y):
        dic = {}
        for i in y:
            try:
                dic[i]+=1
            except:
                dic[i]=1
        total=sum([dic[i] for i in dic])
        return -sum([(dic[i]/total)*math.log(dic[i]/total) for i in dic])


    def split_the_X(self,X,scale=50):
        self.X_point_dic={}
        for i in range(self.feature_len):
            feature_sort = [np.min(X[:, i]), np.max(X[:, i])]
            self.X_point_dic[i]=[feature_sort[0] + k * (feature_sort[1] - feature_sort[0]) / scale for k in range(1, scale-1)]


    def sort(self,x):
        if len(x)>1:
            flag=x[rd.randint(0,len(x)-1)]
            i=0
            j=len(x)-1
            while i<j:
                while x[i]<flag:
                    i+=1
                while x[j]>flag:
                    j-=1
                if x[i]==x[j]:
                    i+=1
                else:
                    tmp=x[i]
                    x[i]=x[j]
                    x[j]=tmp
            x[:i]=self.sort(x[:i])
            x[i:]=self.sort(x[i:])
        return x

    def predict(self,X):
        try:
            X=np.reshape(X,(-1,len(X[0])))
        except:
            X = np.reshape(X, (-1, len(X)))
        out=[]
        for i in X:
            out.append(self.tree.get(i))
        return np.array(out)

    def save_model(self,modelname):
        with open(modelname,'w',encoding='utf-8') as fo:
            model={
                'tree':self.tree.get_dic(),
            }
            json.dump(model, fo, ensure_ascii=False)

    def load_model(self,modelname):
        with open(modelname,'r',encoding='utf-8') as fo:
            model=json.load(fo)
            self.tree=Tree(dic=model['tree'])

    def load_model_by_dic(self,dic):
        self.tree = Tree(dic=dic)


class Tree:
    def __init__(self,deep=0,dic=0):
        if dic==0:
            self.deep=deep
            self.idx=0
            self.point=0
            self.value=0
            self.end=False
            self.value_x=[]
            self.value_y=[]
        else:
            # print(dic['end'])
            if dic['end'] == False:
                self.deep = dic['deep']
                self.idx = dic['idx']
                self.point = dic['point']
                self.end = dic['end']
                self.value = dic['value']
                self.left_node=Tree(dic=dic['left'])
                self.right_node = Tree(dic=dic['right'])
            else:
                self.deep = dic['deep']
                self.idx = dic['idx']
                self.point = dic['point']
                self.end = dic['end']
                self.value = dic['value']
                self.left_node = dic['left']
                self.right_node = dic['right']

    def get(self,x):
        if self.end==True:
            return self.value
        elif x[self.idx]<self.point:
            # print('xiao',self.idx,x[self.idx],self.point)
            return self.left_node.get(x)
        elif x[self.idx]>=self.point:
            # print('da',self.idx,x[self.idx], self.point)
            return self.right_node.get(x)

    def set_node(self,l_or_r,node):
        if l_or_r=='left':

            self.left_node=node
        if l_or_r=='right':
            self.right_node=node

    def set_value(self,value):
        self.value=value
        self.end=True

    def get_dic(self):
        if self.end:
            out = {
                'deep': self.deep,
                'idx': self.idx,
                'point': self.point,
                'end': self.end,
                'value': self.value,
                'left': 0,
                'right': 0
            }
        else:
            out={
                'deep':self.deep,
                'idx':self.idx,
                'point':self.point,
                'end':self.end,
                'value':self.value,
                'left':self.left_node.get_dic(),
                'right':self.right_node.get_dic()
            }
        return out



# dd=DessionTree()
# feature_sort=dd.sort(np.array([0,0,7,21,8,101,101,101,101,68,3,6]))
# print(dd.sort(np.array([0,0,7,21,8,101,101,101,101,68,3,6])))
# print([(feature_sort[idx]+feature_sort[idx+1])/2 for idx,i in enumerate(feature_sort) if idx!=len(feature_sort)-1])


if __name__=='__main__':

    X=np.load('X.npy')
    Y=np.load('Y.npy')
    x_test=X[:1000]
    y_test=Y[:1000]
    x_train=X[1000:]
    y_train=Y[1000:]
    import sys
    import os

    cate=sys.argv[1]
    if cate!='val' and cate!='id3' and cate!='c4.5':
        print('cate error')
    else:
        if not os.path.exists('basemodel.'+cate) or (len(sys.argv)==3 and sys.argv[2]=='force'):
            dd=DessionTree('con',max_deep=6,cate=cate)
            dd.fit(x_train,y_train)
            dd.save_model('basemodel.'+cate)

        dd1=DessionTree()
        dd1.load_model('basemodel.'+cate)

        # y=dd1.predict(x_test)
        # for idx,i in enumerate(y):
        #     print(i,y_test[idx])
        # print(np.mean((y-y_test)*(y-y_test)))







