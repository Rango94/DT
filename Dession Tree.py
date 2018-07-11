#最小二乘
import random as rd
import numpy as np
import json
class DessionTree:

    def __init__(self,max_deep=6):
        self.max_deep=max_deep
        self.tree = Tree(0)
    def fit(self,X,Y):
        self.Split(self.tree,X,Y)

    def Split(self,tree,X,Y):
        if tree.deep<=self.max_deep and np.var(Y)!=0 and X!=[]:
            self.feature_len = len(X[0])
            best_idx = 0
            best_point = 0
            best_s = 9999999999999
            x_l = []
            y_l = []
            x_r = []
            y_r = []
            for i in range(self.feature_len):
                # print(i)
                feature_sort = [np.min(X[:, i]),np.max(X[:,i])]
                # split_point = [(feature_sort[idx] + feature_sort[idx + 1]) / 2 for idx, i in enumerate(feature_sort) if
                #                idx != len(feature_sort) - 1]
                split_point=[feature_sort[0]+k*(feature_sort[len(feature_sort)-1]-feature_sort[0])/20 for k in range(20)]

                for p in split_point:
                    now_s ,x1,y1,x2,y2= self.mini_2(X, Y, i, p)
                    if best_s > now_s:
                        best_s = now_s
                        best_idx = i
                        best_point = p
                        x_l=x1
                        y_l=y1
                        x_r=x2
                        y_r=y2
            tree.idx=best_idx
            tree.point=best_point
            l_tree=Tree(tree.deep+1)
            r_tree=Tree(tree.deep+1)
            # print(best_point)
            tree.set_node('left',self.Split(l_tree,x_l,y_l))
            tree.set_node('right', self.Split(r_tree, x_r, y_r))
        else:
            tree.set_value(np.mean(Y))
            tree.value_x=X
            tree.value_y=Y
        return tree

    def mini_2(self,x,y,IDX,point):
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
        l_mean=np.mean(left_y)
        r_mean=np.mean(right_y)
        #print(l_mean,r_mean)
        # print(left_y)
        return sum([(i-l_mean)*(i-l_mean) for i in left_y])+sum([(i-r_mean)*(i-r_mean) for i in right_y]),np.array(left_x),np.array(left_y),np.array(right_x),np.array(right_y)

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
            json.dump(self.tree.get_dic(), fo, ensure_ascii=False)

    def load_model(self,modelname):
        with open(modelname,'r',encoding='utf-8') as fo:
            self.tree=Tree(dic=json.load(fo))


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


def fur(q,w,e,r,t,y,u,i,o,p):
    return q*q*q*0.4+w*w*0.1+e*0.2+r*r*r*r*-0.7+t+y*y*0.9+u*u*u*0.4+i*0.5+o*0.4*i*p*0.3+p*p

x=[]
y=[]
for k in range(20000):
    q=rd.random()*2-1
    w = rd.random() * 2 - 1
    e = rd.random() * 2 - 1
    r = rd.random() * 2 - 1
    t = rd.random() * 2 - 1
    y_ = rd.random() * 2 - 1
    u = rd.random() * 2 - 1
    i = rd.random() * 2 - 1
    o = rd.random() * 2 - 1
    p = rd.random() * 2 - 1
    x.append(np.array([q,w,e,r,t,y_,u,i,o,p]))
    y.append(fur(q,w,e,r,t,y_,u,i,o,p))

np.save('X',np.array(x))
np.save('Y',np.array(y))
#
# print('done')
X=np.load('X.npy')
Y=np.load('Y.npy')
#
x=X[:1000]
#print(X[:100],Y[:1])
dd=DessionTree(15)
# dd.fit(X[100:],Y[100:])
dd.fit(X[1000:],Y[1000:])
dd.save_model('model')

dd1=DessionTree()
dd1.load_model('model')
# print(x)
y=dd1.predict(x)
for idx,i in enumerate(y):
    print(i,Y[idx])
print(np.mean((y-Y[:1000])*(y-Y[:1000])))







