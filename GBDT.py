from Dession_Tree import DessionTree
import Dession_Tree
import numpy as np
import json

class GBDT:
    def __init__(self,max_trees=6,max_deep=6,cate='val'):
        self.trees_dic={}
        self.max_trees=max_trees
        self.max_deep=max_deep
        for i in range(max_trees):
            self.trees_dic[i]=DessionTree(cate=cate, max_deep=max_deep)

    def fit(self,X,Y):
        for i in range(self.max_trees):
            print('training NO.'+str(i)+' tree')
            self.trees_dic[i].fit(X, Y)
            Y-=self.trees_dic[i].predict(X)

    def predict(self,X):
        out=np.zeros(len(X))
        for i in range(self.max_trees):
            out+=self.trees_dic[i].predict(X)
        return out

    def save_model(self,modelname):
        with open(modelname,'w',encoding='utf-8') as fo:
            json.dump(self.get_dic(), fo, ensure_ascii=False)

    def load_model(self,modelname):
        with open(modelname, 'r', encoding='utf-8') as fo:
            tmp_dic=json.load(fo)
        for key in tmp_dic:
            dt=DessionTree()
            dt.load_model_by_dic(tmp_dic[key])
            self.trees_dic[int(key)]=dt

        self.max_trees=len(tmp_dic)

    def get_dic(self):
        out_dic={}
        for key in self.trees_dic:
            out_dic[key]=self.trees_dic[key].tree.get_dic()
        return out_dic
if __name__ == '__main__':
    Dession_Tree.genaret_data(20000)

    X = np.load('X.npy')
    Y = np.load('Y.npy')
    x_test = X[:1000]
    y_test = Y[:1000]
    x_train = X[1000:]
    y_train = Y[1000:]
    import sys
    import os

    cate = 'val'
    if cate != 'val' and cate != 'id3' and cate != 'c4.4':
        print('cate error')
    else:
        if not os.path.exists('GBDTmodel.' + cate):
            dd = GBDT(max_trees=6, max_deep=6, cate=cate)
            dd.fit(x_train, y_train)
            dd.save_model('GBDTmodel.' + cate)

        dd1 = GBDT()
        dd1.load_model('GBDTmodel.' + cate)

        y = dd1.predict(x_test)
        for idx, i in enumerate(y):
            print(i, y_test[idx])
        print(np.mean((y - y_test) * (y - y_test)))




