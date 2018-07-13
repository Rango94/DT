from Dession_Tree import DessionTree
import random as rd
import numpy as np
import json


class random_forest:
    def __init__(self,max_trees,max_deep,cate='val'):
        self.max_trees=max_trees
        self.max_deep=max_deep
        self.cate=cate
        self.trees_dic={}
        for i in range(self.max_trees):
            self.trees_dic[i]=DessionTree(max_deep=max_deep,cate=cate)

    def dis_the_data(self,X,Y,k):
        out_X=[]
        out_Y=[]
        idx_list=[]
        for i in range(k):
            rd_=rd.randint(0,len(X)-1)
            if rd_ not in idx_list:
                idx_list.append(rd_)

        for i in idx_list:
            out_X.append(X[i])
            out_Y.append(Y[i])
        return np.array(out_X),np.array(out_Y)

    def fit(self,X,Y):
        for i in range(self.max_trees):
            X_tmp,Y_tmp=self.dis_the_data(X,Y,int(len(X)/self.max_trees))
            self.trees_dic[i].fit(X_tmp,Y_tmp)

    def predict(self,X):
        y_list=[]
        for i in range(self.max_trees):
            y_list.append(self.trees_dic[i].predict(X))
        return np.mean(np.array(y_list),0)

    def save_model(self, modelname):
        with open(modelname, 'w', encoding='utf-8') as fo:
            json.dump(self.get_dic(), fo, ensure_ascii=False)

    def load_model(self, modelname):
        with open(modelname, 'r', encoding='utf-8') as fo:
            tmp_dic = json.load(fo)
        for key in tmp_dic:
            dt = DessionTree()
            dt.load_model_by_dic(tmp_dic[key])
            self.trees_dic[int(key)] = dt

        self.max_trees = len(tmp_dic)

    def get_dic(self):
        out_dic = {}
        for key in self.trees_dic:
            out_dic[key] = self.trees_dic[key].tree.get_dic()
        return out_dic

