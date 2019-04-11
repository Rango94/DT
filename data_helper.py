import numpy as np
import sys
import random as rd


def _fur(q, w, e, r, t, y, u, i, o, p, chaos):
    if chaos:
        q, w, e, r, t, y, u, i, o, p=[i+(rd.random()*0.2-0.1) for i in [q, w, e, r, t, y, u, i, o, p]]
    return q*q*q*0.4+w*w*0.1+e*0.2+r*r*r*r*-0.7+t+y*y*0.9+u*u*u*0.4+i*0.5+o*0.4*i*p*0.3+p*p

def genaret_data(num,chaos=False):
    x=[]
    y=[]
    print(num)
    for k in range(num):
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
        y.append(_fur(q, w, e, r, t, y_, u, i, o, p,chaos))
    np.save('X',np.array(x))
    np.save('Y',np.array(y))
    print('done')

if __name__=='__main__':
    num=int(sys.argv[1])
    chaos=0
    if len(sys.argv)==2:
        chaos=False
    elif sys.argv[2]=='true':
        chaos=True
    elif sys.argv[2]=='false':
        chaos=False
    else:
        print('cate error')
    genaret_data(num,chaos)
