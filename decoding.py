
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
	
import scipy
import scipy.io
import numpy as np
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, SelectFdr, f_regression
from sklearn.cross_validation import (cross_val_score, ShuffleSplit,
                                  StratifiedKFold, permutation_test_score)
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
 
def cross_validation_LR(X,Y, n_folds, C_seq, K_seq, verbose = False):
    '''
        To classify Y using X, we first use ANOVA to choose K dimensions
        in X, where the difference between different Ys are highest, then run 
        a logistic regression classifier with regularization parameter C on 
        the K dimensions. 
         
        To quantify how well X can classify Y, without specifying training and 
        testing partition, we do n_folds cross validation.
        In each fold, during training, we do an inner loop cross validation to
        select C and K that give the best classification accuracy from a given 
        range; and then we use this to classify the held-out testing data. 
         
        Inputs:
            X, [n, p], n trials of p dimensional data, used for classification
            Y, [n], class labels
            n_folds,integer,  split the data into n_folds for cross validation
            C_seq, a sequence of regularizatioin parameters for logistic 
                    regression classifiers, smaller values specify stronger
                    regularization.
                    e.g. C_seq = 10.0** np.arange(-3,1,1)
            K_seq, a sequence of integers, 
                    e.g.  K_seq = (np.floor(np.arange(0.2,1,0.2)*p)).astype(np.int)
            verbose: boolean, if ture, print the best C and K chosen
        Output:
            averaged classification accuracy of the n_folds
    '''
    cv0 = StratifiedKFold(Y,n_folds = n_folds)
    cv_acc = np.zeros(n_folds)
    for i in range(n_folds):
        ind_test = cv0.test_folds == i
        ind_train = cv0.test_folds != i
        tmpX_train = X[ind_train,:]
        tmpY_train = Y[ind_train]
        tmpX_test = X[ind_test,:]
        tmpY_test = Y[ind_test]
         
        # grid search
        tmp_cv_score = np.zeros([len(C_seq), len(K_seq)])
        for j in range(len(C_seq)):
            for k in range(len(K_seq)):
                cv1 = StratifiedKFold(tmpY_train,n_folds = n_folds)
                anova_filter = SelectKBest(f_regression, k = K_seq[k])
                clf = LogisticRegression(C = C_seq[j], penalty = "l2")
                anova_clf = make_pipeline(anova_filter, clf)
                tmp_cv_score[j,k] = cross_val_score(anova_clf, tmpX_train,
                                  tmpY_train, scoring = "accuracy",  cv = cv1).mean()
         
        best_ind = np.argmax(tmp_cv_score.ravel())
        best_j, best_k = np.unravel_index(best_ind, tmp_cv_score.shape)
         
        anova_filter = SelectKBest(f_regression, k = K_seq[k])
        clf = LogisticRegression(C = C_seq[j], penalty = "l2")
        anova_clf = make_pipeline(anova_filter, clf)
        tmpY_predict = anova_clf.fit(tmpX_train, tmpY_train).predict(tmpX_test) 
        if verbose: 
            print C_seq[best_j],K_seq[best_k]          
        cv_acc[i] =  np.mean(tmpY_test == tmpY_predict)    
    return np.mean(cv_acc)                 
         
def cross_validation_LR_permutation(X,Y,n_folds, C_seq, K_seq, perm_seq):
    '''
    Do permutation tests to quantify how significantly above chance the 
    classification accuracy is.
     
        Inputs:
            X, [n, p], n trials of p dimensional data, used for classification
            Y, [n], class labels
            n_folds,integer,  split the data into n_folds for cross validation
            C_seq, a sequence of regularizatioin parameters for logistic 
                    regression classifiers, smaller values specify stronger
                    regularization.
                    e.g. C_seq = 10.0** np.arange(-3,1,1)
            K_seq, a sequence of integers, 
                    e.g. K_seq = (np.floor(np.arange(0.2,1,0.2)*p)).astype(np.int)
            perm_seq: [n, n_perm],  permuted indices of trials to use,
                    n_perm is the number of permutation
        Output:
            cv_acc,  scalar, the cross validation classification accuracy
            cv_acc_permu, [n_perm], the permuted cross validation accuracy
            p_val, p-value of the permutation test
    '''
    n_perm = perm_seq.shape[1]
    cv_acc_permu = np.zeros(n_perm)
    for i in range(n_perm):
        Y_permu = Y[perm_seq[:,i]]
        cv_acc_permu[i] = cross_validation_LR(X,Y_permu, n_folds, C_seq, K_seq)
         
    cv_acc = cross_validation_LR(X,Y, n_folds, C_seq, K_seq)
    p_val =np.mean(cv_acc_permu>cv_acc)
    p_val = 1.0/n_perm if p_val ==0 else p_val
    return cv_acc, cv_acc_permu, p_val
 
def get_cv_acc_for_3d_data(X3d, Y, C_seq, K_seq, n_folds = 2, n_perm = 0):
    """
        Given the sensor data as multi-trial time seires, get cross validation accuracy
        for each time point, if required, do permutation tests
        Input:
            X3d [n, p, T], n trials, p sensors, and T time points
            Y,[n], condition labels of the n trials
            C_seq, K_seq, the ranges of tuning parameters
                e.g. C_seq = 10.0** np.arange(-3,1,1)
                     K_seq = (np.floor(np.arange(0.2,1,0.2)*p)).astype(np.int)
            n_folds, number of folds,
            n_perm: number of permutations, if 0, don't do permutation tests, 
                    make sure the permutation sequence is the same across 
                    all time points. 
        Output:
            acc, [T], accuracy for each time point
            acc_permu, [n_perm,T], permutation accuracy for each time point
            p_val, [T], p-values for each time point, if n_perm = 0, then it is
                all 1. 
    """
    [n,p,T] = X3d.shape
    # if n_perm >0, create a permutation sequence for all time points
    acc = np.zeros(T)
    p_val = np.ones(T)
    acc_permu = np.zeros([n_perm,T])
    if n_perm > 0:
        perm_seq = np.zeros([n,n_perm], dtype = np.int)
        orig_seq = range(0,n)
        for i in range(n_perm):
            perm_seq[:,i] = (np.random.permutation(orig_seq)).astype(np.int)
        for t in range(T):
            X = X3d[:,:,t]
            acc[t],acc_permu[:,t], p_val[t] = \
            cross_validation_LR_permutation(X,Y,n_folds, C_seq, K_seq, perm_seq)
    else:
        for t in range(T):
            X = X3d[:,:,t]
            acc[t]= cross_validation_LR(X,Y,n_folds, C_seq, K_seq)
    return acc, p_val, acc_permu
         
 
if __name__=="__main__":
    # a simple simulation 
    n,p,T = 200, 50, 30
    X3d = np.random.randn(n,p,T)
    Y = np.sign(np.dot( X3d[:,0,5:15], np.random.randn(10)))
    #=========================================================================
    # in actual application, X3d are your sensor data, Y is the trial labels,
    # if they are preprocessed in matlab, you can save them in mat format and 
    # use  scipy.io.loadmat to load them
    #=========================================================================
     
    C_seq = 10.0** np.arange(-3,1,1)
    K_seq = (np.floor(np.arange(0.2,1,0.2)*p)).astype(np.int)
    # ==============no permutation==========================================
    acc, p_val, acc_permu = \
          get_cv_acc_for_3d_data(X3d, Y, C_seq, K_seq, n_folds = 2, n_perm = 0)
     
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(acc)
     
    #============= with permutation=====================
    acc, p_val, acc_permu = \
          get_cv_acc_for_3d_data(X3d, Y, C_seq, K_seq, n_folds = 2, n_perm = 50)
    # plot 95% interval, without mulpiple comparison correction
    alpha = 0.05
    lb, ub = np.percentile(acc_permu, [alpha/2.0*100, (1.0-alpha/2.0)*100], axis = 0)
    plt.figure(); plt.plot(acc);
    plt.fill_between(np.arange(0,T), lb, ub, alpha = 0.3)
    # !!! note the p-value and confidence bands here are uncorrected for 
    # multiple comparison, an excursion test is needed to obtain the correct
    # significant time windows. 
 
    #=========================================================================
    # If you do not care about the time windows, you can concatenate all time 
    # points in X3d, i.e. X = X3d.reshape([n, -1]), and then just obtain one
    # single classificaiton accuracy
    # In such a case, you don't have to worry about correction for multiple
    # tests at different time points
    #=========================================================================
    X = X3d.reshape([n, -1])
    n_perm = 100
    perm_seq = np.zeros([n,n_perm], dtype = np.int)
    orig_seq = range(0,n)
    for i in range(n_perm):
        perm_seq[:,i] = (np.random.permutation(orig_seq)).astype(np.int)
    n_folds = 2
    cv_acc, cv_acc_permu, p_val = \
         cross_validation_LR_permutation(X,Y,n_folds, C_seq, K_seq, perm_seq)
     
    
