import numpy as np
import pandas as pd
from MBagging import BaggingRegressor
from pyearth  import Earth
import csv
import  random
# compute running time
import datetime
starttime = datetime.datetime.now()
import warnings

def get_Goldset(file_path):
    gold_set=set()
    file = open(file_path)
    for line in file:
        tmp_str=line.split('\t')
        gold_set.add((tmp_str[0]+"\t"+tmp_str[1]))
    return  gold_set

def getlinks(target_name,name,importance_,gold_set):
    feature_imp=pd.DataFrame(importance_,index=name,columns=['imp'])
    feature_large_set = {}
    for i in range(0,len(feature_imp.index)):
        tmp_name=feature_imp.index[i]
        if tmp_name !=target_name:
            if (tmp_name+"\t"+target_name) not in feature_large_set:
                g_f=0
                if (tmp_name+"\t"+target_name) in gold_set:
                    g_f=1
                tf_score =feature_imp.loc[feature_imp.index[i], 'imp']
                feature_large_set[tmp_name+"\t"+target_name]=[abs(tf_score),g_f]

    return feature_large_set


def compute_feature_imp(bagger,n_feature):
    coff=np.zeros(n_feature)
    fcount=np.ones(n_feature)
    for i in range(len(bagger.estimators_)):
        base_estimate=bagger.estimators_[i]
        feature_index=bagger.estimators_features_[i]
        coff[feature_index]=np.abs(base_estimate.feature_importances_)+coff[feature_index]
        #coff[feature_index] = transform_feature_imp(base_estimate)+ coff[feature_index]
        fcount[feature_index]=fcount[feature_index]+1
    coff=coff/fcount
    #coff=coff/sum(coff)
    return coff

def transform_feature_imp(base_estimators_):
    feature_imp=base_estimators_.feature_importances_
    s_len=len(feature_imp)
    feature_rank=np.zeros(s_len)
    forword_context=str(base_estimators_.forward_trace()).split("\n")
    for i in range(4,len(forword_context)-1):
        sline=forword_context[i].split()
        index=int(sline[2])
        if feature_imp[index]!=0:
            feature_rank[index]=base_estimators_.max_terms-int(sline[0])+1
    return feature_rank


def mainRun(expressionFile,goldFile,outputfile):
    if __name__ == '__main__':
        all_large_df = pd.DataFrame(columns=["score_1", "key_1"])
        data=pd.read_csv(expressionFile,'\t')
        g_set=get_Goldset(goldFile)

        tfs = np.genfromtxt('MutiFactor/net1_transcription_factors.tsv', dtype=str, delimiter='\n')

        tf_num=len(tfs)#debug

        feature_w = np.loadtxt('MutiFactor/pmi2_dream5_order1net1_expression_data.tsv.txt')

        data=data.apply(lambda S:(S-np.mean(S))/(np.std(S)))

        for index in range(0, len(data.columns)):
            t_data=data.copy()
            y=data[data.columns[index]]
            y_normal=(y-np.mean(y))/np.std(y)
            t_data = t_data[tfs.tolist()]
            f_w = feature_w[index, :].copy()
            if data.columns[index] in tfs:
                t_data=t_data.drop(data.columns[index],axis=1)
                tf_index=np.argwhere(tfs==data.columns[index])
                f_w = np.delete(f_w, tf_index)
            f_w=f_w/sum(f_w)
            bagger=BaggingRegressor(base_estimator=Earth(penalty =5,feature_importance_type="gcv",max_terms =10,use_fast=True,fast_h=2,allow_missing=True),n_estimators=1000,n_jobs=2,random_state=random.randint(1,100),bootstrap=True,max_features =20)
            bagger.fit(t_data,y_normal,feature_weight=f_w)
            _importance_per=compute_feature_imp(bagger,len(t_data.columns))
            tmp_large = getlinks(data.columns[index], t_data.columns.values, _importance_per.transpose(), g_set)
            aaa=pd.DataFrame.from_dict(tmp_large, orient='index')
            aaa.columns=["score_1", "key_1"]
            print(index)

            all_large_df = all_large_df.append(aaa)
        all_large_df.to_csv(outputfile, sep="\t", header=False,quoting=csv.QUOTE_NONE,escapechar=" ")

# mainRun("../MutiFactor/insilico_size100_5_multifactorial.tsv","../MutiFactor/DREAM4_GoldStandard_InSilico_Size100_multifactorial_5.tsv",'../Result/mars_test.txt')
warnings.simplefilter("ignore")

mainRun("MutiFactor/net1_expression_data.tsv","MutiFactor/DREAM5_NetworkInference_GoldStandard_Network1 - in silico.tsv",'Result/mar_dream5pmi_1_0222_1.txt')

# runi=1
# mainRun("MutiFactor/insilico_size100_"+str(runi)+"_multifactorial.tsv","MutiFactor/DREAM4_GoldStandard_InSilico_Size100_multifactorial_"+str(runi)+".tsv",'Result/mars_test'+str(runi)+'.txt')


# why the endtime output so many times???