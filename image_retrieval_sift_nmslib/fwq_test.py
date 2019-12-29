#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 19:46:02 2018

@author: kasa
"""
import os
import cv2
import math
import sys
import timeit
import numpy as np
import falconn
import mysql_conn
import gc 
import pickle as pkl

src_folder = "/home/kasa/桌面/图像检索/sift方法/pictures_src0"
lib_folder = "/home/kasa/桌面/图像检索/sift方法/pictures_lib0"
index_folder = '/home/kasa/桌面/图像检索/sift方法/index/'          
def folderCheck(foldername):
    if foldername:
        if not os.path.exists(foldername):
            os.mkdir(foldername)
            print("Info: Folder \"%s\" created" % foldername)
        elif not os.path.isdir(foldername):
            print("Error: Folder \"%s\" conflict" % foldername)
            return False
        
    return True

def gen_near_neighbor(v, r):
    rp = np.random.randn(v.size)
    rp = rp / np.linalg.norm(rp)
    rp = rp - np.dot(rp, v) * v
    rp = rp / np.linalg.norm(rp)
    alpha = 1 - r * r / 2.0
    beta = math.sqrt(1.0 - alpha * alpha)
    
    return alpha * v + beta * rp


def aligned(a, alignment=32):
    if (a.ctypes.data % alignment) == 0:
        return a
    extra = alignment // a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) // a.itemsize
    aa = buf[ofs:ofs + a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    
    return aa

def insert_fea_infos_db(values):
    sql = "insert into img_sift_features values(%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    mysql = mysql_conn.MyPymysqlPool("dbMysql")
    mysql.insertMany(sql,values)
    mysql.dispose()

def insert_fea_info_db(values):
    sql = "insert into img_sift_features values(%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"\
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    mysql = mysql_conn.MyPymysqlPool("dbMysql")
    mysql.insert(sql,values)
    mysql.dispose()

def insert_img_infos_db(values):
    sql = "insert into img_info values(%s,%s,%s,%s)"
    mysql = mysql_conn.MyPymysqlPool("dbMysql")
    mysql.insertMany(sql,values)
    mysql.dispose()

def get_fea_from_db(num_s,num_e):
    if num_s==0:
        sql = "SELECT * FROM img_sift_features limit %s"
        value = [num_e]
    else:
        sql = "SELECT * FROM img_sift_features limit %s,%s"
        value = [num_s-1,num_e-num_s]
    mysql = mysql_conn.MyPymysqlPool("dbMysql")
    result = mysql.getAll(sql, value)
    
    return result

def get_one_pic_sift(pic_path):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures =300)
    img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    kp, des = sift.detectAndCompute(img, None)
    
    return kp,des,img.shape

def one_pic_sift_point_info(pic_path,pic_num):
    fea_infos =[]
    kp,des,size = get_one_pic_sift(pic_path)
    if des.any():
        for i in range(des.shape[0]):
            fea_info = [pic_num,round(kp[i].pt[0]),round(kp[i].pt[1])]
            fea_info +=des[i].astype(np.int8).tolist()
            fea_infos.append(fea_info)
    img_info = [pic_num,pic_path,size[0],size[1]]      
    return fea_infos,img_info    

def many_pic_sift_point_info(file,pic_num_s):    
    if folderCheck(file):
        for fpathe,dirs,fs in os.walk(file):
            #fea_infos = []
            img_infos = []
            for pic_path in fs:
                if pic_path.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF","WEBP"):
                    pic_num_s+=1
                    fea_info,img_info=one_pic_sift_point_info(os.path.join(fpathe,pic_path),pic_num_s)
                    #fea_infos+=fea_info
                    img_infos.append(img_info)
                    insert_fea_infos_db(fea_info)
            insert_img_infos_db(img_infos)
            del img_infos
            gc.collect()
            print('%s get fea finshed!'%fpathe)
    return pic_num_s
                
def queries_sift_fea(file):
    queries=[]
    sift = cv2.xfeatures2d.SIFT_create(nfeatures =300)
    if folderCheck(file):
        for fnsrc in os.listdir(file):
            if fnsrc.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF","WEBP"):
                img = cv2.imdecode(np.fromfile(file + '/' + fnsrc, dtype=np.uint8), -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                kp, des = sift.detectAndCompute(img, None)
                for i in range(des.shape[0]):
                    queries.append(des[i])
    queries = np.array(queries)
    queries = queries.astype(np.float32)   
    queries -= np.mean(queries, axis=0)            
    queries = aligned(queries)
    return queries

def run_experiment(query_obj, queries,data_db):
    sorted_dic = dict.fromkeys(range(data_db[-1]['img_id']+1), 0)
    for query in queries:
        res = query_obj.find_nearest_neighbor(query)
        sorted_dic[data_db[res]['img_id']] +=1
    sorted_dic = sorted(sorted_dic.items(),key = lambda x:x[1],reverse = True)
    
    return sorted_dic

def set_cp(data):
    """
    d = 128
    seed = 119417657
    # Cross polytope hashing
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = d
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
    params_cp.storage_hash_table = falconn.StorageHashTable.FlatHashTable
    params_cp.k = 3
    params_cp.l = 10
    params_cp.num_setup_threads = 0
    params_cp.last_cp_dimension = 16
    params_cp.num_rotations = 3
    params_cp.seed = seed ^ 833840234
    """
    num_points, dim = data.shape
    parms = falconn.get_default_parameters(num_points, dim)
    falconn.compute_number_of_hash_functions(7, parms)

    cp_table = falconn.LSHIndex(parms)
    cp_table.setup(data)
    qo = cp_table.construct_query_object()
    qo.set_num_probes(896)
    return qo

def pre_data_for_cp(data_db):
    data = []
    for res in data_db:
        data.append(np.array([(d,x) for d,x in res.items()])[3:,1])
    data = np.array(data)
    data = data.astype(np.float32)
    data -= np.mean(data, axis=0)
    data = aligned(data)
    return data

def inser_all_to_db(file):
    pic_num_s=-1
    for lib_f in os.listdir(file):
        pic_num_s = many_pic_sift_point_info(file+'/'+lib_f,pic_num_s)
    return pic_num_s

def once_search(queries,num_s,num_e,index_file):    
    data_db = get_fea_from_db(num_s,num_e)
    features = pre_data_for_cp(data_db)
    if os.path.isfile(index_file):
        qo = load_index(index_file)
    else:
        qo = set_cp(features)
        save_index(qo,index_file)
    sorted_result= run_experiment(qo, queries,data_db)
    
    return sorted_result[0]

def many_time_search(queries):
    sorted_result=dict()
    count_fea=50501
    times =3
    j=0
    each_count = count_fea//times
    for i in range(0,times):
        if i==times-2:
            res_tmp=once_search(queries,j,count_fea,index_folder+'index_%d.pkl'%i)
        else:
            res_tmp=once_search(queries,j,j+each_count,index_folder+'index_%d.pkl'%i)
        j+=each_count
        sorted_result[res_tmp[0]] = res_tmp[1]
    sorted_result = sorted(sorted_result.items(),key = lambda x:x[1],reverse = True)
    return sorted_result

def save_index(obj,file):
    pkl.dump(obj, open(file, 'wb'))
    
def load_index(file):
    return pkl.load(open(file, 'rb'))

queries=queries_sift_fea(src_folder)
#pic_count=inser_all_to_db(lib_folder)
#print('%d pictures all!'%pic_count)
sorted_result=many_time_search(queries)
print(sorted_result[0])

