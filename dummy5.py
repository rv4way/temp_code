# import Gist_feat_last
# import HOG_feat2
from sort_dic import sort
import os
import numpy as np
from sklearn.externals import joblib  #save the data
import cv2
import json
# from pymongo import MongoClient
# import afine_search
# import histogram
from scipy.misc import imresize
# with open('Properties.json', 'r') as fp:
#     data = json.load(fp)

# # dir_gist = data["ClassifierGist"]
# # dir_hog = data["ClassifierHog"]
# # m_a = data["MongoUrl"]

# c = MongoClient(m_a)   #taking instance of mongo client
# mer4 = data["ImageDatabase"]
# db = c[mer4] 
# db_classig = db.ClasiGabor
# db_classih = db.ClassiHog
# list1 = {} #dictionary to hold data
# list2 = {}
# list3 = {}
# files_name = []

orgi_gist = {}
orig_hog = {}
aff_gist = {}
aff_hog = {}




#just return the name of company of classiffier
def remove_num(list_temp):
    for i in range(len(list_temp)):
        temp = list_temp[i]
        temp1 = temp.split('_')
        temp = temp1[0]
        list_temp[i] = temp
        
    return list_temp



def Label_classify(feature,files1):
    final_gist = {}
    dir2 = dir_gist #directory where the classifier are
    for subdir2,newdir1,files3 in os.walk(dir2):
        list1[files1]=[]
        files_name.append(files1)
        for files4 in files3:
            machine_path = dir2+'/'+files4
            profile_id = ((files4.split('_'))[0])
            clf = joblib.load(machine_path) #load the classifier
            predict = clf.predict(feature) #predict the class
            predict = np.asarray(predict)
            if predict.all()==1:  #if class is one then add it                
                list1[files1].append(files4)
                if final_gist.has_key(profile_id):
                     temp = final_gist[profile_id]
                     temp += 1
                     final_gist[profile_id] = temp
                else:
                     final_gist[profile_id] = int(1)
    #print final_gist
    return final_gist
    #return list1
    

                
def Label_classify2(feature,files1):
    final_rv = {}
    dir2 = dir_hog #directory where the classifier are
    for subdir2,newdir1,files3 in os.walk(dir2):
        list2[files1]=[]
    #print files3
        for files4 in files3:
            machine_path = dir2+'/'+files4
            profile_id = (files4.split('_'))[0]
            clf = joblib.load(machine_path) #load the classifier
            predict = clf.predict(feature) #predict the class
            predict = np.asarray(predict)
            if predict.all()==1:  #if class is one then add it
                #print 'Prediction is:',files4
                list2[files1].append(files4)
                if final_rv.has_key(profile_id):
                     temp = final_rv[profile_id]
                     temp += 1
                     final_rv[profile_id] = temp
                else:
                     final_rv[profile_id] = int(1)
    #print final_rv
    return final_rv
    #return list2

def gen_res(final_rv, final_gist):
    temp_gist = final_gist.keys()
    temp_hog = final_rv.keys()
    final_temp = list(set(temp_gist).intersection(temp_hog))
    #print temp_gist
    #print temp_hog
    #print final_temp
    return final_temp      
         
    
def image_calc(d1,d2):
    '''
    img = imresize(img, (47*2, 55*2), interp = 'bicubic')
    correct_fea = Gist_feat_last.singleImage2(img)
    feat = HOG_feat2.hog_call(img)

    orig_hog = Label_classify2(feat,'batman')
    orig_gist = Label_classify(correct_fea,'batman')

    orig_res = gen_res(orig_hog, orig_gist)

    af_img = afine_search.affine_transform(img)

    af_gist = Gist_feat_last.singleImage2(af_img)
    af_hog = HOG_feat2.hog_call(af_img)
    aff_hog = Label_classify2(feat,'batman')
    aff_gist = Label_classify(correct_fea,'batman')
    af_res = gen_res(aff_hog, aff_gist)

    final_res = list(set(orig_res).intersection(af_res))

    #final_response = histogram.search_hist(final_res, img)
    print final_res
    print orig_res
    print af_res
    print orig_hog
    print orig_gist
    print aff_hog
    print aff_gist
    
    new_hog, hog_weig = make_final_hog(aff_hog, orig_hog)
    new_gist, gist_weig = make_final_hog(aff_gist, orig_gist)
    '''
    new_hog, hog_weig = make_final_hog(d1, d2)
    new_gist, gist_weig = make_final_hog(d1, d2)
    
    if hog_weig[0] > gist_weig[0]:

        new_hog_shrink = []
        if len(new_hog) > 3:
            for x in range(3):
                new_hog_shrink.append(new_hog[x])
            return new_hog_shrink
        else:
            return new_hog

    elif hog_weig[0] < gist_weig[0]:

    	if gist_weig > 1:

            new_gist_shrink = []
            if len(new_gist) > 3:
                for x in range(3):
                    new_gist_shrink.append(new_gist[x])

                return new_gist_shrink
            else: 
                return new_gist
    	else:
    	    return []

    elif hog_weig[0] == gist_weig[0]:
        temp = []
        for x in new_hog:
            temp.append(x)
        for x in new_gist:
            temp.append(x)

        temp_shrink = []
        if len(temp) > 3:
            for x in range(3):
                temp_shrink.append(temp[x])

            return temp_shrink
        else: 
            return temp
        

def search(img_arr):

    res1 = image_calc(img_arr)
    in_logo = afine_search.inside_logo(img)
    
    res2 = image_calc(in_logo)

    res3 = list(set(res1).intersection(res2))

    print res3
    return res3

def make_final_hog(d1, d2):

    for x in d2:
        if d1.has_key(x):
            temp = d2[x]
            if d1[x] < temp:
                d1[x] = temp
                
        else:
            d1[x] = d2[x]

    #d_sort_k, d_sort_v = sort(d1)
    

    #print d_sort_k, d_sort_v

    return sort(d1)
    



if __name__ == '__main__':
    # path = '/root/ideaswire/imageprocessing/logo_rv/database/1.png'
    # img = cv2.imread(path)
    # print img
    # search(img)
    dic_inp = {
        'first' : 54, 'forth' : 33, 'second' : 9030, 'third' : 43, 'sixth' : 21
    }
    dic_inp2 = {
        'first' : 1, 'forth' : 4, 'second' : 900, 'third' : 44, 'fifth' : 32 
    }
    #make_final_hog(dic_inp, dic_inp2)
    x = image_calc(dic_inp, dic_inp2)
    print x