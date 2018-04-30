# coding: utf-8

"""
created on 2018/04/10 by wzh
"""

import numpy as np
import pandas as pd
import glob
import MeCab
import re
import nltk
import traceback
#from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet as wn
from nltk import *

retrieve_from_db = False
preprocessing = False
textout = False


frame = pd.DataFrame()
list_jp = []
list_merge = []
list_en = []
month_start = 1
month_end =  12
year = 2014
period = ["%.2d" % i for i in range(month_start, month_end+1)]

column_need = ["PNAC", "UNIQUE_STORY_INDEX", "HEADLINE_ALERT_TEXT", "TAKE_TEXT",
               "corrected","update_code","reference_code"]
column_need_JP = ["PNAC_JP", "UNIQUE_STORY_INDEX_JP", "HEADLINE_ALERT_TEXT_JP",
                  "TAKE_TEXT_JP","UPDATE_CODE_JP"]
column_need_EN = ["PNAC_EN", "UNIQUE_STORY_INDEX_EN", "HEADLINE_ALERT_TEXT_EN",
                  "TAKE_TEXT_EN","UPDATE_CODE_EN"]
df_text_jp = pd.DataFrame()
df_text_en = pd.DataFrame()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--retrieve", help="increase output verbosity")
parser.add_argument("--preprocess", help="increase output verbosity")
parser.add_argument("--year", help="increase output verbosity")
parser.add_argument("--textout", help="get all cleaned text of EN and JP")
parser
args = parser.parse_args()
if args.retrieve:
    retrieve_from_db = True
if args.preprocess:
    preprocessing = True
if args.year:
    year = args.year
if args.textout:
    textout = True
    
base_path = "/data/data_wzh/RTRS/"+str(year)+"/"
work_path = "~/research/data_prepare/"
file_name = str(year)+"_merge_jp_en.csv"
file_name2 = str(year)+"_cleaned_jp_en.csv"
file_name_text_en = str(year)+"_corpus_en.txt"
file_name_text_jp = str(year)+"_corpus_jp.txt"
    
# --- Load CSV file with JP and EN filtering --- #
if retrieve_from_db:
    for month in period:
        # Load the file monthly
        file_path = base_path + "rna002_RTRS_" + str(year) + "_" + month + ".csv"
        try :
            df = pd.read_csv(file_path, index_col=None, header=0)

            # Filter out the Japanese articles
            df_month_jp = df[df["LANGUAGE"].isin(["JA"])
                             & df["EVENT_TYPE"].isin(["STORY_TAKE_OVERWRITE"])
                             & df["TAKE_TEXT"].str.contains("参照番号")]
                             #& not((df["HEADLINE_ALERT_TEXT"].str.contains("〔表〕")))]

            # Filter out the English articles
            df_month_en = df[df["LANGUAGE"].isin(["EN"])
                             & df["EVENT_TYPE"].isin(["STORY_TAKE_OVERWRITE"])]
                             #& not((df["HEADLINE_ALERT_TEXT"].str.contains("TABLE")))]

            # Extract the reference code for Japanese news [会有警告，为何？]
            df_month_jp["reference_code"] = df_month_jp["TAKE_TEXT"].str.extract('参照番号\\[([\\w]+)\\]')

            # Drop the row where reference code is NaN
            df_month_jp = df_month_jp.dropna(subset=["reference_code"])
            
            # Extract update code for all news
            df_month_en["update_code"] = df_month_en["HEADLINE_ALERT_TEXT"].str.extract('UPDATE[\s]*([1-9])')
            df_month_jp["update_code"] = df_month_jp["HEADLINE_ALERT_TEXT"].str.extract('UPDATE[\s]*([1-9])')
            
            df_month_en["update_code"] = df_month_en["update_code"].fillna(value = 1)
            df_month_jp["update_code"] = df_month_jp["update_code"].fillna(value = 1)
            
            #Extract part of "CORRECTED"
            df_month_en["corrected"] = df_month_en["HEADLINE_ALERT_TEXT"].str.extract('(CORRECTED)')
            df_month_jp["corrected"] = df_month_jp["HEADLINE_ALERT_TEXT"].str.extract('(訂正)')
            
            df_month_en["corrected"] = df_month_en["corrected"].where(df_month_en["corrected"].isnull(), 1)
            df_month_jp["corrected"] = df_month_jp["corrected"].where(df_month_jp["corrected"].isnull(), 1)
            
            df_month_en["corrected"] = df_month_en["corrected"].fillna(value = 0)
            df_month_jp["corrected"] = df_month_jp["corrected"].fillna(value = 0)
            
            
            # Find the English news basing on the Japanese news
            df_month_merge = pd.merge(df_month_jp[column_need].reset_index(),
                                      df_month_en[column_need[:-1]].reset_index(),
                                      left_on=['reference_code', "update_code","corrected"],
                                      right_on=['PNAC',"update_code","corrected"],
                                      how="left").dropna()
            list_merge.append(df_month_merge)

            # # Append the Data frame to the list
            # list_jp.append(df_month_jp[column_need])
            # list_en.append(df_month_en[column_need[:-1]])
        except BaseException:
            print(("[I] Error! in extraction for month", month))
            #print ('traceback.format_exc():\n%s' % traceback.format_exc())
            #print ('########################################################') 
        else:
            print ("[I] Finish referenece extraction for month", month)

    # Merge to one dataframe
    df_merge = pd.concat(list_merge)

    # Save to a csv file
    df_merge.to_csv(work_path + file_name)

else:
    df_merge = pd.read_csv(work_path + file_name, index_col=0).reset_index()


# ---- Pre-processing ---- #

#def tagging_jp(text_jp):
# fixed bug under python3, fixed dictionary route
#    tagger = MeCab.Tagger("-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd/")
#    tagger.parse('') #python3 bug, must parse empty sentecse at first.
#    node = tagger.parseToNode(text_jp)
#    line_tagged=[]
#    newLine = []
#    while node:
#        word_tagged = (node.surface, node.feature)
#        line_tagged.append(word_tagged)
#        list_feature = node.feature.split(',')
#        if '動詞' in list_feature[0] or '名詞' in list_feature[0] or '接頭詞' in list_feature[0]:
#            if '数' not in list_feature[1] and '接尾' not in list_feature[1]:
#                if '*' not in list_feature[6]:
#                    newLine.append(list_feature[6])    
#        node = node.next
#
#    text_tagged_jp = ' '.join(newLine)
#    return text_tagged_jp

def clean_symbol(sentence):
    
    sentence = re.sub("http:\/\/([^/:]+)(:\d*)?([^# ]*)","",sentence)
    sentence = sentence.replace("-", " ")
    sentence = sentence.replace("=", " ")
    sentence = sentence.replace(".", "\n")
    sentence = sentence.replace("。", "\n")
    sentence = re.sub("<.*>","",sentence)
    sentence = re.sub("<.*>","",sentence)
    sentence = re.sub("[\<]*[\s]*[\^]+","",sentence)
    sentence = re.sub("\[.*\]","",sentence)
    sentence = re.sub("\(.*\)","",sentence)
    sentence = re.sub("^UPDATE[\s]*[0-9]+","UPDATE",sentence)
    sentence = re.sub("^WARPUP[\s]*[0-9]+","WARPUP",sentence)
    
    
    # Output for normal tagging
    newLine1 = sentence
    #newLine1 = re.sub(r'^[A-Z]*[\s]*[1-9]+[\s]*-', '', newLine1)
    #newLine1 = re.sub(r'^[A-Z-]*[\s]*[1-9]*[\s]*-', '', newLine1)
    # Remove punctuations and other symbols
    #newLine1 = re.sub(r'[^ \n]+_[^A-Za-z \n]+', '', newLine1)
    # Remove unknown nouns xxx_nn
    #newLine1 = re.sub(r'[^ ]*[^A-Za-z.]_[Nn][^ \n]+', '',
    #                  newLine1)  # Here is the reason that the ' are deleted!!! CODE:W1
    # Remove all the tagger notatation "_xxx"
    #newLine1 = re.sub(r'_[^ \n]+', '', newLine1)
    # Reshape the string by removing continuous space
    #newLine1 = re.sub(r' [ ]+', ' ', newLine1)

    # Remove 's
    #newLine1 = newLine1.replace("'s", "")

    # Remove '
    #newLine1 = newLine1.replace("'", "")

    return newLine1


def tagging_jp_wzh(text_jp):
    text_jp = clean_symbol(text_jp)
    text_jp = re.sub('※英文参照番号[\s\S]*','',text_jp)
    tagger = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
    #tagger = MeCab.Tagger()
    tagger.parse('')
    node = tagger.parseToNode(text_jp)
    line_tagged=[]
    newLine = []
    s = ""
    while node:
        #print(node.surface)
        word_tagged = (node.surface, node.feature)
        line_tagged.append(word_tagged)
        list_feature = node.feature.split(',')
        if '動詞' in list_feature[0] or '名詞' in list_feature[0] or '接頭詞' in list_feature[0] or '副詞' in list_feature[0] or '形容詞' in list_feature[0]:
            if '数' not in list_feature[1] and '接尾' not in list_feature[1] and '助' not in list_feature[0] and '記号' not in list_feature[0]:
                if (re.sub(r"[A-Z]*[a-z]*[1-9]*", '',node.surface) != ''):
                    if '*' not in list_feature[6]:
                        newLine.append(list_feature[6])
                        
        #print(word_tagged)
        node = node.next

    text_tagged_jp = ' '.join(newLine)
    
    return text_tagged_jp

def tagging_jp_wzh_2(text_jp):
    text_jp = clean_symbol(text_jp)
    #text_jp = re.sub('※英文参照番号[\s\S]*','',text_jp)
    tagger = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
    #tagger = MeCab.Tagger()
    tagger.parse('')
    node = tagger.parseToNode(text_jp)
    line_tagged=[]
    newLine = []
    s = ""
    while node:
        #print(node.surface)
        word_tagged = (node.surface, node.feature)
        line_tagged.append(word_tagged)
        list_feature = node.feature.split(',')
        if '動詞' in list_feature[0] or '名詞' in list_feature[0] or '接頭詞' in list_feature[0] or '副詞' in list_feature[0] or '形容詞' in list_feature[0]:
            if '数' not in list_feature[1] and '接尾' not in list_feature[1] and '助' not in list_feature[0] and '記号' not in list_feature[0]:
                if (re.sub(r"[A-Z]*[a-z]*[1-9]*", '',node.surface) != ''):
                    if '*' not in list_feature[6]:
                        newLine.append(list_feature[6])
                    else:
                        newLine.append(node.surface)
                        
        #print(word_tagged)
        node = node.next

    text_tagged_jp = ' '.join(newLine)
    
    return text_tagged_jp

def clean_tag_jp(tagged_text_jp):

    reg=[]
    reg.append(r'[ ]た[ ]*')   #When to use r'' When to use u''?
    reg.append(r'[ ]ない[ ]*')
    reg.append(r'[ ]だ[ ]*')

    for reg1 in reg:
        tagged_text_jp = re.sub(reg1, ' ', tagged_text_jp)

    return tagged_text_jp

def tagging_en(text_en):

    # Before tagging, remove unneeded parts
    #text_en = text_en.replace("-", " ")
    #text_en = text_en.replace("=", " ")
    text_en = clean_symbol(text_en)
    
    tagger = nltk.PerceptronTagger()
    text_en_tagged = ""
    text_en = text_en#.decode('utf-8','ignore') # you have to decode the line using the corresponded coding!

    word_list = nltk.word_tokenize(text_en)
    line_tagged = tagger.tag(word_list)

    for t in line_tagged:
        text_en_tagged += ('_'.join(t)+' ')
    return text_en_tagged

#from nltk.corpus import wordnet
#from nltk import word_tokenize, pos_tag
#from nltk.stem import WordNetLemmatizer


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    
    sentence = clean_symbol(sentence)

    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return res

def tagging_en_wzh(text_en):
    return ' '.join(lemmatize_sentence(text_en))

def clean_tag_en(tagged_text_en):

    wnl = WordNetLemmatizer()

    # the reference http://www.comp.leeds.ac.uk/amalgam/tagsets/upenn.html
    reg = []
    reg.append(r'[^ ]+_CD')  # mid-1890 nine-thirty forty-two one-tenth ten million 0.5
    reg.append(r'[^ ]+_DT')  # all an another any both del each either every half la many
    reg.append(r'[^ ]+_EX')  # there
    reg.append(r'[^ ]+_CC')  # & 'n and both but either et for less minus neither nor or plus so
    reg.append(r'[^ ]+_IN')  # astride among uppon whether out inside pro despite on by throughou
    reg.append(r'[^ ]+_SYM')  # % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R \* \*\* \*\*\*
    reg.append(r'[^ ]+_RP')  # aboard about across along apart around aside at away back before
    reg.append(r'[^ ]+_TO')  # to

    for reg1 in reg:
        tagged_text_en = re.sub(reg1,'',tagged_text_en)

    wordList = tagged_text_en.lower().split()

    finalList = []
    for i, w in enumerate(wordList):
        # ADD 16/1/26 Transform ADV to ADJ
        # ADD 16/1/28 ('_JJ' in w) to resovle words like "only_JJ"
        # if ('_RB' in w) or ('_JJ' in w):
        if ('_rb' in w):
            advset = w[:w.find('_')] + ".r.1"
            try:
                adj = wn.synset(advset).lemmas()[0].pertainyms()[0].name()
                w = w.replace(w, adj + '_jjr')
            except (IndexError, WordNetError):
                w = w.replace(w, w[:w.find('_')] + '_jjr')

        if ('_jjr' in w) or ('_jjs' in w):
            # newADJ=wnl.lemmatize(w[:-4], 'a')
            newADJ = wnl.lemmatize(w[:w.find('_')], 'a')
            w = w.replace(w, newADJ + '_jj')
        # print "JJR replacement,the NewList:",w,"To",newADJ

        # HERE the ('_nn' in w) is to remedy the ERROR of Tagging('weaker_NN')
        if '_nn' in w:
            old = w[:w.find('_')]
            newADJ = wnl.lemmatize(w[:w.find('_')], 'a')
            w = w.replace(w, newADJ + '_nn')
            if old != newADJ:
                #print ("NN--ADJ error: " + old + " " + newADJ)
                pass

        # CODE:W1
        # Here is a big hazard, since _p can refer to '_pos'!!which will also be converted to nn!!
        # PDT Predeterminer POS Possessive ending PRP Personal pronoun PRP$ Possessive pronoun
        # convert 'its' to 'it'
        if ('_nn' in w or '_pr' in w):
            newNoun = wnl.lemmatize(w[:w.find('_')], 'n')
            w = w.replace(w, newNoun + '_nn')

        if ('_v' in w):
            newNoun = wnl.lemmatize(w[:w.find('_')], 'v')
            w = w.replace(w, newNoun + '_vb')


        finalList.append(w)

    # Re-combine into a string and remove all the POS-tags
    newLine = " ".join(finalList)

    # Output for normal tagging
    newLine1 = newLine
    # Remove punctuations and other symbols
    newLine1 = re.sub(r'[^ \n]+_[^A-Za-z \n]+', '', newLine1)
    # Remove unknown nouns xxx_nn
    newLine1 = re.sub(r'[^ ]*[^A-Za-z.]_[Nn][^ \n]+', '',
                      newLine1)  # Here is the reason that the ' are deleted!!! CODE:W1
    # Remove all the tagger notatation "_xxx"
    newLine1 = re.sub(r'_[^ \n]+', '', newLine1)
    # Reshape the string by removing continuous space
    newLine1 = re.sub(r' [ ]+', ' ', newLine1)

    # Remove 's
    newLine1 = newLine1.replace("'s", "")

    # Remove '
    newLine1 = newLine1.replace("'", "")
    
    newLine1 = newLine1.replace("\/", " ")
    
    newLine1 = newLine1.replace("*", " ")

    return newLine1

# 一句一行的准备
def cleanjp(centences_jp):
    return([tagging_jp_wzh(line) for line in centences_jp])

def cleanen(centences_en):
    return([clean_tag_en(tagging_en(line)) for line in centences_en])



if preprocessing:
    print("=========================preprocessing============================")
    if not retrieve_from_db:
        df_merge = pd.read_csv(work_path + file_name)
            
    df_title = df_merge[["HEADLINE_ALERT_TEXT_x","HEADLINE_ALERT_TEXT_y"]]

        # --- Pre-processing for Japanese --- #
    df_title["head_tagged_jp"] = df_title["HEADLINE_ALERT_TEXT_x"].apply(tagging_jp_wzh_2)
        #df_title["head_tagged2_jp"] = df_title["head_tagged_jp"].apply(clean_tag_jp)

        # --- Pre-processing for English --- #
        #df_title["head_tagged_en"] = df_title["HEADLINE_ALERT_TEXT_y"].apply(tagging_en)
        #df_title["head_tagged2_en"] = df_title["head_tagged_en"].apply(clean_tag_en)
    df_title["head_tagged_en"] = df_title["HEADLINE_ALERT_TEXT_y"].apply(tagging_en).apply(clean_tag_en)


        #df_title[["head_tagged_en", "head_tagged_jp"]].to_csv(work_path + file_name2, encoding = "utf-8")
        #using wzh version
    df_title[["head_tagged_en", "head_tagged_jp"]].to_csv(work_path + file_name2, encoding = "utf-8",index = False)
        # df_title.to_json("test.json")
        #
        # df = pd.read_json(df_title)

if textout:
    

        
    if not retrieve_from_db:
        df_merge = pd.read_csv(work_path + file_name)
    
    if not preprocessing:
        df_title = pd.read_csv(work_path + file_name2, header = None, names=["HEADLINE_ALERT_TEXT_x","HEADLINE_ALERT_TEXT_y"])
    
    print("========================textout=========================")
    
#可以在这里split之后判断空格数量
    df_text_jp["HEADLINE_ALERT_TEXT_x"] = df_title["head_tagged_jp"]
    df_text_jp["TAKE_TEXT_x"] = df_merge["TAKE_TEXT_x"].apply(tagging_jp_wzh)
    #df_text_jp["TAKE_TEXT_x"] = df_merge["TAKE_TEXT_x"].str.split('。').apply(cleanjp) #不需要每行分句子
    #df_text_jp["TAKE_TEXT_x"] = [tagging_jp_wzh(line) for line in df_text_jp["TAKE_TEXT_x"]]
    df_text_jp.to_csv(work_path + file_name_text_jp ,sep='\n', encoding = "utf-8", index=False, header=False)
    
    df_text_en["HEADLINE_ALERT_TEXT_y"] = df_title["head_tagged_en"]
    df_text_en["TAKE_TEXT_y"] = df_merge["TAKE_TEXT_y"].apply(tagging_en).apply(clean_tag_en)
    #df_text_en["TAKE_TEXT_y"] = df_merge["TAKE_TEXT_y"].str.split('.').apply(cleanen)# 不需要每行分句子
    #df_text_en["TAKE_TEXT_y"] = [clean_tag_en(tagging_en(line)) for line in df_text_en["TAKE_TEXT_y"]]
    df_text_en.to_csv(work_path + file_name_text_en ,sep='\n', encoding = "utf-8", index=False, header=False)
    

    
    