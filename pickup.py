#!/usr/bin/ python
# -*- coding: utf-8 -*-
import sys, os
from zhon.hanzi import punctuation
from zhon.hanzi import non_stops
from zhon.hanzi import stops
import logging

zhon_char = punctuation + non_stops + stops
import pyltp

import gc
import re
import numpy as np

def tlist_to_dkeys(a):
    return np.transpose(np.array(a), axes=[1,0])[0]

# 中文文本分句，返回值为字符串组成的列表
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def get_stops_words():
    with open("./data/stopwords.txt", "r", encoding='utf-8') as fo:
        stop_words = fo.read().split('\n')

    return stop_words

def intro_speech(sentence):

    sentence = sentence.strip("：“”")
    sentence = sentence.strip("，“”")
    sentence = sentence.strip('“”')
    return sentence


 # focus_point_set
class Si:

    def __init__(self, words, postags, old_SI = {}):

        self.words = words # 名词及索引

        self.postags = postags # 主语、宾语、辅助名词及索引
        self.params = {}  # {'word':'score'}
        self.old_SI = old_SI

    def score(self):
        # 根据名词和索引及主宾辅的索引情况进行归类并且赋值初始分数
        weight_adment = []
        for i,j in self.postags.items():
            # i代表["SBV",'VOB'],j代表索引
            # 加入一个权重的修正，将名词词组的得分提高，如score(习近平) = score(总书记)

            if i=='SBV':
                for k in j:
                    if k[0] in self.words.keys() and self.words[k[0]] == k[1]:
                        self.params[k[0]] = 5
                        weight_adment.append(k[1])
                    else:
                        pass
            elif i=='VOB':
                for m in j:
                    if m[0] in self.words.keys() and self.words[m[0]] == m[1]:
                        self.params[m[0]] = 2
                    else:
                        pass
            else:
                self.params[i] = 1# "SBV":(词，初始分数)

        for word, index in self.words.items():
            if word not in self.params.keys():
                self.params[word] = 1
        # 修正
        #print(weight_adment)
        for f, k in self.words.items():
            if k in (np.array(weight_adment) + 1).tolist() or k in (np.array(weight_adment) - 1).tolist():
                self.params[f] = 5

        return self.params


    def update(self):
        # 先调用一次score，获得当前的params
        params = self.score()
        # 比较当前postags和之前的postags
        if self.old_SI:
            for i, j in self.old_SI.params.items():
                if i not in list(params.keys()) and j>=1:
                    j -= 1  # 进行简化，衰减程度都变成1
                    self.old_SI.params[i] = j

                if j <= 0:
                    self.old_SI.params[i] = 0
        #print('update', self.old_SI.params)
        return self.old_SI.params

def ltp_process(sentence, old_SI={}):

    stop_words = get_stops_words()  # 提取停用词，为SBV词【如“是”等停用词】在SBV词中删除。

    # 分词
    segmentor = pyltp.Segmentor()
    segmentor.load("./model/cws.model")
    words = segmentor.segment(sentence)
    #print("\t".join(words))
    segmentor.release()

    # 词性
    postagger = pyltp.Postagger()
    postagger.load("./model/pos.model")
    postags = postagger.postag(words)
    # list-of-string parameter is support in 0.1.5
    #print("\t".join(postags))
    postagger.release()

    # 依存句法分析
    parser = pyltp.Parser()
    parser.load("./model/parser.model")
    arcs = parser.parse(words, postags)
    parser.release()

    # 拿到前面来是有用意的，在进行判断了当前的SBV的子节点与"说"有关后，需要抽取这个词，简而言之，是SBV，又对应A0，则这个词一定是主语。
    labeller = pyltp.SementicRoleLabeller()
    labeller.load("./model/pisrl.model")
    roles = labeller.label(words, postags, arcs)

    #print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)) # 依存句法分析

    noun_tags = ['nh','nd','n','ni','nl','ns','nt','nz']
    SI_words = {}  # 词和索引
    for tag in noun_tags:
        SI_index = np.argwhere(np.array(postags)==tag).reshape(-1).tolist()
        for j in SI_index:
            SI_words[words[j]] = j

    #print(SI_words)

    SBV_set = list()
    Subject_label_set = list()
    Word_of_speech_content = list()

    Index_of_Subjet = 0
    SI_postags = {}
    si_SBV_postag = []
    si_VOB_postag = []
    for arc in arcs:
        # SBV_index = get_repeat(arc.head, "SBV")
        k = Index_of_Subjet
        if arc.relation == "SBV" and words[arc.head - 1] not in stop_words:  # 这个地方难道真的不够严谨，不能只判断是不是SBV，因为一旦判断有SBV了，那么必然这个词就是A0
            #print(arc.head, words[arc.head -1])
            SBV_set.append(words[arc.head - 1])  # arc.head是从1开始计数，存储SBV指向的谓语动词
            # 加入主语的判断
            if words[Index_of_Subjet] in ['他','他们','你','你们','我','我们', '她','她们']:
                # 进行指代消解
                # 查看当前old_SI，如果old_SI中有相同角色，取积分最高值进行替换人称代词。需要做一次修正，名词词组如习近平+总书记应该是一个词，或者把习近平的权重设置为总书记一样
                if old_SI:
                    ag2entity = np.argmax(old_SI.params.keys())
                    
                    words[Index_of_Subjet] = list(old_SI.params.keys())[ag2entity]

                else:
                    pass

                Subject_label_set.append(words[Index_of_Subjet])

            else:
                Subject_label_set.append(words[Index_of_Subjet])  # 如果不是指示代词，那么这个词对应的位置肯定是主语
                #SI_postag[words[Index_of_Subjet].split(':')[1]] = Index_of_Subjet
                if postags[arc.head -1] == 'v':
                    si_SBV_postag.append((words[Index_of_Subjet], Index_of_Subjet))

            Word_of_speech_content.append(intro_speech(''.join(words[arc.head:])))  # 拿出所说内容。
            #print(intro_speech(''.join(words[arc.head:])))
            Index_of_Subjet += 1
            SI_postags[arc.relation] = si_SBV_postag



        elif arc.relation == 'VOB' and words[arc.head -1] not in stop_words:
            # 加入宾语的判断
            if words[Index_of_Subjet] in ['他','他们','你','你们','我','我们', '她','她们']:
                # 进行指代消解
                # 引入前一句的宾语位置和积分最高元素
                pass

            else:
                Subject_label_set.append(words[Index_of_Subjet])  # 如果不是指示代词，那么这个词对应的位置肯定是主语
                

                si_VOB_postag.append((words[Index_of_Subjet], Index_of_Subjet))

            Index_of_Subjet += 1
            SI_postags[arc.relation] = si_VOB_postag

        else:
            Index_of_Subjet += 1
            continue  # 如果为空列表，该句子没有分析的必要性

    Forcus_point = Si(SI_words, SI_postags,old_SI) # 关注焦点集
    # 需要更新self.params
    Forcus_point.score()

    recognizer = pyltp.NamedEntityRecognizer()
    recognizer.load("./model/ner.model")
    netags = recognizer.recognize(words, postags)
    #print("\t".join(netags))
    '''
    labeller = pyltp.SementicRoleLabeller()
    labeller.load("./pisrl.model")
    roles = labeller.label(words, postags, arcs)

    for role in roles:
        print(role.index, "".join(
                ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    '''
    return SBV_set, Subject_label_set, Word_of_speech_content, Forcus_point  # 返回的是一个列表，第一个值是SBV的子节点词（HED），第二个是当前SBV的主语，结果有可能是空。


import pandas as pd
from collections import defaultdict


def pd_DataFram():
    '''
    DataFram 结构是这样的

               列索引
                 0
              分句结果           SVB                                说话人                    说话内容
    行索引 0    句子0    句子0的SVB结果[SVB词的索引列表]      [per1, per2, per3,...]      [speech1, speech2, speech3]
          1    句子1     句子1的SVB结果[SVB次的索引列表]      [per1, per2, per3,...]      [speech1, speech2, speech3]
          2     .       ...
          3     .
    '''

    # 句子成分分析结果

    result_of_data = defaultdict(list)

    return result_of_data

import pickle as pkl
import gensim as gm


# verb_speak是一个包含SBV的列表，非一个词
def is_there_speak(Verb_and_Subject_result_of_ltp_process):
    # 1 投入SBV的词进行检测，看下是不是位于词的列表中。
    # 1.1 手动写的这些
    manl_add_speak = ['讨论', '交流', '讲话', '说话', '吵架', '争吵', '勉励', '告诫', '戏言', '嗫嚅', '沟通', '切磋', '争论', '研究', '辩解', '坦言',
                      '喧哗', '嚷嚷', '吵吵', '喊叫', '呼唤', '讥讽', '咒骂', '漫骂', '七嘴八舌', '滔滔不绝', '口若悬河', '侃侃而谈', '念念有词', '喋喋不休',
                      '说话', '谈话', '讲话', '叙述', '陈述', '复述', '申述', '说明', '声明', '讲明', '谈论', '辩论', '议论', '讨论', '商谈', '哈谈',
                      '商量', '畅谈', '商讨', '话语', '呓语', '梦话', '怪话', '黑话', '谣言', '谰言', '恶言', '狂言', '流言', '巧言', '忠言', '疾言',
                      '傻话', '胡话', '俗话', '废话', '发言', '婉言', '危言', '谎言', '直言', '预言', '诺言', '诤言']
    # 1.2 迭代10次加入的这些
    gensim_speak_list = ['说', '却说', '答道', '回答', '问起', '说出', '追问', '干什么', '质问', '不在乎', '没错', '请问', '问及', '问到', '感叹',
                         '告诫', '说谎', '说完', '开玩笑', '讲出', '直言', '想想', '想到', '问说', '责骂', '讥笑', '问过', '转述', '闭嘴', '撒谎',
                         '谈到', '有没有', '骂', '责怪', '取笑', '吹嘘', '听过', '不该', '询问', '坚称', '怒斥', '答', '该死', '忘掉', '说起', '反问',
                         '问道', '忍不住', '苦笑', '没事', '道谢', '问', '对不起', '戏弄', '埋怨', '发脾气', '说道', '原谅', '责备', '大笑', '听罢',
                         '放过', '不解', '没想', '训斥', '认错', '想来', '自嘲', '还好', '打趣', '嘲笑', '后悔', '讥讽', '发怒', '心疼', '发牢骚',
                         '数落', '点头', '认得', '醒悟', '装作', '要死', '流泪', '哭诉', '走开', '动怒', '不屑', '悔恨', '咒骂', '捉弄']
    speak_list = manl_add_speak + gensim_speak_list

    final_speak = list()
    final_label = list()
    final_speech_content = list()
    # tmp_SBV_label_set = Verb_and_Subject_result_of_ltp_process

    for speak, label, speech_content in Verb_and_Subject_result_of_ltp_process:
        if speak in speak_list:

            final_speak.append(speak)
            final_label.append(label)
            final_speech_content.append(speech_content)

        else:
            # # 2 如果不在列表中，采用gensim.most_similar()进行判断，相似性大于0.4，认为就是说的同义词
            logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
            fi = open('./model/new.model','rb')
            model = pkl.load(fi)
            fi.close()
            try:
                if model.similarity('说', speak) > 0.35:

                    #print(speak, model.similarity('说', speak))

                    final_speak.append(speak)
                    final_label.append(label)
                    final_speech_content.append(speech_content)

                else:

                    continue
            except KeyError:
                continue
    #print(final_speak, final_label, final_speech_content)
    if final_speak:
        return final_speak, final_label, final_speech_content
    else:
        return None, None, None


# 主函数
def main(para):
    # 0 输入一个句子或者一个段落
    #paragraph = '党的十八大以来，习近平总书记一直牵挂着我们广大人民群众的身体健康和用药安全。'
    #paragraph = "党的十八大以来，习近平总书记一直牵挂着广大人民群众的身体健康和用药安全，他强调：要始终把人民群众的身体健康放在首位。他指出，“要密切监测药品短缺情况，采取有效措施，解决好低价药、‘救命药’、‘孤儿药’以及儿童用药的供应问题。"
    #paragraph = "党的十八大以来，习近平总书记一直牵挂着我们广大人民群众的身体健康和用药安全。他强调：“要始终把人民群众的身体健康放在首位。他指出，“要密切监测药品短缺情况，采取有效措施，解决好低价药、‘救命药’、‘孤儿药’以及儿童用药的供应问题。”"
    #paragraph = "蒋丽芸说，自己曾经与年轻人交谈过，很多人都不想“港独”，并表示支持“一国两制”，但经常会有人将“独立”加诸年轻人口中。她提到，有民调亦显示绝大多数年轻人是不支持“港独”的。"
    paragraph = para
    result_of_data = pd_DataFram()  # 空defaultdict
    # 0.1 判断句子是否可以分句
    cut_result = cut_sent(paragraph)
    # result_of_data["sentence"] = cut_result

    #print(result_of_data["sentence"])
    if cut_result:

        # 第一句特殊，拿出来单独计算关注焦点集
        first_ltp_result = ltp_process(cut_result[0], old_SI={})
        fist_sent_verb, first_label, first_speech_cont, first_SI = first_ltp_result
        global old_SI
        old_SI = first_SI
        #print(1, old_SI.params)
        if fist_sent_verb:
            (first_sk_verb, first_sk_label, fisrt_sk_content) = is_there_speak(zip(fist_sent_verb, first_label, first_speech_cont))

            if first_sk_verb:

                result_of_data["SBV"].append(first_sk_verb)
                result_of_data["Person"].append(first_sk_label)
                result_of_data["speech_content"].append(fisrt_sk_content)


        # 第二句开始
        for sent in cut_result[1:]:  # 一个句子一个句子分析
            #refine_speech_content(sent)
            # 0.2 判断这个句子是不是含有SBV的成分
            #print(type(old_SI))
            #print(old_SI.params)
            ltp_result = ltp_process(sent, old_SI=old_SI)  # 返回的已经是words[“SVB”]和SBV对应的主语。
            verb_of_ltp_result, label_of_ltp_result, Word_of_speech_content, SI_result = ltp_result
            old_SI = SI_result  # 指代消歧的继续
            if verb_of_ltp_result:  # 只要verb_of_ltp_result不为空，就进行判断说，否则该句子就没有说
                (sp_list, label_list, speech_list) = is_there_speak(
                    zip(verb_of_ltp_result, label_of_ltp_result, Word_of_speech_content))

                if sp_list:

                    result_of_data["SBV"].append(sp_list)
                    result_of_data["Person"].append(label_list)
                    result_of_data["speech_content"].append(speech_list)

            else:
                result_of_data["sentence"].pop()
                #print("警告：该句子不是SBV类型")  # 不是SBV类型可以再加一部分判断
            old_SI.params = old_SI.update()
            #print(2, old_SI.params)
    #result_of_data["sentence"] = cut_result
    result_all = pd.DataFrame(result_of_data)
    #result_data = result_all.drop(columns=['sentence'])
    if result_all.empty is False:
        Index = list(range(1, len(cut_result)+1))
        OUP = list(zip(Index,result_all['Person'],result_all['SBV'],result_all['speech_content']))
        return OUP
    else:
        OUP = ""
        return OUP
    #result_data = result_data[['Index','Person','SBV','speech_content']]


if __name__ == "__main__":
    print(main())
