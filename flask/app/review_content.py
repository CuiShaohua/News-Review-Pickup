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
# 从single_sents读句子也可以
def sentence_read():
    
    # 0 
    #Input = open("./single_sents", "r", encoding="utf-8")
    # 1 利用sentencesplitter再分一遍句子
    with open("./article_corpus", 'r', encoding="utf-8") as fo:
        pass

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


# ltp_process，只分析SBV类型的句子。输出SBV主语、谓语“说”和说话内容
def ltp_process(sentence):
    stop_words = get_stops_words()  # 提取停用词，为SBV词【如“是”等停用词】在SBV词中删除。
    
    # 分词
    segmentor = pyltp.Segmentor()
    segmentor.load("./cws.model")
    words = segmentor.segment(sentence)
    print("\t".join(words))
    segmentor.release()
    
    # 词性
    postagger = pyltp.Postagger()
    postagger.load("./pos.model")
    postags = postagger.postag(words)
    # list-of-string parameter is support in 0.1.5
    # postags = postagger.postag(["中国","进出口","银行","与","中国银行","加强","合作"])
    print("\t".join(postags))
    postagger.release()

    # 依存句法分析
    parser = pyltp.Parser()
    parser.load("./parser.model")
    arcs = parser.parse(words, postags)
    parser.release()

    # 角色分析，暂时没用上
    # 拿到前面来是有用意的，在进行判断了当前的SBV的子节点与"说"有关后，需要抽取这个词，简而言之，是SBV，又对应A0，则这个词一定是主语。
    labeller = pyltp.SementicRoleLabeller()
    labeller.load("./pisrl.model")
    roles = labeller.label(words, postags, arcs)

    print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    
    SBV_set = list()
    Subject_label_set = list()
    Word_of_speech_content = list()

    Index_of_Subjet = 0

    for arc in arcs:
        #SBV_index = get_repeat(arc.head, "SBV")
        k = Index_of_Subjet
        if arc.relation == "SBV" and words[arc.head - 1] not in stop_words:  # 这个地方难道真的不够严谨，不能只判断是不是SBV，因为一旦判断有SBV了，那么必然这个词就是A0

            SBV_set.append(words[arc.head - 1])  # arc.head是从1开始计数，存储SBV指向的谓语动词
            Subject_label_set.append(words[Index_of_Subjet])  # 如果有SBV，那么这个词对应的位置肯定是主语
            Word_of_speech_content.append(words[arc.head:])  # 拿出来的相当于SBV主语词以后的部分。

            Index_of_Subjet += 1

        else:
            Index_of_Subjet += 1
            continue  # 如果为空列表，该句子没有分析的必要性
        


    '''
    recognizer = pyltp.NamedEntityRecognizer()
    recognizer.load("./ner.model")
    netags = recognizer.recognize(words, postags)
    print("\t".join(netags))
    
    labeller = pyltp.SementicRoleLabeller()
    labeller.load("./pisrl.model")
    roles = labeller.label(words, postags, arcs)

    for role in roles:
        print(role.index, "".join(
                ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    '''

    return SBV_set, Subject_label_set, Word_of_speech_content  # 返回的是一个列表，第一个值是SBV的子节点词（HED），第二个是当前SBV的主语。一定要注意，是不是都是[]


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

    #句子成分分析结果

    result_of_data = defaultdict(list)

    return result_of_data


import gensim as gm

# verb_speak是一个包含SBV的列表，非一个词
def is_there_speak(Verb_and_Subject_result_of_ltp_process):

    # 1 投入SBV的词进行检测，看下是不是位于词的列表中。
    # 1.1 手动写的这些
    manl_add_speak = ['讨论', '交流', '讲话', '说话', '吵架', '争吵', '勉励', '告诫', '戏言', '嗫嚅', '沟通', '切磋', '争论', '研究', '辩解', '坦言', '喧哗',      '嚷嚷', '吵吵', '喊叫', '呼唤', '讥讽', '咒骂', '漫骂', '七嘴八舌', '滔滔不绝', '口若悬河', '侃侃而谈', '念念有词', '喋喋不休', '说话',      '谈话', '讲话', '叙述', '陈述', '复述', '申述', '说明', '声明', '讲明', '谈论', '辩论', '议论', '讨论', '商谈', '哈谈', '商量', '畅谈',      '商讨', '话语', '呓语', '梦话', '怪话', '黑话', '谣言', '谰言', '恶言', '狂言', '流言', '巧言', '忠言', '疾言', '傻话', '胡话', '俗话', '废话', '发言', '婉言', '危言', '谎言', '直言', '预言', '诺言', '诤言']
    # 1.2 迭代10次加入的这些
    gensim_speak_list =['说', '却说', '答道', '回答', '问起', '说出', '追问', '干什么', '质问', '不在乎', '没错', '请问', '问及', '问到', '感叹', '告诫', '说谎', '说完', '开玩笑', '讲出', '直言', '想想', '想到', '问说', '责骂', '讥笑', '问过', '转述', '闭嘴', '撒谎', '谈到', '有没有', '骂', '责怪', '取笑', '吹嘘', '听过', '不该', '询问', '坚称', '怒斥', '答', '该死', '忘掉', '说起', '反问', '问道', '忍不住', '苦笑', '没事', '道谢', '问', '对不起', '戏弄', '埋怨', '发脾气', '说道', '原谅', '责备', '大笑', '听罢', '放过', '不解', '没想', '训斥', '认错', '想来', '自嘲', '还好', '打趣', '嘲笑', '后悔', '讥讽', '发怒', '心疼', '发牢骚', '数落', '点头', '认得', '醒悟', '装作', '要死', '流泪', '哭诉', '走开', '动怒', '不屑', '悔恨', '咒骂', '捉弄']
    speak_list = manl_add_speak + gensim_speak_list

    final_speak = list()
    final_label = list()
    final_speech_content = list()
    #tmp_SBV_label_set = Verb_and_Subject_result_of_ltp_process

    for speak, label, speech_content in Verb_and_Subject_result_of_ltp_process:
        if speak in speak_list:

            final_speak.append(speak)
            final_label.append(label)
            final_speech_content.append(speech_content)
            
        else:
            # # 2 如果不在列表中，采用gensim.most_similar()进行判断，相似性大于0.4，认为就是说的同义词
            logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
            mode = gm.models.Word2Vec.load("./merge_corpus.model")
            if mode.wv.similarity('说', speak) > 0.4:
                print(speak, mode.wv.similarity('说', speak))
                
                final_speak.append(speak)
                final_label.append(label)
                final_speech_content.append(speech_content)

            else:
                
                continue
    print(final_speak, final_label, final_speech_content)
    return final_speak, final_label, final_speech_content

# 把停用词再删除
def get_stops_words():
    with open("./stopwords.txt", "r") as fo:

        stop_words = fo.read().split('\n')

    return stop_words

def refine_speech_content():

    # 01 如果只有引号能把所有内容包括，则把引号里的内容拿出来
    # 02 如果没有引号，则按逗号进行拆分，并判断是不是一句话。
    pass



# 主函数
def review_fc(para):

    #0 输入一个句子或者一个段落
    paragraph = "党的十八大以来，习近平总书记一直牵挂着广大人民群众的身体健康和用药安全，他强调：要始终把人民群众的身体健康放在首位。他指出，“要密切监测药品短缺情况，采取有效措施，解决好低价药、‘救命药’、‘孤儿药’以及儿童用药的供应问题。"
    paragraph = para
    result_of_data = pd_DataFram() # 空defaultdict
    #0.1 判断句子是否可以分句
    cut_result = cut_sent(paragraph)
    #result_of_data["sentence"] = cut_result

    print(result_of_data["sentence"])   
    if cut_result:
        for sent in cut_result:  # 一个句子一个句子分析
            #0.2 判断这个句子是不是含有SBV的成分
            ltp_result = ltp_process(sent)  # 返回的已经是words[“SVB”]和SBV对应的主语。
            verb_of_ltp_result, label_of_ltp_result, Word_of_speech_content = ltp_result
            if verb_of_ltp_result:  #只要verb_of_ltp_result不为空，就进行判断说，否则该句子就没有说
                (sp_list, label_list, speech_list) = is_there_speak(zip(verb_of_ltp_result, label_of_ltp_result, Word_of_speech_content))
                
                if sp_list != None:

                    result_of_data["SBV"].append(sp_list)
                    result_of_data["Person"].append(label_list)
                    result_of_data["speech_content"].append([''.join(k) for k in speech_list if k != '，'])
                
                else:
                    
                    result_of_data["sentence"].pop()

                    print("句子不含有动词说")

            else:
                result_of_data["sentence"].pop()
                print("该句子不是SBV类型")  # 不是SBV类型可以再加一部分判断，按照袁禾那种从动词verb出发，判断是不是说
                #可以调袁禾的接口去做

    result_of_data["sentence"] = cut_result
    result_all = pd.DataFrame(result_of_data)

    return result_all

def main():
    main_fuc()

if __name__ == "__main__":

    print(main)
