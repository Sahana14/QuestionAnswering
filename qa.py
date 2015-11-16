import sys
import os, glob
import string
import nltk
import re
import textwrap
import copy
import numpy
nltk.data.path.append("/home/sandeep/nltk_data")
from nltk.corpus import state_union
#java_path = "C:/Program Files/Java/jdk1.8.0_60/bin/java.exe"
#os.environ['JAVAHOME'] = java_path
#nltk.internals.config_java("C:/Program Files/Java/jdk1.8.0_60/bin/java.exe")
from nltk.corpus import stopwords
#from nltk.tag.stanford import StanfordNERTagger
#st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
from nltk.stem import PorterStemmer
ps = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
STOPWORDS = [lemmatizer.lemmatize(t) for t in stopwords.words('english')]
quoted = re.compile('"([^"]*)"')
#with open(sys.argv[1], 'r') as f:
   # contents = f.readlines()
ques_words  = ["where", "when", "what", "how", "why", "who", "whose", "which"]
ans_type = {"where":"location", "who":"person", "what":"organization", "how jj":"numeric", "when":"date time numeric",
            "why":"description", "how mod":"description", "how": "description", "whose":"person",
            "which":"location", "how long": "numeric", "how far": "numeric", "how much": "numeric",
            "how old": "numeric", "how often": "numeric"} #"what np":"entity"

MONTH = [ "january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

TIME = ["nowadays", "these days","lately","recently","last night","last week","last month",
        "last year","ago","since","previously","recently","just","in a week","in a moment",
        "in 3 days","next week","next month","next year","next summer","soon","presently",
        "in the end","eventually","at the end","finally","once","by the time","after","before",
        "for","on time","until","during","whenever","forever","by friday","by next week",
        "by the 7th of","immediately","january", "february", "march", "april", "may", "june",
        "july", "august","september", "october", "november", "december"]

LOCATION = ["Afghanistan","Albania","Algeria","American Samoa","Andorra","Angola","Anguilla","Antarctica","Antigua and Barbuda","Argentina",
"Armenia","Aruba","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados","Belarus","Belgium","Belize","Benin",
"Bermuda","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana","Brazil","Brunei Darussalam","Bulgaria","Burkina Faso","Burundi",
"Cambodia","Cameroon","Canada","Cape Verde","Cayman Islands","Central African Republic","Chad","Chile","China","Christmas Island",
"Cocos (Keeling) Islands","Colombia","Comoros","Democratic Republic of the Congo (Kinshasa)","Congo, Republic of(Brazzaville)",
"Cook Islands","Costa Rica","Ivory Coast","Croatia","Cuba","Cyprus","Czech Republic","Denmark","Djibouti","Dominica","Dominican Republic",
"East Timor (Timor-Leste)","Ecuador","Egypt","El Salvador","Equatorial Guinea","Eritrea","Estonia","Ethiopia","Falkland Islands","Faroe Islands",
"Fiji","Finland","France","French Guiana","French Polynesia","French Southern Territories","Gabon","Gambia","Georgia","Germany","Ghana","Gibraltar",
"Great Britain","Greece","Greenland","Grenada","Guadeloupe","Guam","Guatemala","Guinea","Guinea-Bissau","Guyana","","Haiti","Holy See","Honduras","Hong Kong",
"Hungary","Iceland","India","Indonesia","Iran (Islamic Republic of)","Iraq","Ireland","Israel","Italy","","Jamaica","Japan","Jordan","","","Kazakhstan",
"Kenya","Kiribati","Korea, Democratic People's Rep. (North Korea)","Korea, Republic of (South Korea)","Kosovo","","Kuwait","Kyrgyzstan","",
"Lao, People's Democratic Republic","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania","Luxembourg","Macau","Macedonia, Rep. of",
"Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Marshall Islands","Martinique","Mauritania","Mauritius","Mayotte","Mexico",
"Micronesia, Federal States of","Moldova, Republic of","Monaco","Mongolia","Montenegro","Montserrat","Morocco","Mozambique","Myanmar, Burma","Namibia",
"Nauru","Nepal","Netherlands","Netherlands Antilles","New Caledonia","New Zealand","Nicaragua","Niger","Nigeria","Niue","Northern Mariana Islands","Norway",
"Oman","Pakistan","Palau","Palestinian territories","Panama","Papua New Guinea","Paraguay","Peru","Philippines","Pitcairn Island","Poland","Portugal","Puerto Rico",
"Qatar","Reunion Island","Romania","Russian Federation","Rwanda","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Samoa","San Marino",
"Sao Tome and Principe","Saudi Arabia","Senegal","Serbia","Seychelles","Sierra Leone","Singapore","Slovakia (Slovak Republic)","Slovenia","Solomon Islands","Somalia",
"South Africa","South Sudan","Spain","Sri Lanka","Sudan","Suriname","Swaziland","Sweden","Switzerland","Syria, Syrian Arab Republic","Taiwan (Republic of China)",
"Tajikistan","Tanzania; officially the United Republic of Tanzania","Thailand","Tibet","Timor-Leste (East Timor)","Togo","Tokelau","Tonga","Trinidad and Tobago",
"Tunisia","Turkey","Turkmenistan","Turks and Caicos Islands","Tuvalu","Ugandax","Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay","Uzbekistan",
"Vanuatu","Vatican City State (Holy See)","Venezuela","Vietnam","Virgin Islands (British)","Virgin Islands (U.S.)","Wallis and Futuna Islands","Western Sahara","Yemen",
"Zambia","Zimbabwe","Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois Indiana","Iowa",
"Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana Nebraska","Nevada","New Hampshire","New Jersey",
"New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
"Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]

clue = 3
good_clue = 4
confident = 6
slam_dunk = 20

locPrep = ["in", "outside", "on", "between", "at", "beside", "by", "beyond", "near", "in front of", "nearby", "in back of", "above", "behind", "below", "next to", "over", "on top of", "under", "within", "up", "beneath", "down", "underneath", "around", "among", "through", "along", "inside", "against"]
def remstwords(str):
    t_str = copy.deepcopy(str)
    for w in t_str:
        word_lm = lemmatizer.lemmatize(w[0].lower())
        if word_lm in STOPWORDS or string.punctuation:
            str.remove(w)
    return str

def checkStopWord(word):
    word_lm = lemmatizer.lemmatize(word.lower())
    if word_lm in STOPWORDS:
        return True
    return False

def PosTag(sent):
    sent_tag = nltk.pos_tag(nltk.word_tokenize(sent))
    return sent_tag

def WordMatch(ques, sent):
    score = 0
    ques_tag = PosTag(ques)
    for q in ques_tag:
        if not checkStopWord(q[0]):
            if q[1] == "VB" or q[1] == "VBD" or q[1] == "VBG" or q[1] == "VBN" or q[1] == "VBP" or q[1] == "VBZ":
                w = ps.stem(q[0].lower())
                for s in sent.split():
                    s_lm = ps.stem(s.lower())
                    if w == s_lm:
                        score = score + 6
            else:
                w = ps.stem(q[0].lower())
                for s in sent.split():
                    s_lm = ps.stem(s.lower())
                    if w == s_lm:
                        score = score + 3
    return score

def containsNER(q,category):
    q_tokens  = nltk.word_tokenize(q)
    q_tag = nltk.pos_tag(q_tokens)
    ne_tag = nltk.ne_chunk(q_tag)
    for tree in ne_tag.subtrees():
        if tree.label() == category:
            return True
    return False

def contains(L,string):
    for q in L.split():
        if q == string:
            return True
    return False

def containsPOSTag(L,string):
    L_tag = PosTag(L)
    s = [True if x[1] == string else False for x in L_tag]
    return s

def containsList(L,lt):
    for q in L.split():
        if q in lt:
            return True
    return False

def containsList_lemma(L,lt):
    q_lm = ps.stem(L.lower())
    for q in q_lm.split():
        if q in lt:
            return True
    return False

def containsPP(q,string):
    q_tag = PosTag(q)
    for i,w1 in enumerate(q.split()):
        if w1 == string:
            if q_tag[i+1][0] == "IN":
                return True
    return False

def containsNPwithPP(s):
    q_tag = PosTag(s)
    proper_noun = [x[0] for x in q_tag if x[1] == "NNP" or x[1] == "NNPS"]
    containsPP(s,proper_noun)

def whoRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if (not containsNER(q,"PERSON")) and containsNER(s,"PERSON"):
        score = score + confident
    if (not containsNER(q,"PERSON")) and contains(s,"name"):
        score = score + good_clue
    if containsPOSTag(s,"NNP") or containsPOSTag(s, "NNPS"):
        score= score + good_clue
    return score

def whatRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if containsList(q, MONTH) and containsList(s, ["today", "yesterday", "tomorrow", "last night"]):
        score = score + clue
    if contains(q,"kind") and containsList_lemma(s, ["call", "from"]):
        score = score + good_clue
    if contains(q,"name") and containsList_lemma(s, ["name", "call", "known"]):
        score = score + slam_dunk
    if containsPP(q, "name") and containsNPwithPP(s):
        score =  score + slam_dunk
    return score

def whenRule(q,s):
    score = 0
    year_list = []
    for i in range(1400,2000):
        year_list.append(str(i))
    if containsList(s, TIME) or containsList(s, year_list) :
        score = score + good_clue
        score = score + WordMatch(q,s)
    if contains(q,"the last") and containsList(s, ["first", "last", "since", "ago"]):
        score = score + slam_dunk
    if containsList(q,["start", "begin"]) and containsList(s, ["start", "begin", "since", "year"]):
        score = score + slam_dunk
    return score

def whereRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if containsList(s, locPrep):
        score = score + good_clue
    if containsNER(s, "LOCATION") or containsList(s, LOCATION):
        score = score + confident
    return score

def whyRule(q,s, best, prev_sent, next_sent):
    score = 0
    for b_sent in best:
        if s == b_sent:#doubt if in works
            score = score + good_clue
        if  next_sent == b_sent:
            score = score + clue
        if prev_sent == b_sent:
            score = score + clue
    if contains(s, "want"):
        score = score + good_clue
    if containsList(s, ["so", "because"]):
        score = score + good_clue
    return score

def datelineRule(q):
    score = 0
    if contains(q, "happen"):
        score = score + good_clue
    if contains(q, "take") and contains(q, "place"):
        score = score + good_clue
    if contains(q, "this"):
        score = score + slam_dunk
    if contains(q, "story"):
        score = score + slam_dunk
    return score

os.chdir(sys.argv[1])
o = open('../out.txt', 'w')
files = glob.glob('*.questions')
for file in files:
    fileId = file.title().split('.')[0]
    ques_list = []
    qid_list = []
    with open(file, 'r') as g:
        ques_lines = g.readlines()
        for line in ques_lines:
            if line.__contains__("QuestionID:"):
                qid_list.append(line)
            if line.__contains__("Question:"):
                ques = line.split("Question:")[1]
                ques_list.append(ques)
    with open(fileId + '.story', 'r') as f:
        story_text = f.read()
        header = story_text.split("TEXT:")[0]
        date = None
        if header.__contains__("DATE: "):
            dateline = header.split("DATE:")[1]
            date = dateline.split("\n")[0]
        #print date
        sent_list = nltk.sent_tokenize(story_text.split("TEXT:")[1])
    for z,q in enumerate(ques_list):
        question_type = ""
        for x in reversed(q.split()):
            if x.lower() == "who":
                question_type="who"
                break
            if x.lower() == "what":
                question_type="what"
                break
            if x.lower() == "when":
                question_type="when"
                break
            if x.lower() == "where":
                question_type="where"
                break
            if x.lower() == "why":
                question_type="why"
                break
        list1 = []
        dateline_score = 0
        if question_type=="who":
            for s in sent_list:
                score = whoRule(q,s)
                list1.append((s,score))
        elif question_type=="what":
            for s in sent_list:
                score = whatRule(q,s)
                list1.append((s,score))
        elif question_type=="when":
            dateline_score = datelineRule(q)
            for s in sent_list:
                score = whenRule(q,s)
                list1.append((s,score))
        elif question_type=="where":
            dateline_score = datelineRule(q)
            for s in sent_list:
                score = whereRule(q,s)
                list1.append((s,score))
        elif question_type=="why":
            score = []
            best = []
            for k,s in enumerate(sent_list):
                score.append(WordMatch(q,s))
                best.append((s, score[k]))
            best_list = sorted(best, key=lambda  x: (-x[1],x[0]))
            sent_best = []
            for j in range(10):
                sent_best.append(best_list[j][0])
            for i,s in enumerate(sent_list):
                if i == 0:
                    score = whyRule(q,s,sent_best,None,sent_list[i+1])
                elif  i == len(sent_list) - 1:
                    score = whyRule(q,s,sent_best,sent_list[i-1],None)
                else:
                    score = whyRule(q,s,sent_best,sent_list[i-1],sent_list[i+1])
                list1.append((s,score))
        else:
            for s in sent_list:
                score = WordMatch(q,s)
                list1.append((s,score))
        max_s = max(list1, key=lambda x:x[1])
        s = ""
        for x in list1:
            if question_type == "why":
                if x[1] == max_s[1]:
                    s = x[0]
            else:
                if x[1] == max_s[1]:
                    s = x[0]
                    break
        if question_type == "when" or question_type == "where":
            if dateline_score > max_s[1]:
                s = date
        if max_s[1] == 0:
            if question_type == "when" or question_type == "where":
                s = date
            elif question_type == "why":
                s = sent_list[len(sent_list)-1]
            else:
                s = sent_list[0]
        # Tie rule
        # list1 contains the sentences with scores
        s = remstwords(s)
        s1 = s.strip()
        s2 = s1.replace("\n"," ")
        value = qid_list[z] + "Answer: " + s2 + "\n\n"
        o.write(value)
        sorted_list = sorted(list1, key=lambda  x: (-x[1],x[0]))
o.close()

'''def Postagger(sent):
    sent_tag = nltk.pos_tag(nltk.word_tokenize(sent))
    return sent_tag

def findWH(sent):
    for word in sent.split():
        if ques_words.__contains__(word.lower()):
            return word

def findNNP(sent_tagged):
    #i = [x for x in sent_tagged if x[1] == "NNP" or x[1] == "NNPS"]
    nnp = []
    flag = 0
    s = None
    for x in sent_tagged:
        if x[1] == "NNP" and flag == 0:
            s = x[0]
            flag = 1
        elif x[1] == "NNP" and flag == 1:
            s = s + " " + x[0]
        else:
            if s != None:
                nnp.append(s)
            s = None
            flag = 0
    if sent_tagged[-1][1] == "NNP":
        nnp.append(s)
    #for x in i:
        #sent_tagged.remove(x)
    return nnp

def removestopwords_punct(ques):
    t_ques = copy.deepcopy(ques)
    for w in t_ques:
        word_lm = lemmatizer.lemmatize(w[0].lower())
        if word_lm in STOPWORDS or word_lm in string.punctuation:
            ques.remove(w)
    return ques

def findcomplNominal(tag_sent):

    nom = []
    for i,(x,t) in enumerate(tag_sent):
        if tag_sent[i][1] == 'NN' and tag_sent[i-1][1] == 'JJ':
            nom.append(tag_sent[i-1][0] + " " + tag_sent[i][0])
            tag_sent.remove(tag_sent[i-1])
            tag_sent.remove((x,t))
    return nom

def findotherNominal(tag_sent):
    nom = []
    for i,(x,t) in enumerate(tag_sent):
        if tag_sent[i][1] == 'NN' and tag_sent[i-1][1] == 'NN':
            nom.append(tag_sent[i-1][0] + " " + tag_sent[i][0])
            tag_sent.remove(tag_sent[i-1])
            tag_sent.remove((x,t))
    return nom

def findnounAdj(tag_sent):
    na = []
    i = [tag_sent.index(x) for x in tag_sent if x[1] == "JJ"]
    for z in i:
        noun = [y[0] for y in tag_sent if tag_sent.index(y) > z and y[1] == "NN" or y[1] == "NNS"]
        na.append(tag_sent[z][0] + " " + ' '.join(map(str, noun)))
    return na

def findallNoun(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "NN" or x[1] == "NNS"]
    return i

def findallVerb(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "VBZ" or x[1] == "VB" or x[1] == "VBD" or x[1] == "VBG" or x[1] == "VBN" or x[1] == "VBP"]
    return i

def findallAdverb(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "RB" or x[1] == "RBR" or x[1] == "RBS"]
    return i

def countexp_nnp(para, nnp):
    cnt = 0
    for stxt in nnp:
        if stxt in para:
            cnt = cnt + 1
            break
        else:
            for w in stxt.split():
                if w in para:
                    cnt = cnt + 1
                    break
    return cnt

def countexp(para, searchText):
    cnt = 0
    for stxt in searchText:
        if stxt.lower() in para.lower():
            cnt = cnt + 1
    return cnt

def countexp_verb(para, searchText):
    cnt = 0
    searchWords = [lemmatizer.lemmatize(s) for s in searchText]
    for stxt in searchWords:
        if stxt.lower() in para.lower():
            cnt = cnt + 1
    return cnt

def countexp_na(para, searchText):
    cnt = 0
    flag = 0
    for stxt in searchText:
        for sen in nltk.sent_tokenize(para):
            for word in stxt.split():
                if word in sen:
                    flag = 1
                else:
                    flag = 0
            if flag == 1:
                cnt = cnt + 1
                break
    return cnt


os.chdir(sys.argv[1])
files = glob.glob('*.questions')
for file in files:
    ans_types = []
    fileId = file.title().split('.')[0]
    quotations = []
    ner_tag = []
    nnp_words = []
    complNominal = []
    otherNominal = []
    nounAdj = []
    otherNoun = []
    verb = []
    adverb = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.__contains__("Question:"):
                ques = line.split("Question:")
                #print "\n" + ques[1]
                Wh_word = findWH(ques[1])
                #print Wh_word
                q_tagged = Postagger(ques[1])
                q_tagged_copy = copy.deepcopy(q_tagged)
                #print "Answer Type: "
                if Wh_word.lower() == "how":
                    i = [x[0] for x in q_tagged].index(Wh_word)
                    if q_tagged[i+1][1].lower() == 'jj':
                        ans_types.append(ans_type[Wh_word.lower() + " jj"])
                    elif q_tagged[i+1][1].lower() == "mod":
                        ans_types.append(ans_type[Wh_word.lower() + " mod"])
                    elif q_tagged[i+1][1].lower() == "long":
                        ans_types.append(ans_type[Wh_word.lower() + " long"])
                    elif q_tagged[i+1][1].lower() == "far":
                        ans_types.append(ans_type[Wh_word.lower() + " far"])
                    elif q_tagged[i+1][1].lower() == "much":
                        ans_types.append(ans_type[Wh_word.lower() + " much"])
                    elif q_tagged[i+1][1].lower() == "old":
                        ans_types.append(ans_type[Wh_word.lower() + " old"])
                    elif q_tagged[i+1][1].lower() == "often":
                        ans_types.append(ans_type[Wh_word.lower() + " often"])
                    else:
                        ans_types.append(ans_type[Wh_word.lower()])
                else :
                    ans_types.append(ans_type[Wh_word.lower()])
                #findNominal(q_tagged)
                #ner_tag.append(st.tag(ques[1].split()))
                #print ner_tag
                quot = []
                for val in quoted.findall(ques[1]):
                    if val:
                        quot.append(val)
                quotations.append(quot)
                #print quotations
                nnp_words.append(findNNP(q_tagged))
                complNominal.append(findcomplNominal(q_tagged))
                otherNominal.append(findotherNominal(q_tagged))
                nounAdj.append(findnounAdj(q_tagged))
                otherNoun.append(findallNoun(q_tagged))
                removestopwords_punct(q_tagged)
                verb.append(findallVerb(q_tagged))
                adverb.append(findallAdverb(q_tagged))
    #print fileId
    #print nnp_words
    with open(fileId + '.story', 'r') as g:
        story = g.read().split("TEXT:")[1]
        paras = re.split("[\.|\"|!]\s*\n\n+", story)
        #sents = nltk.sent_tokenize(story)

        #for each question
        for i in range(len(quotations)):
            count_list = []
            for para in paras:
                count = 0
                count = count + countexp(para, quotations[i])
                count = count + countexp_nnp(para, nnp_words[i])
                count = count + countexp(para, complNominal[i])
                count = count + countexp(para, otherNominal[i])
                count = count + countexp_na(para, nounAdj[i])
                count = count + countexp(para, otherNoun[i])
                count = count + countexp_verb(para, verb[i])
                count = count + countexp(para, adverb[i])
                count_list.append(count)
                if count > 0:
                    ner_tagged = st.tag(para.split())
                    matched_words = [x[0] for x in ner_tagged if x[1].lower() in ans_types[i]]'''
                    #print ner_tagged
                    #print matched_words
                    # we matched the words in the tagged para to the ans types.
                    #print "\n"
            #print "QUES " + (i+1).__str__() + ": "
            #print count_list

            #print "\n"
        #for sent in sents:
            #print quotations
