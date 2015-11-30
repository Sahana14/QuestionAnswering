import nltk
import constants
import copy
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = nltk.WordNetLemmatizer()
import helper
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
import constants
year_list = []
for i in range(1400,2000):
    year_list.append(str(i))

def formatFinalSent(Whword, s2, q):
    matched_ans = []
    loc = []
    final_ans = []
    if Whword.lower() == "who" or Whword.lower() == "where":
        anstype = constants.ans_type[Whword.lower()]
        s2_str = ""
        for words in s2.split():
            word = helper.removepunc(words)
            s2_str = s2_str + word + " "
        ner_tag = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(s2_str)))
        for tree in ner_tag.subtrees():
            if anstype == "location":
                if tree.label().lower() == anstype or tree.label().lower() == "gpe":
                    s = [x[0] for x in tree.leaves()]
                    matched_ans = matched_ans + s
                loc = helper.findinList(s2, constants.LOCATION)
                s2_li = s2.split()
                for b,word in enumerate(s2_li):
                    if word == "in" or word == "from":
                        loc.append(s2_li[b+1])
                        if b+2 <= len(s2_li) - 1:
                            loc.append(s2_li[b+2])
                        if b+3 <= len(s2_li) - 1:
                            loc.append(s2_li[b+3])
                if not loc == []:
                    matched_ans = matched_ans + loc
            else:
                if tree.label().lower() == anstype:
                    s = [x[0] for x in tree.leaves()]
                    matched_ans = matched_ans + s
                person = helper.findinList(s2, constants.OCCUPATION) + helper.findinList(s2, constants.PERSON_NAMES)
                s2_li = s2.split()
                for b,word in enumerate(s2_li):
                    if word == "by":
                        person.append(s2_li[b+1])
                        if b+2 <= len(s2_li) - 1:
                            person.append(s2_li[b+2])
                        if b+3 <= len(s2_li) - 1:
                            person.append(s2_li[b+3])
                if not person == []:
                    matched_ans = matched_ans + person
        final_ans = matched_ans
    s2_list = s2.split()
    if Whword.lower() == "when":
        s2_nopunct = helper.remove_puncts(s2)
        s2_list1 = s2_nopunct.split()
        ner_tag = st.tag(s2_list1)
        matched_ans = []
        for tree in ner_tag:
            if tree[1].lower() == "date" or tree[1].lower() == "time":
                matched_ans.append(tree[0])
        if matched_ans == []:
            for word in s2_list1:
                if word in constants.TIME1 or word in year_list:
                    matched_ans.append(word)
        final_ans = trimFinalAnsWhen(matched_ans, s2)
    if Whword.lower() == "how":
        q_list = q.split()
        for i,word in enumerate(q_list):
            if word.lower() == "how":
                if word.lower() + " " + q_list[i+1] in constants.ans_type:
                    s2_temp = s2
                    s2_str = ""
                    for words in s2.split():
                        word = helper.removepunc(words)
                        s2_str = s2_str + word + " "
                    ner_tag = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(s2_str)))
                    #copying ner_tag since we need it for date subtraction
                    ner_tag_temp = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(s2_str)))
                    matched_ans = []
                    temp_list = []
                    for tree in ner_tag.subtrees():
                        if tree.label().lower() == "date" or tree.label().lower() == "time":
                            temp_list.append(tree.leaves[0][0])
                    for wrd1 in s2.split():
                        wrd2 = helper.removepunc(wrd1)
                        if wrd2 in constants.TIME1 or wrd2 in year_list:
                            temp_list.append(wrd2)
                    for ind,y in enumerate(ner_tag.leaves()):
                        if y[1] == "CD":
                            if not y[0] in temp_list:
                                if any(char.isdigit() for char in y[0]):
                                    matched_ans.append(ner_tag_temp.leaves()[ind][0])
        final_ans = trimFinalAnsHow(matched_ans, s2)
    if Whword.lower() == "what":
        s2_str = ""
        for words in s2.split():
            word = helper.removepunc(words)
            s2_str = s2_str + word + " "
        ner_tag = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(s2_str)))
        matched_ans = []
        for tree in ner_tag.subtrees():
            if tree.label().lower() == "organization":
                matched_ans.append(tree.leaves()[0][0])
        final_ans = matched_ans
        #final_ans = trimFinalAnsWhat(matched_ans, s2.split())
    return final_ans

def trimFinalAnsWhat(matched_ans, final_sent):
    new_ans = []
    final_sent = helper.remove_puncts(final_sent)
    if not matched_ans == []:
        final_sent_list = final_sent.split()
        for a,words in enumerate(final_sent_list):
            for ans in matched_ans:
                new_ans_str = ""
                if words == ans:
                    if a == len(final_sent_list) - 1:
                        new_ans_str = new_ans_str
                    elif a == len(final_sent_list) - 2:
                        new_ans_str = new_ans_str + " " + final_sent_list[a+1]
                    elif a == len(final_sent_list) - 3:
                        new_ans_str = new_ans_str + " " + final_sent_list[a+1] + " " + final_sent_list[a+2]
                    else:
                        new_ans_str = new_ans_str + " " + final_sent_list[a+1] + " " + final_sent_list[a+2] + " " + final_sent_list[a+3]
                    new_ans.append(new_ans_str)
    return new_ans

def trimFinalAnsHow(matched_ans, final_sent):
    new_ans = []
    final_sent = helper.remove_puncts(final_sent)
    if not matched_ans == []:
        final_sent_list = final_sent.split()
        for a,words in enumerate(final_sent_list):
            for ans in matched_ans:
                new_ans_str = ""
                if words.__contains__(ans):
                    if a == len(final_sent_list) - 1:
                        new_ans_str = new_ans_str + " " + words
                    else:
                        new_ans_str = new_ans_str + " " + words + " " + final_sent_list[a+1]
                    new_ans.append(new_ans_str)
    return new_ans

def trimFinalAnsWhen(matched_ans, final_sent):
    new_ans = []
    final_sent = helper.remove_puncts(final_sent)
    if not matched_ans == []:
        final_sent_list = final_sent.split()
        for a,words in enumerate(final_sent_list):
            for ans in matched_ans:
                new_ans_str = ""
                if words == ans:
                    if a == 0:
                        new_ans_str = new_ans_str + " " + words + " " + final_sent_list[a+1]
                    elif a == len(final_sent_list) - 1:
                        new_ans_str = new_ans_str + " " + final_sent_list[a-1] + " " + words
                    else:
                        new_ans_str = new_ans_str + " " + final_sent_list[a-1] + " " + words + " " + final_sent_list[a+1]
                    new_ans.append(new_ans_str)
    return new_ans

def findMatchingAns(q,sent):
    match_ans = []
    for ans in sent:
        Whword = helper.findWH(q)
        match_ans = match_ans + formatFinalSent(Whword,ans[0],q)
        if not match_ans == []:
            if Whword.lower() == "what" and (q.__contains__("name") or q.__contains__("names")):
                anstype = constants.ans_type["who"]
            else:
                anstype = constants.ans_type[Whword.lower()]
                #this is only for where ques adding org to ans
            if anstype == "location":
                ans_str = ""
                for words in ans[0].split():
                    word = helper.removepunc(words)
                    ans_str = ans_str + word + " "
                ner_tag = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(ans_str)))
                for tree in ner_tag.subtrees():
                    if tree.label().lower() == "organization":
                        s = [x[0] for x in tree.leaves()]
                        match_ans = match_ans + s
    return match_ans

def getUnmatchedAns(Wh_word,q,sent):
    un_match_ans=[]
    if Wh_word == "who":
        q1=q.lower().replace("who","where")
        un_match_ans = findMatchingAns(q1,sent)
    else:
        q1=q.lower().replace("where","who")
        un_match_ans = findMatchingAns(q1,sent)

def matchFinalAnsWhoWhere(q, top_ans_list):
    q_verb_stem = []
    q_verb_list = helper.findverbsinques(q)
    final_ans_list = []
    verb_match_sent = []
    match_ans = []
    Wh_word = helper.findWH(q)
    if not q_verb_list == []:
        for verb in q_verb_list:
            q_verb = WordNetLemmatizer().lemmatize(verb, 'v')
            q_verb_stem.append(q_verb)
    for ans in top_ans_list:
        for q_verb in q_verb_stem:
            for word in ans[0].split():
                w2 = WordNetLemmatizer().lemmatize(word, 'v')
                if q_verb == w2 and ans not in verb_match_sent:
                    verb_match_sent.append(ans)
                    break
    if not q_verb_list == []:
        if not verb_match_sent == []:
        #case a
            match_ans = findMatchingAns(q,verb_match_sent)  # verb + ner found in a sentence if match_ans not empty
            un_match_ans=[]
            un_match_ans=getUnmatchedAns(Wh_word,q,verb_match_sent[0][0])
            sent_tag = nltk.pos_tag(nltk.word_tokenize(verb_match_sent[0][0]))
            match_ans = match_ans + helper.findNNP(sent_tag)
            list(set(match_ans) - set(un_match_ans))
            if match_ans == []:
                #case b
                sent_tag = nltk.pos_tag(nltk.word_tokenize(verb_match_sent[0][0]))
                match_ans = match_ans + helper.findNNP(sent_tag)
                un_match_ans=getUnmatchedAns(Wh_word,q,verb_match_sent[0][0])
                list(set(match_ans) - set(un_match_ans))
        else:
            match_ans = findMatchingAns(q,top_ans_list)
            # ner found in a sentence in the remaining of sentences where verb was not ptresent if match_ans not empty
            if not match_ans == []:
                #case c
                sent_tag = nltk.pos_tag(nltk.word_tokenize(top_ans_list[0][0]))
                match_ans = match_ans + helper.findNNP(sent_tag)
                un_match_ans=getUnmatchedAns(Wh_word,q,top_ans_list[0][0])
                list(set(match_ans) - set(un_match_ans))
            else:
            #case d
                sent_tag = nltk.pos_tag(nltk.word_tokenize(top_ans_list[0][0]))
                match_ans = match_ans + helper.findNNP(sent_tag) # No ner found in a sentence. just return the top score verb sentence
                un_match_ans=getUnmatchedAns(Wh_word,q,top_ans_list[0][0])
                list(set(match_ans) - set(un_match_ans))
    else:
        match_ans1 = []
        sen = ""
        q_nn = findnounsinsent(nltk.pos_tag(nltk.word_tokenize(q)))
        flag = 0
        for nn in q_nn:
            nn = WordNetLemmatizer().lemmatize(nn, 'n')
            for top_ans in top_ans_list:
                sen = top_ans[0]
                sent_tag = nltk.pos_tag(nltk.word_tokenize(sen))
                for s_nn in findnounsinsent(sent_tag):
                    s_n = WordNetLemmatizer().lemmatize(s_nn, 'n')
                    if s_n.lower() == nn.lower():
                        temp_list = []
                        temp_list.append(top_ans)
                        match_ans1 = findMatchingAns(q,temp_list)
                        match_ans = match_ans + match_ans1 + helper.findNNP(sent_tag)
                        un_match_ans=getUnmatchedAns(Wh_word,q,top_ans[0])
                        list(set(match_ans) - set(un_match_ans))
                        flag = 1
        if flag ==0:
            for ans in top_ans_list:
                temp_list = []
                temp_list.append(ans)
                match_ans1 = findMatchingAns(q,temp_list)
                match_ans = match_ans + match_ans1
            sent_tag = nltk.pos_tag(nltk.word_tokenize(top_ans_list[0][0]))
            match_ans = match_ans + helper.findNNP(sent_tag)
            un_match_ans=getUnmatchedAns(Wh_word,q,top_ans_list[0][0])
            list(set(match_ans) - set(un_match_ans))
        '''for ans in top_ans_list:
            temp_list = []
            temp_list.append(ans)
            match_ans1 = findMatchingAns(q,temp_list)
            sent_tag = nltk.pos_tag(nltk.word_tokenize(ans[0]))
            match_ans = match_ans + match_ans1 + helper.findNNP(sent_tag)'''
    return match_ans

def findverbsinsent(sent_tag):
    temp = []
    for s in sent_tag:
        if s[1] == "VB" or s[1] == "VBD" or s[1] == "VBG" or s[1] == "VBN":
            temp.append(s[0])
    return temp

def findnounsinsent(sent_tag):
    temp = []
    for s in sent_tag:
        if s[1] == "NN" or s[1] == "NNS":
            temp.append(s[0])
    return temp

def matchFinalAnsWhat(q, top_ans_list):
    q_verb_stem = []
    q_verb_list = helper.findverbsinques(q)
    final_ans_list = []
    verb_match_sent = []
    match_ans = []
    if not q_verb_list == []:
        for verb in q_verb_list:
            q_verb = WordNetLemmatizer().lemmatize(verb, 'v')
            q_verb_stem.append(q_verb)


    for ans in top_ans_list:
        for q_verb in q_verb_stem:
            for word in ans[0].split():
                w2 = WordNetLemmatizer().lemmatize(word, 'v')
                if q_verb == w2 and ans not in verb_match_sent:
                    verb_match_sent.append(ans)
                    break


    if not q_verb_list == []:
        if not verb_match_sent == []:
        #case a
            match_ans = findMatchingAns(q,verb_match_sent)  # verb + ner found in a sentence if match_ans not empty
            if match_ans == []:
                #case b
                sent_tag = nltk.pos_tag(nltk.word_tokenize(verb_match_sent[0][0]))
                for vb in q_verb_stem:
                    for s_vb in findverbsinsent(sent_tag):
                        s_vb = WordNetLemmatizer().lemmatize(s_vb, 'v')
                        if vb == s_vb:
                            match_ans.append(vb)
                for nn in findnounsinsent(nltk.pos_tag(nltk.word_tokenize(q))):
                    nn = WordNetLemmatizer().lemmatize(nn, 'n')
                    for s_nn in findnounsinsent(sent_tag):
                        s_nn = WordNetLemmatizer().lemmatize(s_nn, 'n')
                        if s_nn.lower() == nn.lower():
                            match_ans.append(s_nn)
                match_ans = trimFinalAnsWhat(match_ans, verb_match_sent[0][0])
        else:
            match_ans = findMatchingAns(q,top_ans_list)
            # ner found in a sentence in the remaining of sentences where verb was not ptresent if match_ans not empty
            if not match_ans == []:
                #case c
                matched_ans = []
                sen = ""
                q_nn = findnounsinsent(nltk.pos_tag(nltk.word_tokenize(q)))
                flag = 0
                for nn in q_nn:
                    nn = WordNetLemmatizer().lemmatize(nn, 'n')
                    for top_ans in top_ans_list:
                        sen = top_ans[0]
                        sent_tag = nltk.pos_tag(nltk.word_tokenize(sen))
                        for s_nn in findnounsinsent(sent_tag):
                            s_n = WordNetLemmatizer().lemmatize(s_nn, 'n')
                            if s_n.lower() == nn.lower() and flag == 0:
                                matched_ans.append(s_nn)
                                flag = 1
                                break
                        if flag == 1:
                            break
                    if flag == 1:
                        break
                match_ans = match_ans + trimFinalAnsWhat(matched_ans, sen)
            else:
            #case d
                matched_ans = []
                sen = ""
                q_nn = findnounsinsent(nltk.pos_tag(nltk.word_tokenize(q)))
                flag = 0
                for nn in q_nn:
                    nn = WordNetLemmatizer().lemmatize(nn, 'n')
                    for top_ans in top_ans_list:
                        sen = top_ans[0]
                        sent_tag = nltk.pos_tag(nltk.word_tokenize(sen))
                        for s_nn in findnounsinsent(sent_tag):
                            s_n = WordNetLemmatizer().lemmatize(s_nn, 'n')
                            if s_n.lower() == nn.lower() and flag == 0:
                                matched_ans.append(s_nn)
                                flag = 1
                                break
                        if flag == 1:
                            break
                    if flag == 1:
                        break
                match_ans = match_ans + trimFinalAnsWhat(matched_ans, sen)
    else:
        matched_ans = []
        sen = ""
        q_nn = findnounsinsent(nltk.pos_tag(nltk.word_tokenize(q)))
        flag = 0
        for nn in q_nn:
            nn = WordNetLemmatizer().lemmatize(nn, 'n')
            for top_ans in top_ans_list:
                sen = top_ans[0]
                sent_tag = nltk.pos_tag(nltk.word_tokenize(sen))
                for s_nn in findnounsinsent(sent_tag):
                    s_n = WordNetLemmatizer().lemmatize(s_nn, 'n')
                    if s_n.lower() == nn.lower() and flag == 0:
                        matched_ans.append(s_nn)
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 1:
                break
        match_ans = match_ans + trimFinalAnsWhat(matched_ans, sen)
    return match_ans

def matchFinalAnsWhenHow(q, top_ans_list):
    q_verb_stem = []
    q_verb_list = helper.findverbsinques(q)
    final_ans_list = []
    verb_match_sent = []
    match_ans = []
    if not q_verb_list == []:
        for verb in q_verb_list:
            q_verb = WordNetLemmatizer().lemmatize(verb, 'v')
            q_verb_stem.append(q_verb)

    for ans in top_ans_list:
        for q_verb in q_verb_stem:
            for word in ans[0].split():
                w2 = WordNetLemmatizer().lemmatize(word, 'v')
                if q_verb == w2 and ans not in verb_match_sent:
                    verb_match_sent.append(ans)
                    break


    if not q_verb_list == []:
        if not verb_match_sent == []:
        #case a
            match_ans = findMatchingAns(q,verb_match_sent)  # verb + ner found in a sentence if match_ans not empty
        else:
            #case c
            match_ans = findMatchingAns(q,top_ans_list)
            # ner found in a sentence in the remaining of sentences where verb was not ptresent if match_ans not empty
    else:
        match_ans1 = []
        sen = ""
        q_nn = findnounsinsent(nltk.pos_tag(nltk.word_tokenize(q)))
        flag = 0
        for nn in q_nn:
            nn = WordNetLemmatizer().lemmatize(nn, 'n')
            for top_ans in top_ans_list:
                sen = top_ans[0]
                sent_tag = nltk.pos_tag(nltk.word_tokenize(sen))
                for s_nn in findnounsinsent(sent_tag):
                    s_n = WordNetLemmatizer().lemmatize(s_nn, 'n')
                    if s_n.lower() == nn.lower():
                        temp_list = []
                        temp_list.append(top_ans)
                        match_ans1 = findMatchingAns(q,temp_list)
                        match_ans = match_ans + match_ans1
                        flag = 1
        if flag ==0:
            for ans in top_ans_list:
                temp_list = []
                temp_list.append(ans)
                match_ans1 = findMatchingAns(q,temp_list)
                match_ans = match_ans + match_ans1
        '''for ans in top_ans_list:
            temp_list = []
            temp_list.append(ans)
            match_ans = findMatchingAns(q,temp_list)'''
    return match_ans

def matchFinalWhy(q, top_ans_list):
    q_verb_stem = []
    q_verb_list = helper.findverbsinques(q)
    final_ans_list = []
    verb_match_sent = []
    match_ans = []
    if not q_verb_list == None:
        for verb in q_verb_list:
            q_verb = WordNetLemmatizer().lemmatize(verb, 'v')
            q_verb_stem.append(q_verb)
    for ans in top_ans_list:
        for q_verb in q_verb_stem:
            for word in ans[0].split():
                w2 = WordNetLemmatizer().lemmatize(word, 'v')
                if q_verb == w2 and ans not in verb_match_sent:
                    verb_match_sent.append(ans)
                    break
    if not q_verb_list == None:
        if not verb_match_sent == []:
        #case a
            match_ans = verb_match_sent[0][0].split() # verb + ner found in a sentence if match_ans not empty
        else:
            #case c
            match_ans = top_ans_list[0][0].split()
            # ner found in a sentence in the remaining of sentences where verb was not ptresent if match_ans not empty
    return match_ans