# Author: Omkar Pathak

import io
import os
import re
import nltk
from nltk import text
import pandas as pd
import docx2txt
from datetime import datetime
from dateutil import relativedelta
import constants as cs
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def extract_text_from_pdf(pdf_path):
    '''
    Helper function to extract the plain text from .pdf files

    :param pdf_path: path to PDF file to be extracted (remote or local)
    :return: iterator of string of extracted text
    '''
    # https://www.blog.pythonlibrary.org/2018/05/03/exporting-data-from-pdfs-with-python/
    # extract text from local pdf file
    with open(pdf_path, 'rb') as fh:
        try:
            for page in PDFPage.get_pages(
                    fh,
                    caching=True,
                    check_extractable=True
            ):
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    fake_file_handle,
                    codec='utf-8',
                    laparams=LAParams()
                )
                page_interpreter = PDFPageInterpreter(
                    resource_manager,
                    converter
                )
                page_interpreter.process_page(page)

                text = fake_file_handle.getvalue()
                yield text

                # close open handles
                converter.close()
                fake_file_handle.close()
        except PDFSyntaxError:
            return

def get_number_of_pages(file_name):
    try:
        # for local pdf file
        if file_name.endswith('.pdf'):
            count = 0
            with open(file_name, 'rb') as fh:
                for page in PDFPage.get_pages(
                        fh,
                        caching=True,
                        check_extractable=True
                ):
                    count += 1
            return count
        else:
            return None
    except PDFSyntaxError:
        return None



def extract_text(file_path, extension):
    '''
    Wrapper function to detect the file extension and call text
    extraction function accordingly

    :param file_path: path of file of which text is to be extracted
    :param extension: extension of file `file_name`
    '''
    text = ''
    if extension == '.pdf':
        for page in extract_text_from_pdf(file_path):
            text += ' ' + page
    else :
        print("incompatible file type")
    return text


def extract_entity_sections_grad(text):
    '''
    Helper function to extract all the raw text from sections of
    resume specifically for graduates and undergraduates

    :param text: Raw text of resume
    :return: dictionary of entities
    '''
    text_split = [i.strip() for i in text.split('\n')]
    # sections_in_resume = [i for i in text_split if i.lower() in sections]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(cs.RESUME_SECTIONS_GRAD)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.RESUME_SECTIONS_GRAD:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)

    # entity_key = False
    # for entity in entities.keys():
    #     sub_entities = {}
    #     for entry in entities[entity]:
    #         if u'\u2022' not in entry:
    #             sub_entities[entry] = []
    #             entity_key = entry
    #         elif entity_key:
    #             sub_entities[entity_key].append(entry)
    #     entities[entity] = sub_entities

    # pprint.pprint(entities)

    # make entities that are not found None
    # for entity in cs.RESUME_SECTIONS:
    #     if entity not in entities.keys():
    #         entities[entity] = None
    return entities


def extract_entities_wih_custom_model(custom_nlp_text):
    '''
    Helper function to extract different entities with custom
    trained model using SpaCy's NER

    :param custom_nlp_text: object of `spacy.tokens.doc.Doc`
    :return: dictionary of entities
    '''
    entities = {}
    for ent in custom_nlp_text.ents:
        if ent.label_ not in entities.keys():
            entities[ent.label_] = [ent.text]
        else:
            entities[ent.label_].append(ent.text)
    for key in entities.keys():
        entities[key] = list(set(entities[key]))
    return entities


def get_total_experience(experience_list):
    '''
    Wrapper function to extract total months of experience from a resume

    :param experience_list: list of experience text extracted
    :return: total months of experience
    '''
    exp_ = []
    for line in experience_list:
        experience = re.search(
            r'(?P<fmonth>\w+.\d+)\s*(\D|to)\s*(?P<smonth>\w+.\d+|present)',
            line,
            re.I
        )
        if experience:
            exp_.append(experience.groups())
    total_exp = sum(
        [get_number_of_months_from_dates(i[0], i[2]) for i in exp_]
    )
    total_experience_in_months = total_exp
    return total_experience_in_months


def get_number_of_months_from_dates(date1, date2):
    '''
    Helper function to extract total months of experience from a resume

    :param date1: Starting date
    :param date2: Ending date
    :return: months of experience from date1 to date2
    '''
    if date2.lower() == 'present':
        date2 = datetime.now().strftime('%b %Y')
    try:
        if len(date1.split()[0]) > 3:
            date1 = date1.split()
            date1 = date1[0][:3] + ' ' + date1[1]
        if len(date2.split()[0]) > 3:
            date2 = date2.split()
            date2 = date2[0][:3] + ' ' + date2[1]
    except IndexError:
        return 0
    try:
        date1 = datetime.strptime(str(date1), '%b %Y')
        date2 = datetime.strptime(str(date2), '%b %Y')
        months_of_experience = relativedelta.relativedelta(date2, date1)
        months_of_experience = (months_of_experience.years
                                * 12 + months_of_experience.months)
    except ValueError:
        return 0
    return months_of_experience


def extract_entity_sections_professional(text):
    '''
    Helper function to extract all the raw text from sections of
    resume specifically for professionals

    :param text: Raw text of resume
    :return: dictionary of entities
    '''
    text_split = [i.strip() for i in text.split('\n')]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) \
                    & set(cs.RESUME_SECTIONS_PROFESSIONAL)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.RESUME_SECTIONS_PROFESSIONAL:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)
    return entities


def extract_email(text):
    '''
    Helper function to extract email id from text

    :param text: plain text extracted from resume file
    '''
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None


def extract_name(nlp_text, matcher):
    '''
    Helper function to extract name from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param matcher: object of `spacy.matcher.Matcher`
    :return: string of full name
    '''
    pattern = [cs.NAME_PATTERN]

    matcher.add('NAME', None, *pattern)

    matches = matcher(nlp_text)

    for _, start, end in matches:
        span = nlp_text[start:end]
        if 'name' not in span.text.lower():
            return span.text


def extract_mobile_number(text, custom_regex=None):
    '''
    Helper function to extract mobile number from text

    :param text: plain text extracted from resume file
    :return: string of extracted mobile numbers
    '''
    # Found this complicated regex on :
    # https://zapier.com/blog/extract-links-email-phone-regex/
    # mob_num_regex = r'''(?:(?:\+?([1-9]|[0-9][0-9]|
    #     [0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|
    #     [2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|
    #     [0-9]1[02-9]|[2-9][02-8]1|
    #     [2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|
    #     [2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{7})
    #     (?:\s*(?:#|x\.?|ext\.?|
    #     extension)\s*(\d+))?'''
    if not custom_regex:
        mob_num_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                        [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
        phone = re.findall(re.compile(mob_num_regex), text)
    else:
        phone = re.findall(re.compile(custom_regex), text)
    if phone:
        number = ''.join(phone[0])
        return number


def extract_skills(nlp_text, noun_chunks, skills_file=None):
    '''
    Helper function to extract skills from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param noun_chunks: noun chunks extracted from nlp text
    :return: list of skills extracted
    '''
    tokens = [token.text for token in nlp_text if not token.is_stop]
    if not skills_file:
        data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'skills.csv')
        )
    else:
        data = pd.read_csv(skills_file)
    skills = list(data.columns.values)
    skillset = []
    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def cleanup(token, lower=True):
    if lower:
        token = token.lower()
    return token.strip()


def extract_education(nlp_text):
    '''
    Helper function to extract education from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :return: tuple of education degree and year if year if found
             else only returns education degree
    '''
    edu = {}
    # Extract education degree
    try:
        for index, text in enumerate(nlp_text):
            for tex in text.split():
                tex = re.sub(r'[?|$|.|!|,]', r'', tex)
                if tex.upper() in cs.EDUCATION and tex not in cs.STOPWORDS:
                    edu[tex] = text + nlp_text[index + 1]
    except IndexError:
        pass

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(cs.YEAR), edu[key])
        if year:
            education.append((key, ''.join(year.group(0))))
        else:
            education.append(key)
    return education


def extract_experience(resume_text):
    '''
    Helper function to extract experience from resume text

    :param resume_text: Plain resume text
    :return: list of experience
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize
    filtered_sentence = [
            w for w in word_tokens if w not
            in stop_words and wordnet_lemmatizer.lemmatize(w)
            not in stop_words
        ]
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)

    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)

    test = []

    for vp in list(
        cs.subtrees(filter=lambda x: x.label() == 'P')
    ):
        test.append(" ".join([
            i[0] for i in vp.leaves()
            if len(vp.leaves()) >= 2])
        )

    # Search the word 'experience' in the chunk and
    # then print out the text after it
    x = [
        x[x.lower().index('experience') + 10:]
        for i, x in enumerate(test)
        if x and 'experience' in x.lower()
    ]
    return x


def get_data(text) : 
    try:
        data = next(text)
        if data is not None:
            return
    except StopIteration:
        return

def main() :
    
    text = """ 

NELSON CHACKO 

+91 - 8281805497 

nchacko.official@gmail.com· linkedin.com/in/nelsonchacko 

An enthusiastic recent graduate who loves to read books, play football and learn all things computer. 

EXPERIENCE 

APRIL 2020 - MAY 2020 
PRICESENZ, INTERNSHIP 
Created a mini service on AWS cloud to automatically parse resumes that are put on a google 
drive. 

MAY 2018 
GOOGLE CROWDSOURCE, WORKSHOP 
Attended Firebase Workshop organized by Google Crowdsource community and successfully 
implemented a Firebase database. 
 
JULY 2017 
NEST CYBER CAMPUS, INTERNSHIP 
Learnt about Basic Web Concepts, HTML, CSS, JavaScript, PHP and My-SQL during my one week 
time there. 
 
13 OCTOBER 2018 
ALL INDIA RESEARCH CHAMPIONSHIP, COORDINATOR 
Conducted a workshop on Ethical Hacking in association with TechiNest. 
 
JANURAY 2018 
TEDXMACE, VOLUNTEER 
Volunteered for the Tedx event conducted at our college, had a month-long preparation setting 
the stage, scheduling the guests and selling tickets. 
 
OCTOBER 2017 
CODAGE, PARTCIPANT 
Participated in the coding event CODAGE that was held as part of our National Level Tech-Fest 
Takshak’17 
 

EDUCATION 

2016 - 2020 
B.TECH COMPUTER SCIENCE, MA COLLEGE OF ENGINEERING 
7.36 ( CGPA ) 

2014 - 2016 
SENIOR SECONDARY, VIJAYAGIRI PUBLIC SCHOOL 
CBSE – 94.4 % 
 
2014 
SECONDARY, ST’ DOMINIC’S CONVENT ENGLISH MEDIUM SCHOOL 
CBSE – 10.0 ( CGPA )  

SKILLS 

  Algorithms 
  Python 
  C++ 
  Data-Structures 
  HTML 
  CSS 
 

Java Script 

ACTIVITIES 

  Leader 
  Problem Solver 
  Team worker 
  Quick Learner 
 

  Runners up at KTU Zonal Football competition representing my college 
  Runners up at 3-s Football tournament at Ilahia College of Engineering 
  Active Quiz club member 

PROJECTS 

  Main Project - Decentralized Car Rental Platform using Block Chain technology 

o  Learnt about live block chain simulator Ganache 
o  Learnt about solidity programming language 
o  Learnt Truffle Framework to integrate block chain with HTML code 
o  Used firebase to export live data from model car to the Block chain network 

 

  Mini Project – PDF analyzer 

o  Created a pdf analyzing software that is able to parse through KTU results and 

report the data found. 

o  Achieved this using PHP and Maria DB Database 
o  Used Xamp to create the local server. 

PAPER PRESENTATION 

EtherRent : A Co-operative Car Rental Platform 
 
 

URL: https://www.irjet.net/archives/V7/i4/IRJET-V7I4343.pdf 
DATE PUBLISHED: APRIL 2020 

2 

"""
 
    text = text.encode('utf-8').decode('ascii','ignore')
    dict = extract_entity_sections_grad(text)

    from pprint import pprint    
    pprint(dict)
    
    return
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    # path = r"pyresparser\resumes\OmkarResume.pdf"
    # abs_path = r"C:\Users\nelson.c\dev\omkar_resume_parser\pyresparser\resumes\OmkarResume.pdf"
    
    # text_gen = extract_text_from_pdf(abs_path)
    # text = []

    # while(True) :
    #     try :
    #         text.append(next(text_gen))
    #     except StopIteration :
    #         break
    # print(len(text))
    
    # data = []
    # data.append(get_data(text))
    
    # print(data)
    
    

if __name__ == "__main__" :
    main()
    exit()
