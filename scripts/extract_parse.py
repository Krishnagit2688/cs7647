import re
import pdfplumber
import os
import pandas as pd


def parse_text(text):
    text_split = text.split("Moderator")
    return text_split

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

def extract_split(company):

    for f in os.listdir("..\\transcripts_pdf\\"+company):

        if f == "june-23.pdf":

            name = f.split(".")
            print(name[0])
            full_text = extract_text_from_pdf('..\\transcripts_pdf\\'+company+'\\'+f)

            # if company == 'icici' or 'maruti':
            text_split = parse_text(full_text)

            i = 0
            for t in text_split:
                i = i+1
                if not os.path.exists("..\\transcripts_text\\"+company+"\\"+name[0]):
                    os.makedirs("..\\transcripts_text\\"+company+"\\"+name[0])

                with open('..\\transcripts_text\\'+company+'\\'+name[0]+'\\'+name[0]+'-'+str(i)+'.txt', "w", encoding="utf-8") as fh:
                    fh.write(t)

def clean(company):
    js = {}
    text_li = []
    id_li = []
    tran_li = []
    for d in os.listdir("..\\transcripts_text_clean\\"+company):
        if d == "june-23":
            print(d)
            i = 0
            for f in os.listdir("..\\transcripts_text_clean\\"+company+"\\"+d):
                with open("..\\transcripts_text_clean\\"+company+"\\"+d+"\\"+f, "r", encoding="utf-8") as fh:
                    i = i+1
                    clean_text = fh.read()
                    text_li.append(clean_text)
                    id_li.append(i)
                    tran_li.append(d)

    js["transcripts"] = tran_li
    js["id"] = id_li
    js["text"] = text_li

    df = pd.DataFrame(js)
    # print(df.head(5, False))

    df.to_excel("..\\transcripts_text_clean\\"+company+"_score_final.xlsx", index=False)


# extract_split("icici")
clean("icici")
#clean("maruti")
#clean("infosys")



