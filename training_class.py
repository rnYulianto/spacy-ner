# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:26:57 2017

@author: rn_yulianto
"""

import pandas as pd
import xml.etree.ElementTree as ET
import spacy
import random
import pickle

"""
Atribute:
    data = berisi data yang telah sesuai format untuk training
    model = berisi model yang digunakan untuk prediksi
    excel = data dari file excel
    root = data tagging dari file gate
    message_col = nama kolom untuk pesan pada file excel
    id_col = nama kolom untuk id pada file excel
    processed_data = data hasil prediksi dari semua data pada file excel
"""

class Trainer:
    data = []
    model = None
    
    #excel_file = file excel data + id (baris pertama adalah nama column/data dimulai dari baris ke 2)
    #gate_file = file hasil tagging dengan gate (semua tag harus terdapat dalam satu grup/AnotationSet)
    #model_path = folder path jika sudah terdapat model, kosongkan jika ingin melakukan training
    #message_column = nama kolom dari pesan
    #id_column = nama kolom dari id
    
    #Inisiasi object
    
    def __init__(self, excel_file, gate_file, model_path=None, message_column = 'message',id_column='id'):
        self.message_col = message_column
        self.id_col = id_column
        
        self.excel = pd.read_excel(excel_file, converters={'id': str})
        
        tree = ET.parse(gate_file)
        self.root = tree.getroot()
        
        if model_path:
            self.model = spacy.load(model_path)
    
    #model_path = path folder dari model
    #Load model yang sudah ada
    def load_model(self, model_path):
        self.model = spacy.load(model_path)
    
    #pickle_name = nama file pickle dari data yang sudah dibuat sebelumnya
    #Digunakan untuk load data yang formatnya sudah sesuai format training (file pickle)
    def load_data(self, pickle_name):
        with open(pickle_name, 'rb') as f:
            self.data = pickle.laod(f)
    
    #tag_name = tag grup pada tagging gate
    #pickle_name = nama file pickle data akan disimpan, jika kosong data tidak akan disimpan
    
    #Build data agar mempunyai format yang sesuai untuk training model
    
    def build_data(self, tag_name, pickle_name=''):
        for ann_set in self.root.findall('AnnotationSet'):
            if 'Name' in ann_set.attrib:
                if ann_set.attrib['Name'] == 'Original markups':
                    html = ann_set
                elif ann_set.attrib['Name'] == tag_name:
                    tag = ann_set
                    
        td = [td_tag for td_tag in html if td_tag.attrib['Type'] == 'td']
        temp_data = list()
        
        if len(self.excel)<len(td):
            td.pop(0)
        
        for index, row in self.excel.iterrows():
            temp_td = td[index]
            temp_tag = [tg for tg in tag if (int(tg.attrib['StartNode'])>=int(temp_td.attrib['StartNode']) and int(tg.attrib['EndNode'])<=int(temp_td.attrib['EndNode']))]
            
            entities = list()
            for e in temp_tag:
                entities.append((int(e.attrib['StartNode'])-int(temp_td.attrib['StartNode']), int(e.attrib['EndNode'])-int(temp_td.attrib['StartNode']), e.attrib['Type']))
            
            temp_data.append((row[0], {'entities': entities}))
            
        print('data constructed')
        
        if(pickle_name!=''):
            with open(pickle_name+'.pickle', 'wb') as f:
                pickle.dump(temp_data, f)
        self.data = temp_data
    
    
    #labels = label yang akan ditraining, berbentuk array, disesuaikan dengan tagging dalam file gate
    #path = path folder untuk menyimpan model yang akan dibuat, kosongkan jika model tidak disimpan
    
    #melakukan training dengan data yang sudah dibangun
    
    def train(self, labels, path=None, itteration=10, sample=1000):
        nlp = spacy.blank('id')
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
        
        if sample>len(self.data):
            sample=len(self.data)
        
        for label in labels:
            ner.add_label(label)
        
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(itteration):
                used_data = random.sample(self.data, sample)
                print('sample selected')
                losses = {}
                for text, annotations in used_data:
                    nlp.update([text], [annotations], sgd=optimizer, drop=0.35, losses=losses)
                print(losses)
        
        self.model = nlp
        
        if path:
            nlp.to_disk(path)
    
    #text = text yang akan diprediksi tag nya
    #Harus load model atau learning terlebih dahulu
    def _predict(self, text):
        if self.model != None:
            print('processing: ', text)
            single_doc = {'text': text}
            doc = self.model(text)
            for ent in doc.ents:
                single_doc[ent.label_] = ent.text
            return single_doc
        else:
            print('Train the model first')
            pass
    
    #output_name = nama dari excel file yang ingin dihasilkan (berakhiran .xls/ .xlsx)
    #kosongkan jika hasil proses tidak ingin disimpan
    
    #Melakukan prediksi terhadap semua data dalam file excel   
    def process_data(self, output_name=None):
        data_dict = list()
        for index, e in self.excel.iterrows():
            single_doc = self._predict(e[self.message_col])
            single_doc['id'] = e[self.id_col]
            data_dict.append(single_doc)
            
        pd_dict = pd.DataFrame(data_dict)
        
        self.processed_data = pd_dict
        
        if output_name:
            writer = pd.ExcelWriter(output_name)
            pd_dict.to_excel(writer,'Sheet1')
            writer.save()