import pandas as pd
import xml.etree.ElementTree as ET
import spacy
import random
import pickle

# ann_file = file dari gate
# excel_file = data dari file excel
# dump_data = jika True data akan di dump jadi data.pickle
def get_data(ann_file, excel_file, dump_data=False):
    tree = ET.parse(ann_file)
    root = tree.getroot()
    
    for ann_set in root.findall('AnnotationSet'):
        if 'Name' in ann_set.attrib:
            if ann_set.attrib['Name'] == 'Original markups':
                html = ann_set
            elif ann_set.attrib['Name'] == 'bank':
                tag = ann_set
                
    td = [td_tag for td_tag in html if td_tag.attrib['Type'] == 'td']
    
    excel = pd.read_excel(excel_file, converters={'id': str})
    
    data = list()
    
    if len(excel)<len(td):
        td.pop(0)
    
    for index, row in excel.iterrows():
        temp_td = td[index]
        temp_tag = [tg for tg in tag if (int(tg.attrib['StartNode'])>=int(temp_td.attrib['StartNode']) and int(tg.attrib['EndNode'])<=int(temp_td.attrib['EndNode']))]
        
        entities = list()
        for e in temp_tag:
            entities.append((int(e.attrib['StartNode'])-int(temp_td.attrib['StartNode']), int(e.attrib['EndNode'])-int(temp_td.attrib['StartNode']), e.attrib['Type']))
        
        data.append((row[0], {'entities': entities}))
        
    print('data constructed')
    
    if(dump_data):
        with open('data.pickle', 'wb') as f:
            pickle.dump(data, f)
    return data

# data = hasil dari function get_data
# path = folder dimana model akan disimpan
# sample = jumlah sample untuk training
# iteration = jumlah iterasi training
def train(data, path, sample=1000, iteration=10):
    nlp = spacy.blank('id')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
    
    ner.add_label('tanggal')
    ner.add_label('no_rekening')
    ner.add_label('nominal')
    ner.add_label('waktu')
    ner.add_label('jenis')
    ner.add_label('to')
    ner.add_label('from')
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iteration):
            used_data = random.sample(data, sample)
            print('sample selected')
            losses = {}
            for text, annotations in used_data:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35, losses=losses)
            print(losses)
            
    nlp.to_disk(path)


