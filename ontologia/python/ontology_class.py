# -*- coding: utf-8 -*-
"""
Created on Tue May  8 01:50:49 2018

@author: Ulrich
"""

# toutes les importations

#import ontology owl
from owlready2 import *

#imports
import numpy as np
import pandas as pd

#import for manipulating string
import re

#import for natural language toolkit
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# import for Feature extraction
from sklearn.feature_extraction.text import CountVectorizer

# import for Training a classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

# import for Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# import for Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

#import for PDF
from . import pdf2txt

#import for similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import os

class Ontology:

    """
    #données brutes de(s) l'ontologie(s) representant la variable X
    data

    #concepts de l'ontologie
    concepts

    #code categories des données brutes de(s) l'ontologie(s) representant la variable y
    target

    #nom des categories des données brutes de(s) l'ontologie(s) representant la variable y
    target_names

    #contient les limites de chaque categories presente dans les donnees globales
    ind_cat

    #pattern utilisé pour separer 2 elements
    pattern

    #donnees de l'ontologie pretraitées et utilisable dans le modele
    corpus

    #données du fichier pdf ou du texte passé en entrée
    input_data

    #version pretraitée du fichier pdf ou du texte passé en entrée
    new_input_data

    #variable pour entrainer le model et effectuer la classification
    X_train_termcounts
    shape = ""
    X_train_tfidf = ""
    X_train = ""
    X_test = ""
    y_train = ""
    y_test = ""

    #definit le pourcentage de données a utilser pour tester le modele
    test_size = 0.3

    #matrice de confusion pour evaluer le mmodele
    cm = ""

    #classificateur
    classifier = ""

    #prediction realisé sur le test set dans le but d'evaluer le modele
    y_pred = ""


    tfidf_transformer = ""
    vectorizer = ""

    #variable pour la prediction
    X_input_termcounts_inp = ""

    X_input_tfidf_inp = ""

    #categories predites
    predicted_categories = ""

    #nombre d'ontologies chargées
    number_ontologies = 0

    #concepts trouvés dans le  texte d'entrée (pdf ou phrase)
    thematic = []
    thematics = []

    #categories des concepts trouvés
    thematics_cat = []

    #nbre de mots du ngram
    self.ngram_nbre_word = 10

    self.obo = ""

    """

    ontologies = {
        'malaria':os.path.join(sys.path[0], "ontologia\ontologies\idomal.owl.xml"),
        'plant':os.path.join(sys.path[0], "ontologia\ontologies\po.owl.xml"),
        'diseases':os.path.join(sys.path[0], "ontologia\ontologies\doid.owl.xml"),
    }

#reerr
    nbre_instance = 0

    #constructeur de la classe
    def __init__(self):

       # Ontology.nbre_instance += 1

        #if (Ontology.nbre_instance == 1):
        self.concepts = []
        self.data = []
        self.target = np.array([], dtype = np.int64)
        self.target_names = []
        self.ind_cat = []
        self.pattern= "~~~"
        self.pattern2= "%¨£"
        self.corpus = []
        self.input_data = []
        self.new_input_data = []
        self.X_train_termcounts = ""
        self.shape = ""
        self.X_train_tfidf = ""
        self.X_train = ""
        self.X_test = ""
        self.y_train = ""
        self.y_test = ""
        self.cm = ""
        self.classifier = ""
        self.y_pred = ""
        self.tfidf_transformer = ""
        self.vectorizer = ""
        self.X_input_termcounts_inp = ""
        self.X_input_tfidf_inp = ""
        self.predicted_categories = ""
        self.number_ontologies = 0
        self.thematic = []
        self.thematics = []
        self.thematics_cat = []
        self.test_size = 0.3
        self.ngram_nbre_word = 6
        self.ontology = []
        self.obo = ""


    ########################### methodes d'instance ############################


    #permet d'enlever les mots entre parentheses d'une chaine de charactere
    def enlv(self,s):
        replaced = re.sub('(\((.*?)\))', '', s)
        return replaced


    #fonction qui permet de charger les ontologies
    def load_ontologies(self, ontology_path=os.path.dirname('/ontologia/ontologies')):
        global number_ontologies
        onto_path.append(ontology_path)
        for name in self.ontologies:
            onto = get_ontology(self.ontologies[name])
            onto.load()
            self.target_names.append(name)
            self.number_ontologies += 1
            self.get_classes_ontology(onto)
            self.obo = onto.get_namespace("http://purl.obolibrary.org/obo/")
        self.cleaning_corpus()
        self.train_and_classify()




    #fonction qui permet davoir les classes d'une ontologie
    def get_classes_ontology(self,ontology,pattern= "~~~"):

        index = self.target.size

        for cl in list(ontology.classes()):
            #c = self.enlv("".join(cl.label)) + self.pattern + self.enlv("".join(cl.IAO_0000115)) + self.pattern + self.enlv("".join(cl.comment)) + self.pattern + self.enlv("".join(cl.hasNarrowSynonym)) + self.pattern + self.enlv("".join(cl.hasRelatedSynonym)) + self.pattern + self.enlv("".join(cl.hasExactSynonym)) +  self.pattern + self.enlv("".join(cl.hasBroadSynonym))
            c, c_obj = self.recupConcept(cl)
            if(len(c.strip())!=0 and len("".join(cl.label).strip())!=0):
                self.data.append(c)
                self.ontology.append(c_obj)
                self.concepts.append(self.enlv("".join(cl.label)))
                b = np.array([self.number_ontologies-1])
                self.target = np.append(self.target, b)
                index += 1
        self.ind_cat.append(index)


    # pretraiter les données de l'ontologie
    def cleaning_corpus(self):

        i = 0
        size = len(self.data)
        while (i < size):
            tab_elt = self.data[i].split(self.pattern)
            chaine = ""
            for elt in tab_elt:
                # enlever les nombres isolés(non liées à un mot ou lettre)
                review = re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b","",elt)
                review = review.lower()
                review = review.split()
                ps = PorterStemmer()
                review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
                review = ' '.join(review)
                if (len(review.strip())!=0):
                    if (chaine ==""):
                        chaine = review
                    else:
                        chaine = chaine + self.pattern + review
            if (len(chaine.strip())!=0):
                self.corpus.append(chaine)
            i = i+1

    #entrainer le modele et effectuer la classification des données
    def train_and_classify(self):

        # Feature extraction
        self.vectorizer = CountVectorizer()
        self.X_train_termcounts = self.vectorizer.fit_transform(self.corpus)
        self.shape = self.X_train_termcounts.shape

        # Training a classifier

        # tf-idf transformer
        self.tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = self.tfidf_transformer.fit_transform(self.X_train_termcounts)

        # split the dataset into test and train set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train_tfidf.todense(), self.target, test_size = self.test_size, random_state = 0)

        # Multinomial Naive Bayes classifier
        self.classifier = MultinomialNB().fit(self.X_train, self.y_train)

        # Predicting the Test set results
        self.y_pred = self.classifier.predict(self.X_test)

        # Making the Confusion Matrix
        self.cm = confusion_matrix(self.y_test, self.y_pred)


    # permet de lire fichier pdf passé en entrée
    def read_input_file(self,filepath):
        #pdf2txt.main(['', '-o', 'test.txt', filepath])
        texte = pdf2txt.convert_pdf_to_txt(filepath)
        # with open('test.txt', 'r', encoding="utf8") as mon_fichier:
        #     texte = mon_fichier.read()
        #     texte = str(texte)
        #     texte = texte.replace("'","")

        #effacer les precedents input
        self.input_data.clear()

        self.input_data.append(texte)

        self.clean_input()

    # permet de lire fichier txt passé en entrée
    def read_input_file_txt(self,filepath):
        with open(filepath, 'r') as mon_fichier:
            texte = mon_fichier.read()
            texte = str(texte)
            texte = texte.replace("'","")

        #effacer les precedents input
        self.input_data.clear()

        self.input_data.append(texte)

        self.clean_input()

    # permet de lire le texte passé en entrée
    def read_input_text(self,text):
        text = str(text)
        text = text.replace("'","")

        #effacer les precedents input
        self.input_data.clear()

        self.input_data.append(text)
        self.clean_input()

    # permet de pretraiter les données entrées
    def clean_input(self):
        size = len(self.input_data)
        #effacer les precedents input
        self.new_input_data.clear()
        i = 0
        while (i < size):
            #enlever les url dans les texte
            review = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", self.input_data[i])
            #enlever les nombres isolés(qui ne sont pas collés a un mot ou lettre)
            review = re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b","",review)
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)

            if (len(review.strip())!=0):
                self.new_input_data.append(review)
            i = i+1


    #predire les categories
    def predict(self):

        self.X_input_termcounts_inp = self.vectorizer.transform(self.new_input_data)

        self.X_input_tfidf_inp = self.tfidf_transformer.transform(self.X_input_termcounts_inp)

        # Predict the output categories
        self.predicted_categories = self.classifier.predict(self.X_input_tfidf_inp)

        #chercher tous les concepts
        self.find_concept()


    # permet de generer le modele ngram
    def words_to_ngrams(self, words, n, sep=" "):
        chaine = ""
        for x in range(1,n+1):
            for i in range(len(words)-x+1) :
                if (chaine ==""):
                    chaine = sep.join(words[i:i+x])
                else:
                    chaine = chaine+ self.pattern + sep.join(words[i:i+x])
        return chaine.split(self.pattern)



    def find_concept(self):

        self.thematics.clear()
        self.thematic.clear()
        self.thematics_cat.clear()

        for sentence, category in zip(self.new_input_data, self.predicted_categories):
            input_ngram = self.words_to_ngrams(sentence.split(' '), self.ngram_nbre_word, sep=" ")

            counter_cat = 0

            pos = category
            indice = self.ind_cat[pos]
            if (pos == 0):
                preced = 0
            else:
                preced = self.ind_cat[pos-1]

            cat_corpus = self.corpus[preced:indice]
            kk = preced
            for ca in input_ngram:
                for x in cat_corpus:
                    if (ca in x.split(self.pattern)):
                        if (ca not in self.thematic):
                            self.thematic.append(ca)
                            self.thematics.append(self.concepts[kk] + self.pattern + ca)
                            counter_cat += 1
                    kk = kk + 1
                kk = preced

            self.thematics_cat.append(counter_cat)

 ###############################################################################


  #recuperer la valeur d'un element de type str ou liste
    def recupVal(self,item):
        val = ""
        if (type(item) == str):
            item = item.strip()
            if(len(item) > 0):
                val = item
        else:
            ind_last = len(item)
            i = 0
            for el in item:
                i += 1
                if (i < ind_last):
                    val += self.enlv(el)+ self.pattern2
                else:
                    val += self.enlv(el)
        return val


 # rechercher dans une liste de dictionnaires
    def search_concept(self, key, value, list_of_dictionaries):
        return [element for element in list_of_dictionaries if element[key] == value]


    # recuperer les elements d'un concept
    def recupConcept(self,concept):
        val = ""
        synonyms = []

        obj = {'id':'','label':'','definition':'','comment':'','synonyms':'','ancestors':'','descendants':''}

        if (len(self.recupVal(concept.label)) > 0):
            if (len(val.strip()) > 0):
                val += self.pattern + self.recupVal(concept.label)
            else:
                if (len(val.strip()) == 0):
                    val += self.recupVal(concept.label)
            obj["id"] = self.recupVal(concept.id)
            obj["label"] = self.recupVal(concept.label)

            if (len(self.recupVal(concept.IAO_0000115)) > 0):
                if (len(val.strip()) > 0):
                    val += self.pattern + self.recupVal(concept.IAO_0000115)
                else:
                    if (len(val.strip()) == 0):
                        val += self.recupVal(concept.IAO_0000115)
                obj["definition"] = self.recupVal(concept.IAO_0000115)

            if (len(self.recupVal(concept.comment)) > 0):
                if (len(val.strip()) > 0):
                    val += self.pattern + self.recupVal(concept.comment)
                else:
                    if (len(val.strip()) == 0):
                        val += self.recupVal(concept.comment)
                obj["comment"] = self.recupVal(concept.comment)

            #synonyms

            if (len(self.recupVal(concept.hasNarrowSynonym)) > 0):
                if (len(val.strip()) > 0):
                    val += self.pattern + self.recupVal(concept.hasNarrowSynonym)

                    syns = self.recupVal(concept.hasNarrowSynonym)
                    syns = syns.split(self.pattern2)
                    for syn in syns:
                        synonyms.append(syn)

                else:
                    if (len(val.strip()) == 0):
                        val += self.recupVal(concept.hasNarrowSynonym)

                        syns = self.recupVal(concept.hasNarrowSynonym)
                        syns = syns.split(self.pattern2)
                        for syn in syns:
                            synonyms.append(syn)

            if (len(self.recupVal(concept.hasRelatedSynonym)) > 0):
                if (len(val.strip()) > 0):
                    val += self.pattern + self.recupVal(concept.hasRelatedSynonym)

                    syns = self.recupVal(concept.hasRelatedSynonym)
                    syns = syns.split(self.pattern2)
                    for syn in syns:
                        synonyms.append(syn)

                else:
                    if (len(val.strip()) == 0):
                        val += self.recupVal(concept.hasRelatedSynonym)

                        syns = self.recupVal(concept.hasRelatedSynonym)
                        syns = syns.split(self.pattern2)
                        for syn in syns:
                            synonyms.append(syn)

            if (len(self.recupVal(concept.hasExactSynonym)) > 0):
                if (len(val.strip()) > 0):
                    val += self.pattern + self.recupVal(concept.hasExactSynonym)

                    syns = self.recupVal(concept.hasExactSynonym)
                    syns = syns.split(self.pattern2)
                    for syn in syns:
                        synonyms.append(syn)

                else:
                    if (len(val.strip()) == 0):
                        val += self.recupVal(concept.hasExactSynonym)

                        syns = self.recupVal(concept.hasExactSynonym)
                        syns = syns.split(self.pattern2)
                        for syn in syns:
                            synonyms.append(syn)


            if (len(self.recupVal(concept.hasBroadSynonym)) > 0):
                if (len(val.strip()) > 0):
                    val += self.pattern + self.recupVal(concept.hasBroadSynonym)

                    syns = self.recupVal(concept.hasBroadSynonym)
                    syns = syns.split(self.pattern2)
                    for syn in syns:
                        synonyms.append(syn)

                else:
                    if (len(val.strip()) == 0):
                        val += self.recupVal(concept.hasBroadSynonym)

                        syns = self.recupVal(concept.hasBroadSynonym)
                        syns = syns.split(self.pattern2)
                        for syn in syns:
                            synonyms.append(syn)


            obj["synonyms"] = synonyms

            #recuperer les ancetres du concept
            obj["ancestors"] = self.recupAncetres(concept)

            #recuperer les descendants du concept
            obj["descendants"] = self.recupDescendants(concept)

        return val,obj



 #generer etiquette du concept
    def genererEtiquetteConcept(self,key,value):
        etiquette = []
        lst_concept = self.search_concept(key, value, self.ontology)
        for concept in lst_concept:
            etiquette.append(concept["label"])
            etiquette.append(concept["definition"])
            etiquette.append(concept["comment"])
            synonyms = concept["synonyms"]

            for el in synonyms:
                    etiquette.append(el)

        return etiquette

  #generer etiquette des parents d'un concept
    def genererEtiquetteParent(self, key, value):
        etiquette = []
        lst_concept = self.search_concept(key, value, self.ontology)
        for concept in lst_concept:

            ancetres = concept["ancestors"]
            for ancetre in ancetres:
                etiq_ancetre = self.genererEtiquetteConcept("id",ancetre)

                for el in etiq_ancetre:
                    etiquette.append(el)

            descendants = concept["descendants"]
            for descendant in descendants:
                etiq_descendant = self.genererEtiquetteConcept("id",descendant)

                for el in etiq_descendant:
                    etiquette.append(el)

        return etiquette


    def genererEtiquette(self, list_thematique):

        etiquette = []

        for thematique in list_thematique:
            et1 = self.genererEtiquetteConcept("label", thematique)
            et2 = self.genererEtiquetteParent("label", thematique)

            for el1 in et1:
                etiquette.append(el1)

            for el2 in et2:
                etiquette.append(el2)

        return etiquette




    #recuperer ancetres d'un concept
    def recupAncetres(self,concept):
        ancetres = concept.ancestors()
        anc = []
        label= str(concept.id)
        label = label.replace("[",'')
        label = label.replace("]",'')
        label = label.replace("'",'')

        for ancetre in ancetres:
            ancetre = str(ancetre)
            ancetre = ancetre[4:]
            if ("_" in ancetre):
                ancetre = ancetre.replace("_",":")
                if (ancetre != label):
                    anc.append(ancetre)

        return anc


    #recuperer descendant d'un concept
    def recupDescendants(self,concept):
        descendants = concept.descendants()
        desc = []
        label= str(concept.id)
        label = label.replace("[",'')
        label = label.replace("]",'')
        label = label.replace("'",'')

        for descendant in descendants:
            descendant = str(descendant)
            descendant = descendant[4:]
            if ("_" in descendant):
                descendant = descendant.replace("_",":")
                if (descendant != label):
                    desc.append(descendant)

        return desc



    def rechercherDocumentVoisin(self,liste_documents, document):

        similarity = []
        list_doc_sim = []

        if (document in liste_documents):
            index = liste_documents.index(document)

            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(liste_documents)
            xxx = cosine_similarity(tfidf_matrix[index:index+1], tfidf_matrix)
            for xx in xxx:
                for x in xx:
                    similarity.append(x)

            list_doc_sim = self.takeDocumentVoisin(liste_documents, similarity)

            similarity.sort(reverse=True)

        return similarity, list_doc_sim



    def takeDocumentVoisin(self,liste_documents, similarity):

        l_d = liste_documents.copy()
        sim = similarity.copy()
        list_doc_sim = []

        while (len(sim)>0):
            ind = sim.index(max(sim))
            list_doc_sim.append(l_d[ind])
            l_d.pop(ind)
            sim.pop(ind)


        return list_doc_sim


 ########################### accesseurs de la classe ###########################

    #retourne la matrice de confusion du model
    def _get_confusion_matrix(self):
        return self.cm

    #retourne les thematiques
    def _get_thematics(self):
        return self.thematics

    #retourne les concepts
    def _get_concepts(self):
        return self.concepts

    #retourne la taille des données de test
    def _get_test_size(self):
        return self.test_size

    #modfie la taille des données de test
    def _set_test_size(self, new_test_size):
        if (new_test_size <= 1):
            self.test_size =new_test_size

    #retourne les categories predites
    def _get_predicted_categories(self):
        return self.predicted_categories

    #retourne le nombre d'ontologies chargées
    def _get_number_ontologies(self):
        return number_ontologies


# import sys
# import os.path
# sys.path[0]

# mal=os.path.join(sys.path[0], "ontologia\ontologies\doid.owl.xml")


# mal = '/ontologia/ontologies/idomal.owl.xml'
# mal = os.path.dirname(mal)
# mal=os.path.join('/ontologia/ontologies/', 'idomal.owl.xml')




# input_data = [
# "Plants are mainly multicellular, predominantly photosynthetic eukaryotes of the kingdom Plantae. They form the clade Viridiplantae (Latin for \"green plants\") that includes the flowering plants, conifers and other gymnosperms, ferns, clubmosses, hornworts, liverworts mosses and the green algae, and excludes the red and brown algae. Historically, plants were treated as one of two kingdoms including, all living things that were not animals and all algae and fungi were treated as plants. However, all current definitions of Plantae exclude the fungi and some algae, as well as the prokaryotes (the archaea and bacteria). Green plants have cell walls containing cellulose and obtain most of their energy from sunlight via photosynthesis by primary chloroplasts that are derived from endosymbiosis with cyanobacteria. Their chloroplasts contain chlorophylls a and b, which gives them their green color. Some plants are secondarily parasitic or mycotrophic and may lose the ability to produce normal amounts of chlorophyll or to photosynthesize. Plants are characterized by sexual reproduction and alternation of generations, although asexual reproduction is also common.",
# "The underlying mechanisms vary depending on the disease in question.[2] Coronary artery disease, stroke, and peripheral artery disease involve atherosclerosis.[2] This may be caused by high blood pressure, smoking, diabetes, lack of exercise, obesity, high blood cholesterol, poor diet, and excessive alcohol consumption, among others.[2] High blood pressure results in 13% of CVD deaths, while tobacco results in 9%, diabetes 6%, lack of exercise 6% and obesity 5%.[2] Rheumatic heart disease may follow untreated strep throat.",
# "The blood film is the gold standard for malaria diagnosis. Ring-forms and gametocytes of Plasmodium falciparum in human blood Owing to the non-specific nature of the presentation of symptoms, diagnosis of malaria in non-endemic areas requires a high degree of suspicion, which might be elicited by any of the following: recent travel history, enlarged spleen, fever, low number of platelets in the blood, and higher-than-normal levels of bilirubin in the blood combined with a normal level of white blood cells.[5] Reports in 2016 and 2017 from countries were malaria is common suggest high levels of over diagnosis due to insufficient or inaccurate laboratory testing.[48][49][50] Malaria is usually confirmed by the microscopic examination of blood films or by antigen-based rapid diagnostic tests (RDT).[51][52] In some areas, RDTs need to be able to distinguish whether the malaria symptoms are caused by Plasmodium falciparum or by other species of parasites since treatment strategies could differ for non-P. falciparum infections.[53] Microscopy is the most commonly used method to detect the malarial parasite—about 165 million blood films were examined for malaria in 2010.[54] Despite its widespread usage, diagnosis by microscopy suffers from two main drawbacks: many settings (especially rural) are not equipped to perform the test, and the accuracy of the results depends on both the skill of the person examining the blood film and the levels of the parasite in the blood. The sensitivity of blood films ranges from 75–90% in optimum conditions, to as low as 50%. Commercially available RDTs are often more accurate than blood films at predicting the presence of malaria parasites, but they are widely variable in diagnostic sensitivity and specificity depending on manufacturer, and are unable to tell how many parasites are present.[54] In regions where laboratory tests are readily available, malaria should be suspected, and tested for, in any unwell person who has been in an area where malaria is endemic. In areas that cannot afford laboratory diagnostic tests, it has become common to use only a history of fever as the indication to treat for malaria—thus the common teaching \"fever equals malaria unless proven otherwise\". A drawback of this practice is overdiagnosis of malaria and mismanagement of non-malarial fever, which wastes limited resources, erodes confidence in the health care system, and contributes to drug resistance.[55] Although polymerase chain reaction-based tests have been developed, they are not widely used in areas where malaria is common as of 2012, due to their complexity.",
# "Heart and blood vessel disease — also called heart disease — includes numerous problems, many of which are related to a process called atherosclerosis. Atherosclerosis is a condition that develops when a substance called plaque builds up in the walls of the arteries. This buildup narrows the arteries, making it harder for blood to flow through. If a blood clot forms, it can stop the blood flow. This can cause a heart attack or stroke. A heart attack occurs when the blood flow to a part of the heart is blocked by a blood clot. If this clot cuts off the blood flow completely, the part of the heart muscle supplied by that artery begins to die. Most people survive their first heart attack and return to their normal lives to enjoy many more years of productive activity. But having a heart attack does mean you have to make some changes. The doctor will advise you of medications and lifestyle changes according to how badly the heart was damaged and what degree of heart disease caused the heart attack. Learn more about heart attack. An ischemic stroke (the most common type) happens when a blood vessel that feeds the brain gets blocked, usually from a blood clot. When the blood supply to a part of the brain is shut off, brain cells will die. The result will be the inability to carry out some of the previous functions as before like walking or talking. A hemorrhagic stroke occurs when a blood vessel within the brain bursts. The most likely cause is uncontrolled hypertension (blood pressure). Some effects of stroke are permanent if too many brain cells die after a stroke due to lack of blood and oxygen to the brain. These cells are never replaced. The good news is that some brain cells don't die — they're only temporarily out of order. Injured cells can repair themselves. Over time, as the repair takes place, some body functioning improves. Also, other brain cells may take control of those areas that were injured. In this way, strength may improve, speech may get better and memory may improve. This recovery process is what rehabilitation is all about. Learn more about stroke. Other Types of Cardiovascular Disease Heart failure: This doesn\'t mean that the heart stops beating. Heart failure, sometimes called congestive heart failure, means the heart isn't pumping blood as well as it should. The heart keeps working, but the body's need for blood and oxygen isn't being met. Heart failure can get worse if it's not treated. If your loved one has heart failure, it's very important to follow the doctor's orders. Learn more about heart failure. Arrhythmia: This is an abnormal rhythm of the heart. There are various types of arrhythmias. The heart can beat too slow, too fast or irregularly. Bradycardia is when the heart rate is less than 60 beats per minute. Tachycardia is when the heart rate is more than 100 beats per minute. An arrhythmia can affect how well the heart works. The heart may not be able to pump enough blood to meet the body's needs. Learn more about arrhythmia. Heart valve problems: When heart valves don\'t open enough to allow the blood to flow through as it should, it's called stenosis. When the heart valves don't close properly and allow blood to leak through, it's called regurgitation. When the valve leaflets bulge or prolapse back into the upper chamber, it’s a condition called prolapse. Discover more about the roles your heart valves play in healthy circulation and learn more about heart valve disease. Cardiovascular Disease	Treatment Heart Valve Problems	Medications Heart Valve Surgery Arrhythmia	Medications Pacemaker Heart Attack	Medications — clotbusters (should be administered as soon as possible for certain types of heart attacks) Coronary Angioplasty Coronary Artery Bypass Graft Surgery Stroke	Medications — clotbusters (must be administered within 3 hours from onset of stroke symptoms for certain types of strokes, see Stroke Treatments) Carotid Endarterectomy (PDF) Diagnostic Tests, Surgical Procedures and Medications In the hospital and during the first few weeks at home, your loved one\'s doctor may perform several tests and procedures. These tests help the doctor determine what caused the stroke or heart attack and how much damage was done. Some tests monitor progress to see if treatment is working. Learn more about diagnostic tests and procedures. Your loved one may have undergone additional surgical procedures. Learn more about cardiac procedures and surgeries. Your first goal is to help your loved one enjoy life again and work to prevent another stroke or heart attack. As a caregiver, you're responsible for helping your loved one take medications as directed and on time. Find out about the new medications your loved one must take. Know what they're for and what they do. It's important to follow your doctor\'s directions closely, so ask questions and take notes. Learn more about cardiac medications. This content was last reviewed May 2017."
# ]

# input_texte1 = "Plants are mainly multicellular, predominantly photosynthetic eukaryotes of the kingdom Plantae. They form the clade Viridiplantae (Latin for \"green plants\") that includes the flowering plants, conifers and other gymnosperms, ferns, clubmosses, hornworts, liverworts mosses and the green algae, and excludes the red and brown algae. Historically, plants were treated as one of two kingdoms including, all living things that were not animals and all algae and fungi were treated as plants. However, all current definitions of Plantae exclude the fungi and some algae, as well as the prokaryotes (the archaea and bacteria). Green plants have cell walls containing cellulose and obtain most of their energy from sunlight via photosynthesis by primary chloroplasts that are derived from endosymbiosis with cyanobacteria. Their chloroplasts contain chlorophylls a and b, which gives them their green color. Some plants are secondarily parasitic or mycotrophic and may lose the ability to produce normal amounts of chlorophyll or to photosynthesize. Plants are characterized by sexual reproduction and alternation of generations, although asexual reproduction is also common.",
# input_texte2 = "The underlying mechanisms vary depending on the disease in question.[2] Coronary artery disease, stroke, and peripheral artery disease involve atherosclerosis.[2] This may be caused by high blood pressure, smoking, diabetes, lack of exercise, obesity, high blood cholesterol, poor diet, and excessive alcohol consumption, among others.[2] High blood pressure results in 13% of CVD deaths, while tobacco results in 9%, diabetes 6%, lack of exercise 6% and obesity 5%.[2] Rheumatic heart disease may follow untreated strep throat.",
# input_texte3 = "The blood film is the gold standard for malaria diagnosis. Ring-forms and gametocytes of Plasmodium falciparum in human blood Owing to the non-specific nature of the presentation of symptoms, diagnosis of malaria in non-endemic areas requires a high degree of suspicion, which might be elicited by any of the following: recent travel history, enlarged spleen, fever, low number of platelets in the blood, and higher-than-normal levels of bilirubin in the blood combined with a normal level of white blood cells.[5] Reports in 2016 and 2017 from countries were malaria is common suggest high levels of over diagnosis due to insufficient or inaccurate laboratory testing.[48][49][50] Malaria is usually confirmed by the microscopic examination of blood films or by antigen-based rapid diagnostic tests (RDT).[51][52] In some areas, RDTs need to be able to distinguish whether the malaria symptoms are caused by Plasmodium falciparum or by other species of parasites since treatment strategies could differ for non-P. falciparum infections.[53] Microscopy is the most commonly used method to detect the malarial parasite—about 165 million blood films were examined for malaria in 2010.[54] Despite its widespread usage, diagnosis by microscopy suffers from two main drawbacks: many settings (especially rural) are not equipped to perform the test, and the accuracy of the results depends on both the skill of the person examining the blood film and the levels of the parasite in the blood. The sensitivity of blood films ranges from 75–90% in optimum conditions, to as low as 50%. Commercially available RDTs are often more accurate than blood films at predicting the presence of malaria parasites, but they are widely variable in diagnostic sensitivity and specificity depending on manufacturer, and are unable to tell how many parasites are present.[54] In regions where laboratory tests are readily available, malaria should be suspected, and tested for, in any unwell person who has been in an area where malaria is endemic. In areas that cannot afford laboratory diagnostic tests, it has become common to use only a history of fever as the indication to treat for malaria—thus the common teaching \"fever equals malaria unless proven otherwise\". A drawback of this practice is overdiagnosis of malaria and mismanagement of non-malarial fever, which wastes limited resources, erodes confidence in the health care system, and contributes to drug resistance.[55] Although polymerase chain reaction-based tests have been developed, they are not widely used in areas where malaria is common as of 2012, due to their complexity.",
# input_texte4 = "Heart and blood vessel disease — also called heart disease — includes numerous problems, many of which are related to a process called atherosclerosis. Atherosclerosis is a condition that develops when a substance called plaque builds up in the walls of the arteries. This buildup narrows the arteries, making it harder for blood to flow through. If a blood clot forms, it can stop the blood flow. This can cause a heart attack or stroke. A heart attack occurs when the blood flow to a part of the heart is blocked by a blood clot. If this clot cuts off the blood flow completely, the part of the heart muscle supplied by that artery begins to die. Most people survive their first heart attack and return to their normal lives to enjoy many more years of productive activity. But having a heart attack does mean you have to make some changes. The doctor will advise you of medications and lifestyle changes according to how badly the heart was damaged and what degree of heart disease caused the heart attack. Learn more about heart attack. An ischemic stroke (the most common type) happens when a blood vessel that feeds the brain gets blocked, usually from a blood clot. When the blood supply to a part of the brain is shut off, brain cells will die. The result will be the inability to carry out some of the previous functions as before like walking or talking. A hemorrhagic stroke occurs when a blood vessel within the brain bursts. The most likely cause is uncontrolled hypertension (blood pressure). Some effects of stroke are permanent if too many brain cells die after a stroke due to lack of blood and oxygen to the brain. These cells are never replaced. The good news is that some brain cells don't die — they're only temporarily out of order. Injured cells can repair themselves. Over time, as the repair takes place, some body functioning improves. Also, other brain cells may take control of those areas that were injured. In this way, strength may improve, speech may get better and memory may improve. This recovery process is what rehabilitation is all about. Learn more about stroke. Other Types of Cardiovascular Disease Heart failure: This doesn\'t mean that the heart stops beating. Heart failure, sometimes called congestive heart failure, means the heart isn't pumping blood as well as it should. The heart keeps working, but the body's need for blood and oxygen isn't being met. Heart failure can get worse if it's not treated. If your loved one has heart failure, it's very important to follow the doctor's orders. Learn more about heart failure. Arrhythmia: This is an abnormal rhythm of the heart. There are various types of arrhythmias. The heart can beat too slow, too fast or irregularly. Bradycardia is when the heart rate is less than 60 beats per minute. Tachycardia is when the heart rate is more than 100 beats per minute. An arrhythmia can affect how well the heart works. The heart may not be able to pump enough blood to meet the body's needs. Learn more about arrhythmia. Heart valve problems: When heart valves don\'t open enough to allow the blood to flow through as it should, it's called stenosis. When the heart valves don't close properly and allow blood to leak through, it's called regurgitation. When the valve leaflets bulge or prolapse back into the upper chamber, it’s a condition called prolapse. Discover more about the roles your heart valves play in healthy circulation and learn more about heart valve disease. Cardiovascular Disease	Treatment Heart Valve Problems	Medications Heart Valve Surgery Arrhythmia	Medications Pacemaker Heart Attack	Medications — clotbusters (should be administered as soon as possible for certain types of heart attacks) Coronary Angioplasty Coronary Artery Bypass Graft Surgery Stroke	Medications — clotbusters (must be administered within 3 hours from onset of stroke symptoms for certain types of strokes, see Stroke Treatments) Carotid Endarterectomy (PDF) Diagnostic Tests, Surgical Procedures and Medications In the hospital and during the first few weeks at home, your loved one\'s doctor may perform several tests and procedures. These tests help the doctor determine what caused the stroke or heart attack and how much damage was done. Some tests monitor progress to see if treatment is working. Learn more about diagnostic tests and procedures. Your loved one may have undergone additional surgical procedures. Learn more about cardiac procedures and surgeries. Your first goal is to help your loved one enjoy life again and work to prevent another stroke or heart attack. As a caregiver, you're responsible for helping your loved one take medications as directed and on time. Find out about the new medications your loved one must take. Know what they're for and what they do. It's important to follow your doctor\'s directions closely, so ask questions and take notes. Learn more about cardiac medications. This content was last reviewed May 2017."


# o = Ontology()

# o.load_ontologies()

# #o.read_input_file(input_file)

# o.read_input_text(input_texte1)

# o.predict()
