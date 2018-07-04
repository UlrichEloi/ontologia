from django.shortcuts import render, get_object_or_404

# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect

#from ontologia.python.mocks import Fichier
from .models import Fichier
from django.http import Http404

from django.urls import reverse

from .forms import FichierForm,FragmentTexteForm

import os
from django.conf import settings


from ontologia.python.ontology_class import Ontology

input_texte1 = "Plants are mainly multicellular, predominantly photosynthetic eukaryotes of the kingdom Plantae. They form the clade Viridiplantae (Latin for \"green plants\") that includes the flowering plants, conifers and other gymnosperms, ferns, clubmosses, hornworts, liverworts mosses and the green algae, and excludes the red and brown algae. Historically, plants were treated as one of two kingdoms including, all living things that were not animals and all algae and fungi were treated as plants. However, all current definitions of Plantae exclude the fungi and some algae, as well as the prokaryotes (the archaea and bacteria). Green plants have cell walls containing cellulose and obtain most of their energy from sunlight via photosynthesis by primary chloroplasts that are derived from endosymbiosis with cyanobacteria. Their chloroplasts contain chlorophylls a and b, which gives them their green color. Some plants are secondarily parasitic or mycotrophic and may lose the ability to produce normal amounts of chlorophyll or to photosynthesize. Plants are characterized by sexual reproduction and alternation of generations, although asexual reproduction is also common.",


o = Ontology()
o.load_ontologies()



def all(request):
	
	fichiers = Fichier.objects.all().order_by('-id')
	
	
	return render(request,"ontologia/pages/all.html",{'fichiers':fichiers})

def show(request, id):
	
	fichier = get_object_or_404(Fichier, pk=id)

	name = fichier.nom
	typ = fichier.typ
	thematics = fichier.thematiques.split("²²²")
	# if(str(thematics).strip() == ""):review.split()
	# 	o.predict()
	# 	thematics = o.thematiques
	contenu = fichier.contenu
	categorie = fichier.categorie
	etiquettes = fichier.etiquettes.split("²²²")

	# try:
	# 	fichier = Fichier.objects.get(pk=id)
	# except Fichier.DoesNotExist:
	# 	raise Http404('desolé, fichier #{} introuvable'.format(id))
	
	return render(request, 'ontologia/pages/show.html',{'fichier': fichier,'thematiques':thematics,'typ':typ, 'categorie': categorie, 'etiquettes':etiquettes})


#pour la creation des fichiers pdf et txt
def index(request):

	if (request.POST):
		form = FichierForm(request.POST, request.FILES)
		if (form.is_valid()):

			form.save()

			nom = form.cleaned_data['nom']
			fichier = Fichier.objects.get(nom=nom)

			filename = str(fichier.document)

			# recuper l'extension du fichier
			val = filename.split(".")
			extension = val[len(val)-1]

			MEDIA_ROOT = str(getattr(settings, "MEDIA_ROOT", None))
			filename = os.path.join(MEDIA_ROOT, filename)

			if (str(extension) == "txt" ):
				o.read_input_file_txt(filename)
				fichier.typ = "txt"

			if (str(extension) == "pdf" ):
				o.read_input_file(filename)
				fichier.typ = "pdf"

			
			#o.read_input_file(filename)
			#o.read_input_text(filename)
			#o.read_input_file_txt(filename)

			o.predict()
			if (len(o.thematics)==0):
				fichier.thematiques = ""
				fichier.categorie = "unknown"
			else:
				fichier.thematiques = '²²²'.join(o.thematics)
				predict_cat = o.predicted_categories
				fichier.categorie = o.target_names[int(predict_cat)]
				fichier.etiquettes = '²²²'.join(o.genererEtiquette(o.thematics))
			fichier.contenu = o.input_data[0].strip()
			
			
			fichier.save()

			url = reverse('ontologia:show', kwargs={'id': fichier.id})
			return HttpResponseRedirect(url)


			#return render(request,"ontologia/pages/show.html", {'id': fichier.id})
	else:
		form = FichierForm()
	return render(request,"ontologia/pages/index.html", {'form': form})



#pour la creation des fragments de texte
def fragment(request):

	if (request.POST):
		form = FragmentTexteForm(request.POST)
		if (form.is_valid()):

			form.save()

			nom = form.cleaned_data['nom']
			fichier = Fichier.objects.get(nom=nom)

			fichier.document = nom + ".txt"

			
			fichier.type = "fragment"
			texte = form.cleaned_data['contenu']
			o.read_input_text(texte)
			

			o.predict()
			if (len(o.thematics)==0):
				fichier.thematiques = ""
				fichier.categorie = "unknown"
			else:
				fichier.thematiques = '²²²'.join(o.thematics)
				thematiques = str(fichier.thematiques)
				thematiques = thematiques.split('²²²')
				etiquettes = o.genererEtiquette(thematiques)
				predict_cat = o.predicted_categories
				fichier.categorie = o.target_names[int(predict_cat)]
				fichier.etiquettes = '²²²'.join(etiquettes)
			fichier.contenu = o.input_data[0].strip()
			
			
			fichier.save()

			url = reverse('ontologia:show', kwargs={'id': fichier.id})
			return HttpResponseRedirect(url)


			#return render(request,"ontologia/pages/show.html", {'id': fichier.id})
	else:
		form = FragmentTexteForm()
	return render(request,"ontologia/pages/fragment.html", {'form': form})

def about(request):
	return render(request, "ontologia/pages/about.html")

def contact(request):
	return render(request, "ontologia/pages/contact.html")


def handler404(request):
	return render(request, 'ontologia/errors/404.html',{}, status=404)

def handler500(request):
	return render(request, 'ontologia/errors/500.html',{}, status=500)

