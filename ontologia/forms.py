from django import forms
from .models import Fichier

class FichierForm(forms.ModelForm):
	"""docstring for FichierForm"""


	nom = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control','placeholder': 'Entrer le nom'}))
	document = forms.FileField()


	class Meta:
		model = Fichier
		fields = ['nom','document']


class FragmentTexteForm(forms.ModelForm):
	"""docstring for FichierForm"""

	nom = forms.CharField(label='Nom', widget=forms.TextInput(attrs={'class':'form-control','placeholder': 'Entrer le nom'}))
	contenu = forms.Textarea(attrs={'placeholder': 'Coller un Fragment de texte ici','cols': 200, 'rows': 100})
    

	class Meta:
		model = Fichier
		fields = ['nom','contenu']

		


	