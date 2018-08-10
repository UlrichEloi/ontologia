from django.db import models

from .validators import validate_file_extension

# Create your models here.


def upload_location(instance,filename):
	val = filename.split(".")
	extension = val[len(val)-1]
	return "%s.%s" % (instance.nom,extension)

class TimestampModel(models.Model):
	"""docstring for TimestamtedModel"""

	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		abstract = True


class Fichier(TimestampModel):
	"""docstring for Fichier"""

	nom = models.CharField(max_length=100)
	typ = models.CharField(max_length=100)
	categorie = models.CharField(max_length=100)
	contenu = models.TextField()
	thematiques = models.TextField()
	occurence_thematiques = models.TextField(default = "",null=True)
	presence_thematiques = models.TextField(default = "",null=True)
	document = models.FileField(upload_to = upload_location,null=True,blank=True,validators=[validate_file_extension])
	etiquettes = models.TextField()


	def __str__(self):
		return self.nom

	def concepts_as_list(self):
		concepts = []
		raisons = []
		if (len(self.thematiques)> 0):
			them = self.thematiques.split("²²²")
			if ( len(them) > 0):

				for th in them:
					ch = th.split("~~~")
					if ( len(ch) >= 1):
						concepts.append(str(ch[0]))
						raisons.append(str(ch[1]))

		return zip(concepts,raisons)

	# def raisons_as_list(self):
	# 	concepts = []
	# 	raisons = []
	# 	them = self.thematiques.split("²²²")
	# 	for th in them:
	# 		ch = th.split("~~~")
	# 		concepts.append(str(ch[0]))
	# 		raisons.append(str(ch[1]))

	# 	return concepts,raisons
