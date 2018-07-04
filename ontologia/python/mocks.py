from django.http import Http404

class Fichier():


	Fichiers = [
		{"id" : 1, "title":"first fichier", "body": "this is my first fichier"},
		{"id" : 2, "title":"second fichier", "body": "this is my second fichier"},
		{"id" : 3, "title":"first fichier", "body": "this is my third fichier"},
	]

	@classmethod
	def all(cls):
		return cls.Fichiers

	@classmethod
	def find(cls, id):
		try:
			return cls.Fichiers[int(id)-1]
		except:
			raise Http404('desolé, fichier #{} introuvable'.format(id))
		