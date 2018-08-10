from django.urls import path

from . import views

app_name = 'ontologia'

urlpatterns  = [
	path('', views.creer_fichier_pdf_txt, name = 'creer_fichier_pdf_txt'),
	path('fichiers/', views.all, name = 'all'),
	path('fichiers/<int:id>/', views.show, name = 'show'),
	path('about/', views.about, name = 'about'),
	path('contact/', views.contact, name = 'contact'),
	path('extrait-texte/', views.creer_extrait_texte, name = 'creer_extrait_texte'),
	path('graphique/', views.chart_view, name = 'chart_view'),
	path('api/chart/data/', views.ChartData.as_view(), name = 'api_chartdata'),


]
