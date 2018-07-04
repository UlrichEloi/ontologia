from django.urls import path

from . import views

app_name = 'ontologia'

urlpatterns  = [
	path('', views.index, name = 'index'),
	path('fichiers/', views.all, name = 'all'),
	path('fichiers/<int:id>/', views.show, name = 'show'),
	path('about/', views.about, name = 'about'),
	path('contact/', views.contact, name = 'contact'),
	path('fragment/', views.fragment, name = 'fragment'),
	


]
