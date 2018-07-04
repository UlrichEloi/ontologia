from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def home(request):
	return render(request, "home.html")

def about(request):
	return render(request, "pages/about.html")

def contact(request):
	return render(request, "pages/contact.html")

def login(request):
	return render(request, "registration/login.html")