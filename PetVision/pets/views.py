from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

def pets(request):
    template = loader.get_template("landing_page.html")
    return HttpResponse(template.render())

def training(request):
    template = loader.get_template("training.html")
    return HttpResponse(template.render())

def prediction(request):
    template = loader.get_template("prediction.html")
    return HttpResponse(template.render())


