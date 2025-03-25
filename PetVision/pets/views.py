import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

def pets(request):
    template = loader.get_template("landing_page.html")
    return HttpResponse(template.render())

def training(request):
    
    
    # cat_images = os.listdir(os.path.join(settings.MEDIA_ROOT, 'cats'))
    # dog_images = os.listdir(os.path.join(settings.MEDIA_ROOT, 'dogs'))
    cat_images = sorted(os.listdir(os.path.join(settings.MEDIA_ROOT, 'cats')))[:5]
    dog_images = sorted(os.listdir(os.path.join(settings.MEDIA_ROOT, 'dogs')))[:5]
    
    context = {
        'cat_images': cat_images,
        'dog_images': dog_images,
        'MEDIA_URL': settings.MEDIA_URL,
    }
    return render(request, 'training.html', context)

def prediction(request):
    template = loader.get_template("prediction.html")
    return HttpResponse(template.render())