import os
import time
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
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

def simulate_prediction(request):
    if request.method == "POST":
        time.sleep(5)

        
        file_name = request.POST.get("file_name", "").lower()

        if not file_name:
            return JsonResponse({"error": "No file name provided"}, status=400)
        if "cat" in file_name:
            result = {"prediction": "Cat", "confidence": "95%"}
        elif "dog" in file_name:
            result = {"prediction": "Dog", "confidence": "92%"}
        else:
            result = {"prediction": "Unknown", "confidence": "N/A"}

        return JsonResponse(result)
    return JsonResponse({"error": "Invalid request method"}, status=400)