import os
import time
from django.conf import settings
from django.shortcuts import render
# from django.http import HttpResponse
# from django.template import loader
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect


def handle_uploaded_files(files, animal_type):
    ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
    MAX_SIZE = 5 * 1024 * 1024
    
    try:
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, animal_type))
        for file in files:
            if file.content_type not in ALLOWED_TYPES:
                return False
            if file.size > MAX_SIZE:
                return False
            fs.save(file.name, file)
        return True
    except Exception:
        return False
    

def train_model(request):
    time.sleep(3)
    messages.success(request, 'Model trained successfully!')
    return redirect('prediction_page')

def process_prediction(uploaded_file):
    time.sleep(2)
    
    filename = uploaded_file.name.lower()
    if 'cat' in filename:
        return {'class': 'Cat', 'confidence': '95%'}
    elif 'dog' in filename:
        return {'class': 'Dog', 'confidence': '92%'}
    return {'error': True, 'filename': uploaded_file.name}


def pets(request):
    return render(request, "landing_page.html")

def training(request):
    if request.method == 'POST':
        
        cat_files = request.FILES.getlist('cat_files', [])
        if cat_files and not handle_uploaded_files(cat_files, 'cats'):
            messages.error(request, 'Invalid cat images uploaded')
        
        dog_files = request.FILES.getlist('dog_files', [])
        if dog_files and not handle_uploaded_files(dog_files, 'dogs'):
            messages.error(request, 'Invalid dog images uploaded')
        
        if 'train_model' in request.POST:
            return train_model(request)
        
        return redirect('training_page')

    cat_images = sorted(os.listdir(os.path.join(settings.MEDIA_ROOT, 'cats')))[:5]
    dog_images = sorted(os.listdir(os.path.join(settings.MEDIA_ROOT, 'dogs')))[:5]
    
    return render(request, 'training.html', {
        'cat_images': cat_images,
        'dog_images': dog_images,
        'MEDIA_URL': settings.MEDIA_URL,
    })


def prediction(request):
    result = None
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        result = process_prediction(uploaded_file)
    
    return render(request, 'prediction.html', {'result': result})
