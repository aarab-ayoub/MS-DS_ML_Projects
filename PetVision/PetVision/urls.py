from django.contrib import admin
from django.urls import path
from pets.views import pets, training, prediction, simulate_prediction
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", pets, name="landing_page"),
    path("training/", training, name="training_page"),
    path("prediction/", prediction, name="prediction_page"),
    path("simulate-prediction/", simulate_prediction, name="simulate_prediction"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
