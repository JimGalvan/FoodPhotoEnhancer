from django.urls import path
from .views import index, upload_photo

urlpatterns = [
    path('', index, name='index'),
    path('upload/', upload_photo, name='upload_photo'),
]
