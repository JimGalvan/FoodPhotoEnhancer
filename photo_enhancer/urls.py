from django.urls import path
from .views import index, upload_photo, enhance_photo

urlpatterns = [
    path('', index, name='index'),
    path('upload_photo', upload_photo, name='upload_photo'),
    path('enhance_photo', enhance_photo, name='enhance_photo'),
]
