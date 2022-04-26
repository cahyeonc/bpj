from django.urls import path
from . import views

app_name = 'mp'
urlpatterns = [
    path('home/', views.home, name ='home'),
    path('home/detectme/', views.detectme, name ='detectme'),
    path('', views.show, name ='show'),
    path('mpmodel/', views.mp_model, name = 'model'),
]
