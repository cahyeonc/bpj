from django.urls import path
from . import views

urlpatterns = [
    path('', views.index2),
    path('mpmodel/', views.mp_model),
     
]
