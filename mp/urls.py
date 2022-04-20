from django.urls import path
from . import views
app_name = 'mp'
urlpatterns = [
    path('', views.index2),
    path('mpmodel/', views.mp_model, name = 'model'),
]
