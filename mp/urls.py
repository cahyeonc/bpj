from django.urls import path
from . import views

app_name = 'mp'
urlpatterns = [
    path('str/', views.home, name ='home'), # 웹 스트리밍 
    path('str/signlanguage/', views.signlanguage, name ='signlanguage'), # 테스트
    path('', views.show, name ='show'), # 웹캠 실행 전 메인?
    path('mpmodel/', views.mp_model, name = 'model'), # 웹캠 실행
    path('str/test/', views.mp_model, name = 'test'), # 웹캠 실행
]
