from django.contrib import admin
from django.urls import path

from webapp import views

app_name = 'webapp'

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('health/', views.health, name='health'),
]