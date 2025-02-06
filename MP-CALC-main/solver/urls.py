from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('graphical_solve/', views.graphical_solve, name='graphical_solve'),
    path('simplex_solve/', views.simplex_solver, name='simplex_solve'),
    path('transportation_solve/', views.transportation_solver, name='transportation_solve'),
]
