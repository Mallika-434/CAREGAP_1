from django.urls import path
from . import views

urlpatterns = [
    path('suggest/', views.generate_suggestions, name='rag-suggest'),
    path('status/',  views.rag_status,            name='rag-status'),
]
