from django.urls import path
from . import views
from patients.views import explain_result

urlpatterns = [
    path('suggest/', views.generate_suggestions, name='rag-suggest'),
    path('status/',  views.rag_status,            name='rag-status'),
    path('explain/', explain_result,              name='rag-explain'),
]
