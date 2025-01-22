from django.urls import path
from .views import PrediccionView

urlpatterns = [
    path('predict/', PrediccionView.as_view(), name='predict'),
]
