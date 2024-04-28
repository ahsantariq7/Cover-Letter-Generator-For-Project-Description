from django.urls import path
from .views import CoverLetterView, HomeView

urlpatterns = [
    path("cover/", CoverLetterView.as_view(), name="cover"),
    path("", HomeView.as_view(), name="home"),
]
