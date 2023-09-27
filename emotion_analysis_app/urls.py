from django.urls import path
from . import views
app_name="emotion_analysis_app"

urlpatterns = [
    path('',views.emotion_analysis,name="emotion_analysis"),
]