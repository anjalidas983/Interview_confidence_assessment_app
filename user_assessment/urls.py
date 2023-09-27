from django.urls import path
from . import views
app_name="user_assessment"

urlpatterns = [
    path('',views.home,name='home'),
    path('user-signup/',views.user_signup,name="user_signup"),
    path('start-assessment/',views.start_assessment,name="start_assessment"),
    path('user-logout/',views.user_logout,name='user_logout')
]