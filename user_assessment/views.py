from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from . forms import UserSignupForm
from django.contrib.auth.decorators import login_required

# Create your views here.
def home(request):
    error=None
    if request.method=="POST":
        username=request.POST['username']
        password =request.POST['password']
        user =authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect('user_assessment:start_assessment')
        else:
            error='Invalid Credentials.'
        
    return render(request,'home.html',{'error':error})


def user_signup(request):
    error=None
    if request.method=="POST":
        form=UserSignupForm(request.POST)
        if form.is_valid():
            username=request.POST['username']
            password =request.POST['password']
            confirmpassword=request.POST['confirmpassword']
            email=request.POST['email']
            if password==confirmpassword:
                user =User.objects.create_user(username=username,password=password,email=email)
                user.save()
                login(request,user)
                return redirect('user_assessment:home')
            else:
                error="Password & Confirm Password must match."
        else:
            error="Invalid Datas"

    return render(request,'user_signup.html',{'error':error})
@login_required
def start_assessment(request):
    return render(request,'start-assessment.html')

def user_logout(request):
    logout(request)
    return redirect('user_assessment:home')