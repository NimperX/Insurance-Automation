from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import User, UserData, ClaimData

class CustomUserCreationForm(UserCreationForm):

    class Meta:
        model = User
        fields = ('username','password')

class CustomUserChangeForm(UserChangeForm):

    class Meta:
        model = User
        fields = ('username','password')

        widgets = {
            'username': forms.TextInput(attrs={'class':'form-control custom-input'}),
            'password': forms.TextInput(attrs={'class':'form-control custom-input'})
        }

class UserDataForm(forms.ModelForm):

    class Meta:
        model = UserData
        fields = ('First_Name','Last_Name','Gender','Age','Email','Contact_Number','Marital_Status',)
        widgets = {
            'First_Name': forms.TextInput(attrs={'class':'form-control custom-input'}),
            'Last_Name': forms.TextInput(attrs={'class':'form-control custom-input'}),
            'Gender': forms.RadioSelect(choices=(('Male', 'Male'),('Female', 'Female'))),
            'Age': forms.NumberInput(attrs={'class':'form-control custom-input'}),
            'Email': forms.TextInput(attrs={'class':'form-control custom-input'}),
            'Contact_Number': forms.TextInput(attrs={'class':'form-control custom-input'}),
            'Marital_Status': forms.RadioSelect(choices=(('Married', 'Married'),('Unmarried', 'Unmarried'))),
        }