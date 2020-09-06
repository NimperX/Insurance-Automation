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
            'username': forms.TextInput(attrs={'class':'form-control'}),
            # 'firstname': forms.TextInput(attrs={'class':'form-control'}),
            # 'lastname': forms.TextInput(attrs={'class':'form-control'}),
            # 'contact_no': forms.TextInput(attrs={'class':'form-control'}),
            # 'credit_source': forms.Textarea(attrs={'class':'form-control'}),
            # 'salary': forms.TextInput(attrs={'class':'form-control'}),
            # 'marital_status': forms.RadioSelect(choices=((True, 'Yes'),(False, 'No'))),
            'password': forms.TextInput(attrs={'class':'form-control'})
        }

class UserDataForm(forms.ModelForm):

    class Meta:
        model = UserData
        fields = ('First_Name','Last_Name','Gender','Age','Email','Contact_Number','Marital_Status',)
        widgets = {
            'First_Name': forms.TextInput(attrs={'class':'form-control'}),
            'Last_Name': forms.TextInput(attrs={'class':'form-control'}),
            'Gender': forms.RadioSelect(choices=(('Male', 'Male'),('Female', 'Female'))),
            'Age': forms.NumberInput(attrs={'class':'form-control'}),
            'Email': forms.TextInput(attrs={'class':'form-control'}),
            'Contact_Number': forms.TextInput(attrs={'class':'form-control'}),
            'Marital_Status': forms.RadioSelect(choices=(('Married', 'Married'),('Unmarried', 'Unmarried'))),
        }