from django import forms
from .models import AccidentClaim

class AccidentClaimForm(forms.ModelForm):

    class Meta:
        model = AccidentClaim
        fields = ('vehicle_image', 'damaged_image',)