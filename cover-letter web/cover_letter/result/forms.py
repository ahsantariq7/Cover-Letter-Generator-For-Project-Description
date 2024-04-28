from django import forms
from .models import CoverLetterGenerator


class CoverLetterGeneratorForm(forms.ModelForm):
    class Meta:
        model = CoverLetterGenerator
        fields = "__all__"
