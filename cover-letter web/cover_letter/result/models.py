from django.db import models


# Create your models here.
class CoverLetterGenerator(models.Model):
    project_description = models.CharField(max_length=2000)
