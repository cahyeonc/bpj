from django.db import models


# Create your models here.
class WToS(models.Model):
    text = models.CharField(max_length=100)
