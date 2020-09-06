from django.db import models

class AccidentClaim(models.Model):
    vehicle_image = models.ImageField(upload_to='images/vehicle/%Y/%m/%d/')
    damaged_image = models.ImageField(upload_to='images/damaged/%Y/%m/%d/')

    objects = models.Manager()
