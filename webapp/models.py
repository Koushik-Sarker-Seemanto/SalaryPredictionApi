from django.db import models

# Create your models here.


class RequestModel(models.Model):
    java = models.IntegerField()

    def __str__(self):
        return self.java
