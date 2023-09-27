from django.db import models

# Create your models here.
class CapturedImage(models.Model):
    image = models.ImageField(upload_to='captured_images/')
    emotion = models.CharField(max_length=300)
    confidence = models.FloatField()

    def __str__(self):
        return f"{self.emotion} {self.confidence:.2f}"