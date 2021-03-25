from rest_framework import serializers
from .models import RequestModel


class RequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = RequestModel
        fields = '__all__'
