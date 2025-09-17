from .models import ConcreteModel
from dvadmin.utils.serializers import CustomModelSerializer


class ConcreteModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = ConcreteModel
        fields = "__all__"


class ConcreteModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = ConcreteModel
        fields = '__all__'
