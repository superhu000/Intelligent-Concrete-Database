from django.db import models


class ConcreteModel(models.Model):
    model_id = models.IntegerField(db_column='model_id', primary_key=True)
    model_name = models.CharField(max_length=100, blank=True, null=True)
    model_type = models.CharField(max_length=100, blank=True, null=True)
    feature_output = models.CharField(max_length=100, blank=True, null=True)
    input = models.TextField()
    sampleNum = models.IntegerField()
    model_R2 = models.CharField(max_length=100, blank=True, null=True)
    model_RMSE = models.CharField(max_length=100, blank=True, null=True)
    model_MSE = models.CharField(max_length=100, blank=True, null=True)
    model_MAE = models.CharField(max_length=100, blank=True, null=True)
    model_parameter = models.TextField()
    # model_performance = models.TextField()
    create_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False  # 表示 Django 不会自动创建和管理与这个模型关联的数据库表
        db_table = 'concrete_model'  # 表示这个模型对应的数据库表名
