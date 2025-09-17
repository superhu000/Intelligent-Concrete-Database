import base64
import json
import io
import math
import os
from datetime import date, datetime

from neo4j import GraphDatabase
from py2neo import Graph
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, FileResponse

from .serializers import ConcreteModelSerializer, ConcreteModelCreateUpdateSerializer
from .utils.dataExtrac import *
from .predict.predict import *
from .predict.test import *
from .models import ConcreteModel
from dvadmin.utils.viewset import CustomModelViewSet

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# 连接Neo4j数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123456789"))
graph = Graph("bolt://localhost:7687", auth=("neo4j", "123456789"))

# 尝试连接并执行一个简单的查询
try:
    with driver.session() as session:
        session.run("RETURN 1")  # 执行一个简单的查询，测试连接
    print("Connected to Neo4j successfully!")
except Exception as e:
    print("Failed to connect to Neo4j:", e)


@csrf_exempt
def handle_excel(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        try:
            df = pd.read_excel(uploaded_file)  # 使用pandas库读取Excel文件数据

            # 开始处理
            feature = df.columns.tolist()
            performance_columns = ['28d抗压强度', '扩散系数', '渗透系数', '电通量']
            # feature = [col for col in feature_all if col not in performance_columns]

            not_empty_count = []
            for column in df.columns:
                not_empty_count.append(df[column].notnull().sum().item())

            max_value = df.max().tolist()
            max_value = [0 if math.isnan(x) or x is None else x for x in max_value]

            min_value = df.min().tolist()
            min_value = [0 if math.isnan(x) or x is None else x for x in min_value]

            average_value = df.mean().round(4).tolist()
            average_value = [0 if math.isnan(x) or x is None else x for x in average_value]

            correlations = []
            # 计算并打印相关系数
            for performance_column in performance_columns:
                correlation = df[feature].corrwith(df[performance_column])
                # correlations.append((performance_column, correlation))
                correlations.append({"performance_column": performance_column, "correlation": correlation})

            row_length = len(correlations[0]['correlation'])
            correlation_values = [[''] * row_length for _ in range(4)]

            for i in range(4):  # 假设您有4个不同的correlation_data_str需要遍历
                correlation_data_str = correlations[i]['correlation']  # 获取第i个相关性数据字符串

                # 遍历correlation_data_str中的每个字符
                for j in range(len(correlation_data_str)):  # 使用len获取字符串长度
                    if pd.isna(correlation_data_str[j]):
                        correlation_values[i][j] = -1
                    else:
                        correlation_values[i][j] = round(correlation_data_str[j], 4)

            # 打印存储的字符列表
            print(correlation_values[0])
            print(not_empty_count)

            print(len(correlation_values[0]))
            print(len(not_empty_count))

            # 构造结果数据
            result = {
                'feature': feature,
                'notEmptyCount': not_empty_count,
                'maxValue': max_value,
                'minValue': min_value,
                'averageValue': average_value,
                'correlation': correlation_values,
                # 'correlation1': correlation_values[1],
                # 'correlation2': correlation_values[2],
                # 'correlation3': correlation_values[3]
            }

            # saveToExcel(df)
            output_filename = os.path.join('./concrete/data', 'output.xlsx')
            df.to_excel(output_filename, index=False)

            return JsonResponse({'success': True, 'result': result})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return JsonResponse({'success': False, 'error': '未上传文件或请求方法不允许'})


@csrf_exempt
def img(request):
    # 绘制图形
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')

    # 保存图像为 BytesIO 对象
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    print('****************************************************************')
    print(img_buffer)
    # 返回图像数据给前端
    return FileResponse(img_buffer, content_type='image/png')


@csrf_exempt
def predict(request):
    if request.method == "POST":
        data = json.loads(request.body)  # 前端向后端发送的参数  data = request.get_json(silent=True)
        feature_cols = data["feature_col"]
        target_cols = data["target_col"]
        select_model = data["model"]
        parameter = data["model_parameter"]
        print(feature_cols)
        print(target_cols)
        print(select_model)
        print(parameter)

        train_x, test_x, train_y, test_y = divide(feature_cols, target_cols)

        if select_model == "RF":
            predictor = RandomForestRegressor()
            # predictor = RandomForestRegressor(n_estimators=int(parameter[0]), max_features=int(parameter[1]),
            #                                   max_depth=int(parameter[2]))
        elif select_model == "MLP":
            predictor = MLPRegressor(hidden_layer_sizes=(int(parameter[0]),), activation=parameter[1],
                                     max_iter=int(parameter[2]))
        elif select_model == "Adaboost":
            predictor = ensemble.AdaBoostRegressor(n_estimators=int(parameter[0]), learning_rate=float(parameter[1]),
                                                   loss=parameter[2])
        elif select_model == "ExtraTree":
            predictor = ExtraTreesRegressor(n_estimators=int(parameter[0]), max_depth=int(parameter[1]))
        elif select_model == "xgboost":
            predictor = XGBRegressor(n_estimators=int(parameter[0]), max_depth=int(parameter[1]),
                                     learning_rate=float(parameter[2]),
                                     booster='gbtree')
        elif select_model == "SVR":
            predictor = SVR(kernel="poly", C=100, degree=3)
        else:
            raise Exception("Model Not Supported")

        rmse, mae, mse, r2, image_buffer = evaluate(feature_cols, target_cols, train_x, train_y, test_x, test_y,
                                                    "Features", predictor)

        # 将预测评分JSON数据转换为Base64编码的字符串
        data = [rmse, mae, mse, r2]
        json_str = json.dumps(data)
        json_data = base64.b64encode(json_str.encode()).decode()
        # 将预测图像数据转换为Base64编码的字符串
        image_data = base64.b64encode(image_buffer.getvalue()).decode()

        # 构建JSON响应，将JSON数据和图像数据一起返回给前端
        response_data = {
            'json_data': json_data,
            'image_data': image_data
        }
        return JsonResponse(response_data, status=200)

        # print(image_buffer)
        # # 返回图像数据给前端
        # return FileResponse(image_buffer, content_type='image/png')


@csrf_exempt
def save_model(request):
    if request.method == "POST":
        data = json.loads(request.body)  # 前端向后端发送的参数  data = request.get_json(silent=True)
        model_name = data["model_name"]
        model_type = data["model_type"]
        input1 = data["input"]
        feature_output = data["feature_output"]
        sampleNum = data["sampleNum"]
        model_parameter = data["model_parameter"]
        model_performance = data["model_performance"]
        create_time = datetime.now()
        print(model_name)
        print(model_type)
        print(model_parameter[0])
        print(model_performance[0])
        print(create_time)

        instance = ConcreteModel(model_name=model_name, model_type=model_type, input=input1,
                                 feature_output=feature_output, sampleNum=sampleNum,
                                 model_R2=model_performance[0]['R2'],
                                 model_RMSE=model_performance[0]['RMSE'],
                                 model_MSE=model_performance[0]['MSE'], model_MAE=model_performance[0]['MAE'],
                                 model_parameter=model_parameter[0], create_time=create_time)
        instance.save()

        return JsonResponse({'status': 'success'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})


@csrf_exempt
def search(request):
    if request.method == "POST":
        data = json.loads(request.body)  # 前端向后端发送的参数  data = request.get_json(silent=True)
        n_node = data["nNodeList"]
        rel = data["relList"]
        m_node = data["mNodeList"]
        print(n_node)
        print(rel)
        print(m_node)

        data_KG = pd.read_excel(r'E:\Study\Project\MaterialGene\Coding\backend\concrete\data\bbb.xlsx', header=0,
                                nrows=120)
        start_KG = 0
        end_KG = 100  # len(invoice_data)
        create_KG(data_KG, start_KG, end_KG)

        dict1 = dataProcessing(n_node, rel, m_node)
        print(dict1)
        return JsonResponse(dict1, safe=False)


def dataProcessing(Nnode, Rel, Mnode):
    nodes = []
    links = []
    nodes_set = []
    if Nnode[0] is None or Nnode[0] == '':
        n_node = ''
    else:
        n_node = ':`' + str(Nnode[0]) + '`'

    if Rel[0] is None or Rel[0] == '':
        rel = ''
    else:
        rel = ':`' + str(Rel[0]) + '`'

    if Mnode[0] is None or Mnode[0] == '':
        m_node = ''
    else:
        m_node = ':`' + str(Mnode[0]) + '`'

    print("n_node: " + n_node + " rel: " + rel + " m_node: " + m_node)

    with driver.session() as session:
        result = session.run(
            'MATCH (n' + n_node + ')-[r' + rel + ']->(m' + m_node + ') RETURN ' +
            'id(n) as source, labels(n) as source_labels, properties(n) as source_properties, ' +
            'id(m) as target, labels(m) as target_labels, properties(m) as target_properties, ' +
            'id(r) as link, type(r) as r_type, properties(r) as r_properties ' +
            'LIMIT 100')

        for record in result:
            nodes.append({"id": record['source'], "label": record['source_labels'][0],
                          "properties": record['source_properties']})
            nodes.append({"id": record['target'], "label": record['target_labels'][0],
                          "properties": record['target_properties']})
            links.append({"source": record['source'], "target": record['target'], "type": record['r_type'],
                          "properties": record['r_properties']})

        for i in nodes:
            if i not in nodes_set:
                nodes_set.append(i)
        result_dict = dict(zip(['nodes', 'links'], [nodes_set, links]))
        # print(result_dict)
        return result_dict


@csrf_exempt
def search_correlation(request):
    if request.method == "POST":
        data = json.loads(request.body)  # 前端向后端发送的参数  data = request.get_json(silent=True)
        num_1 = data["num"]
        print(num_1)

        df = pd.read_excel(r"E:\Study\Project\MaterialGene\Coding\backend\concrete\data\output.xlsx")
        original_shape, reduced_shape, image_buffer = correlation(df, num_1)

        # 将预测评分JSON数据转换为Base64编码的字符串
        data = [original_shape, reduced_shape]
        json_str = json.dumps(data)
        json_data = base64.b64encode(json_str.encode()).decode()
        # 将预测图像数据转换为Base64编码的字符串
        image_data = base64.b64encode(image_buffer.getvalue()).decode()

        # 构建JSON响应，将JSON数据和图像数据一起返回给前端
        response_data = {
            'json_data': json_data,
            'image_data': image_data
        }
        return JsonResponse(response_data, status=200)


@csrf_exempt
def search_correlation1(request):
    if request.method == "POST":
        data = json.loads(request.body)  # 前端向后端发送的参数  data = request.get_json(silent=True)

        data_scaled, data_pca, image_buffer, image_buffer1 = correlation1()

        # 将预测评分JSON数据转换为Base64编码的字符串
        data = [data_scaled, data_pca]
        json_str = json.dumps(data)
        json_data = base64.b64encode(json_str.encode()).decode()
        # 将预测图像数据转换为Base64编码的字符串
        image_data = base64.b64encode(image_buffer.getvalue()).decode()
        image_data1 = base64.b64encode(image_buffer1.getvalue()).decode()

        # 构建JSON响应，将JSON数据和图像数据一起返回给前端
        response_data = {
            'json_data': json_data,
            'image_data': image_data,
            'image_data1': image_data1
        }
        return JsonResponse(response_data, status=200)


# 在concrete_data包里注册URL
class ConcreteModelViewSet(CustomModelViewSet):
    """
    list:查询
    create:新增
    update:修改
    retrieve:单例
    destroy:删除
    """
    queryset = ConcreteModel.objects.all()
    serializer_class = ConcreteModelSerializer
    create_serializer_class = ConcreteModelCreateUpdateSerializer
    update_serializer_class = ConcreteModelCreateUpdateSerializer
    filter_fields = ['model_id', 'feature_output', 'model_R2']
    search_fields = ['model_id', 'feature_output', 'model_R2']
    ordering_fields = ['model_id', 'feature_output', 'model_R2']
