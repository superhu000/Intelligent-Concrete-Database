# -*- coding: utf-8 -*-
import pandas as pd
from concrete.utils.dataToNeo4j import *


def data_ColExtrac(property_list, invoice_data, start_row, end_row):
    """抽取列节点数据"""
    data_list = []
    for k in range(len(property_list)):
        feature_list = []
        for i in range(start_row, end_row):
            cell_value = invoice_data[invoice_data.columns[property_list[k][1]]][i]
            if cell_value is not None:
                cell_value = (str(cell_value))[:7]
            else:
                cell_value = str(0)
            feature_list.append(cell_value)
            feature_list = list(set(feature_list))

        _id = property_list[k][0]

        data_list.append({"id": _id, "feature": feature_list})

    return data_list


def data_RowExtrac(property_list, invoice_data, start_row, end_row):
    """
    抽取行节点数据:[[('name', 0), ('C_CaO', 2), ('C_SiO2', 3)...], ['18', '61.9', '20.2', '4.7', '2.6', '3.0']...]
    """
    data_list = [property_list]
    for i in range(start_row, end_row):
        feature_list = []
        for k in range(len(property_list)):
            cell_value = invoice_data[invoice_data.columns[property_list[k][1]]][i]
            if cell_value is not None:
                cell_value = (str(cell_value))[:7]
            else:
                cell_value = str(0)
            feature_list.append(cell_value)

        data_list.append(feature_list)
    # print(data_list)
    return data_list


def rel_extraction(property_list, invoice_data, start_row, end_row):
    """联系数据抽取"""
    links_dict = {}
    for k in range(len(property_list)):
        feature_list = []
        for i in range(start_row, end_row):
            cell_value = invoice_data[invoice_data.columns[property_list[k][1]]][i]
            if cell_value is not None:
                cell_value = str(cell_value)
            else:
                cell_value = str(0)
            feature_list.append(cell_value)

        _id = property_list[k][0]
        links_dict[property_list[k][0]] = feature_list

    # links_dict.append({"id": "rel","feature":rel})
    df_data = pd.DataFrame(links_dict)
    # print(df_data)
    return df_data


def create_KG(invoice_data, start_row, end_row):
    create_data = DataToNeo4j()
    create_data.create_node(data_ColExtrac(property_Alone, invoice_data, start_row, end_row))  # 创建单属性的节点
    create_data.create_node1("水泥", data_RowExtrac(property_Cement, invoice_data, start_row, end_row))  # 创建水泥节点
    create_data.create_node1("粉煤灰", data_RowExtrac(property_FlyAsh, invoice_data, start_row, end_row))  # 创建粉煤灰节点
    create_data.create_node1("矿渣", data_RowExtrac(property_Slag, invoice_data, start_row, end_row))  # 创建矿渣节点
    create_data.create_node1("硅灰", data_RowExtrac(property_SF, invoice_data, start_row, end_row))  # 创建硅灰节点
    create_data.create_rel(rel_extraction(property_All, invoice_data, start_row, end_row))  # 创建关系
    print("数据已成功插入 Neo4j 数据库!!!")

# data = pd.read_excel(r'F:\从零开始的研究生生活\研究生项目\混凝土预测系统\5_1\backend\concrete\data\bbb.xlsx', header=0,
#                      nrows=120)
# start = 0
# end = 100  # len(invoice_data)
# create_KG(data, start, end)
