# Concrete-backend

[![img](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitee.com/liqianglog/django-vue-admin/blob/master/LICENSE)  [![img](https://img.shields.io/badge/python-%3E=3.8.x-green.svg)](https://python.org/)  [![PyPI - Django Version badge](https://img.shields.io/badge/django%20versions-4.2-blue)](https://docs.djangoproject.com/zh-hans/4.2/) [![img](https://img.shields.io/badge/node-%3E%3D%2014.0.0-brightgreen)](https://nodejs.org/zh-cn/)

## 平台简介

Concrete项目前后端分离，backend后端采用django + django-rest-framework

## 准备工作

~~~
Python >= 3.8.0 (推荐3.11版本)
django >= 4.2.7
nodejs >= 14.0 (推荐最新)
Mysql >= 5.7.0 (可选，推荐8.0版本)
Redis(可选，最新版)
~~~

## 知识图谱配置

~~~
1.安装Neo4j Desktop

2.Add建一个Local DBMS(concrete),在后端concrete文件夹的setting和views的数据库url、账号和密码改成自己的

3.将数据文件bbb.xlsx放在concrete/utils/data/中

4.运行 dataExtrac.py(注意最下方的注释部分先取消注释在运行，只需运行一次，运行前需打开运行Neo4j Desktop创建的数据库)


Neo4j 

~~~

## 前端-web♝（建议使用IDEA）

```bash
# 克隆项目
git clone https://gitee.com/luo-zixuan/web.git

# 进入项目目录
cd web

# 安装依赖
npm install --registry=https://registry.npm.taobao.org

# 启动服务
npm run dev
# 浏览器访问 http://localhost:8080
# .env.development 文件中可配置启动端口等参数
# 构建生产环境
# npm run build
```

## 后端💈（建议使用pyCharm）

~~~bash
1. https://gitee.com/luo-zixuan/backend.git

2. 在项目根目录中，复制 ./conf/env.example.py 文件为一份新的到 ./conf 文件夹下，并重命名为 env.py

3. 在 env.py 中配置数据库信息
    mysql数据库名：django-vue-admin
	mysql数据库版本建议：8.0
	mysql数据库字符集：utf8mb4
	
4. 安装依赖环境
	pip3 install -r requirements.txt
	
5. 执行迁移命令：
	python manage.py makemigrations
	python manage.py migrate
	
6. 初始化数据
	python manage.py init
	
7. 初始化省市县数据:
	python manage.py init_area
	
8. 启动项目
	python manage.py runserver 0.0.0.0:8000
	
9. mysql建议使用Navicat16，在运行完1-8步骤后在Navicat的django-vue-admin点击右键运行SQL文件，sql文件在./conf文件夹下
~~~

### 访问项目

- 访问地址：[http://localhost:8080](http://localhost:8080) (默认为此地址，如有修改请按照配置文件)
- 账号：`superadmin` 密码：`admin123456`

## 演示图✅

![image-01](https://images.gitee.com/uploads/images/2022/0530/234137_b58c8f98_5074988.png)

