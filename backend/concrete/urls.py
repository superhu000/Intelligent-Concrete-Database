from django.urls import path
from .views import search, img, predict, handle_excel, save_model, search_correlation,search_correlation1
from .viewKG import handle_excel_KG, search_create_KG

app_name = 'concrete'

urlpatterns = [
    path('handle_excel', handle_excel),
    path('handle_excel_KG', handle_excel_KG),
    path('search', search),
    path('search_create_KG', search_create_KG),
    path('predict', predict),
    path('search_correlation', search_correlation),
    path('search_correlation1', search_correlation1),
    path('save_model', save_model),
    path('img', img),
]
