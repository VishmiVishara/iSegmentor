
from django.urls import path, re_path
from app import views

urlpatterns = [

    # The home page
    path('', views.index, name='home'),
    path('download/', views.download, name='download'),
    re_path(r'^live-chart.html', views.loadChart, name='live'),
    re_path(r'^live-chart-dis.html', views.loadChartDis, name='livedis'),
    re_path(r'^search.html', views.search,  name="search"),
    re_path(r'^train.html', views.train,  name="train"),
    re_path(r'^evaluate.html', views.evaluate,  name="evaluate"),
 

    # # Matches any html file
    # re_path(r'^.*\.*', views.pages, name='pages'),

]
