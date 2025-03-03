from django.urls import path
from .views import generate_skin_report,myself,chat_api,page

urlpatterns = [
    path('generate_skin_report', generate_skin_report, name='generate_skin_report'),
    path('', myself, name='home'),
    path("chat-api/", chat_api, name="chat_api"),
    path("404.html/", page, name="404"),


]
