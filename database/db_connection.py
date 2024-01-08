import pyodbc as db
import os
from datetime import datetime
import socket

#region Database Functions
def database_connection():     
    try:
        cnxn = db.connect(DRIVER='{SQL Server}',
                            SERVER='GUNCEL-WS',
                            DATABASE='LeoFabricVisionResults',
                            UID='leofabricvision',
                            PWD='hetket')
        if cnxn !=None:
            return  cnxn.cursor()        

    except Exception as e:
        print('DATABASE CONNECTION FAILED-sema',str(e))
        
cursor=database_connection()
print('ok:',cursor)

def insert_data(image_path, image_name, model_name, prediction_time,label, prediction_score, ok_nok):
    try:
        conn = database_connection()
        cursor=conn.cursor()
        time_now=datetime.now()
        formatted_date = time_now.strftime("%Y-%m-%d %H:%M:%S")

        sql_command = f"INSERT INTO LeoFabricVisionTestResult (AddDate,image_path, image_name, model_name, AUC, accuracy, prediction_time,label, prediction_score, ok_nok) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        sql_value=[formatted_date,str(image_path),str(image_name),str(model_name),str(prediction_time),int(label),str(prediction_score),str(ok_nok)]
        cursor.execute(sql_command,sql_value)
        cursor.commit()

    except Exception as e:
        print("INSERT ERROR", str(e))

    finally:
        cursor.close()