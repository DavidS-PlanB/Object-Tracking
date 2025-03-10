import os
import os.path
import urllib.request
import dotenv

class SmartDownloader():
    def __init__(self):
        dotenv.load_dotenv()

    def Check_for_Yolomodel(self):
        filename= os.getenv("Weights")
        folder= os.getenv("FolderPath")
        url = os.path.join(os.getenv("DownloadPath"),filename)
        file_path = os.path.join(folder,filename)
        
        if os.path.exists(file_path):
            print("datein ist vorhanden")
            return
        print("datei ist nicht vorhanden und wird runtergeladen")
        urllib.request.urlretrieve(url, file_path)
        print("Download Fertiggestellt")
