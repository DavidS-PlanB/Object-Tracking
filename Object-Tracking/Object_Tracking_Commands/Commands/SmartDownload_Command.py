from Object_Tracking_Commands.Commands.Command import Command
from Object_Tracking_Core.Services.SmartDownloader import SmartDownloader

class SmartDownload_Command(Command):
    def __init__(self, SmartDonwloader_service: SmartDownloader):
        self.service = SmartDonwloader_service

    def execute(self):
        SmartDonwloader_service = self.service
        SmartDonwloader_service.Check_for_Yolomodel()
    