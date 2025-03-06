from Object_Tracking_Commands.Commands.Command import Command
class CameraInvoker:
    def __init__(self):
        self.command = []
    def set_command(self, command : Command):
        self.command.append(command)
    def execute_commands(self):
        for command in self.command:
            command.execute()
        self.command.clear()