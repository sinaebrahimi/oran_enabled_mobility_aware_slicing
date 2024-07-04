# Class of different styles for printing
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    time_string = ""
    if hours > 0:
        time_string += f"{hours} hour{'s' if hours > 1 else ''} "
    if minutes > 0:
        time_string += f"{minutes} minute{'s' if minutes > 1 else ''} "
    if seconds > 0:
        time_string += f"{seconds} second{'s' if seconds > 1 else ''}"

    return time_string.strip()