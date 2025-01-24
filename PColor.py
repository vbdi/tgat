import builtins
# cput color to bash
class ColorPrint:
    def __init__(self):
        self.HEADER = '\033[95m'
        self.OKBLUE = '\033[44m'
        self.OKGREEN = '\033[42m'
        self.RED = '\033[101m'
        self.WARNING = '\033[45m'
        self.YELLOW = '\033[43m'
        self.WHITE = '\033[95m'
        self.FAIL = '\033[92m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'
        self.DEFAULT = '\033[33m'

    def print(self,*text):
        text_ = [str(t) for t in text]
        color = text_[-1] if len(text_) is not 0 else 'underline'
        text = ''.join(text_[:-1]) if len(text_) is not 0 else '???'

        if color =='green':
            print(f'{self.OKGREEN}{text}{self.ENDC}')
        elif color == 'blue':
            print(f'{self.OKBLUE}{text}{self.ENDC}')
        elif color == 'yellow':
            print(f'{self.YELLOW}{text}{self.ENDC}')
        elif color == 'warning':
            print(f'{self.WARNING}{text}{self.ENDC}')
        elif color =='fail':
            print(f'{self.FAIL}{text}{self.ENDC}')
        elif color=='bold':
            print(f'{self.BOLD}{text}{self.ENDC}')
        elif color == 'underline':
            print(f'{self.UNDERLINE}{text}{self.ENDC}')
        elif color == 'red':
            print(f'{self.RED}{text}{self.ENDC}')
        else:
            text = ''.join(text_[:])
            print(f'{self.DEFAULT}{text}{self.ENDC}')
