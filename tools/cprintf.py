from colorama import Fore, Style, init

# 初始化 colorama 以支持颜色输出
init(autoreset=True)


def cprintf(style, format_str, *args):
    """
    用于在 Python 控制台中实现类似 MATLAB cprintf 的功能。

    参数:
        style (str): 文本样式，例如 'text', 'cyan', 'red', 'green' 等。
                     特殊样式包括 '*bold'（加粗）, '_underline'（下划线）等。
        format_str (str): 格式化字符串，与 Python 的 print 格式类似。
        *args: 格式化字符串中的变量。
    """
    # 处理样式
    underline = False
    bold = False

    # 判断是否有下划线或加粗样式
    if style.startswith('-') or style.startswith('_'):
        underline = True
        style = style.lstrip('-_')

    if style.startswith('*'):
        bold = True
        style = style.lstrip('*')

    # 定义颜色映射
    color_map = {
        'text': Fore.RESET,
        'red': Fore.RED,
        'green': Fore.GREEN,
        'blue': Fore.BLUE,
        'cyan': Fore.CYAN,
        'magenta': Fore.MAGENTA,
        'yellow': Fore.YELLOW,
        'white': Fore.WHITE,
        'black': Fore.BLACK,
        'err': Fore.RED + Style.BRIGHT,  # 错误文本为亮红色
        'hyper': Fore.CYAN + Style.BRIGHT  # 超链接文本为亮青色
    }

    # 获取对应的颜色
    color = color_map.get(style.lower(), Fore.RESET)
    text = format_str % args

    # 加粗处理
    if bold:
        color += Style.BRIGHT
    # 下划线处理
    if underline:
        text = f"\033[4m{text}\033[0m"  # 使用 ANSI 转义序列添加下划线

    # 输出格式化文本
    print(color + text)


# 演示
if __name__ == "__main__":
    cprintf('text', "这是普通的黑色文本。\n")
    cprintf('cyan', "这是青色文本。\n")
    cprintf('-green', "这是带下划线的绿色文本。\n")
    cprintf('*red', "这是加粗的红色文本。\n")
    cprintf('*-blue', "这是加粗且带下划线的蓝色文本。\n")