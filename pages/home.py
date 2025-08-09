from textual.screen import Screen
from textual.widgets import Header,Footer,Static
from textual.containers import Vertical
from textual.app import ComposeResult
from utils.cutil import read_text

class HomeScreen(Screen):

    # 渲染 docs/下的文件
    def compose(self) -> ComposeResult:
        intro_text = read_text("docs/项目说明.txt")
        datasource_text = read_text("docs/数据来源.txt")
        algorithm_text = read_text("docs/分析算法.txt")

        yield Header()
        yield Vertical(
            Static(f"{intro_text}", id="home_intro"),
            Static(f"{datasource_text}", id="home_datasource"),
            Static(f"{algorithm_text}", id="home_algorithm"),
            id="home"
        )
        yield Footer()

