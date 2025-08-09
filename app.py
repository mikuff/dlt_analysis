import logging

from typing import Iterable
from textual.app import App, SystemCommand
from textual.screen import Screen
from textual.widgets import Header, Footer,Static
from textual.containers import Vertical
from pages.home import HomeScreen
from pages.data import DataScreen
from pages.analysis import AnalysisScreen


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("./application.log", encoding='utf-8')
    ]
)

class Application(App):

    CSS_PATH = "styles.tcss"  

    def compose(self):
        yield Header()
        yield Footer()

    # 项目启动指向页面
    def on_mount(self):
        self.push_screen(AnalysisScreen())
    
    # 自定义命令列表
    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield SystemCommand(
            title="首页",
            help="首页",
            callback=self.action_home_screen,
        )
        yield SystemCommand(
            title="数据",
            help="数据",
            callback=self.action_data_screen,
        )
        yield SystemCommand(
            title="训练",
            help="训练",
            callback=self.action_analysis_screen,
        )
        yield SystemCommand(
            title="退出",
            help="退出",
            callback=self.action_quit,
        )
    
    # 切换至首页
    async def action_home_screen(self) -> None:
        await self.push_screen(HomeScreen())

    # 切换至数据页面
    async def action_data_screen(self) -> None:
        await self.push_screen(DataScreen())

    # 切换至分析页面
    async def action_analysis_screen(self) -> None:
        await self.push_screen(AnalysisScreen())

    # 退出
    def action_quit(self) -> None:
        self.exit()

if __name__ == "__main__":
    Application().run(mouse=True)
